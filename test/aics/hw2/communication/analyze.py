# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Derive per-forward compute/collective unions from CANN device exports."""
import bisect
import collections
import csv
import json
from pathlib import Path
import statistics
from aics.hw2.communication.intervals import account, intersect, measure, merge

ROOT = Path(__file__).resolve().parent
TP_DIRS = {1: 'tp1-final', 2: 'tp2-clean'}
CORE_TYPES = {'AI_CORE', 'AI_VECTOR_CORE', 'MIX_AIC', 'MIX_AIV',
              'KERNEL_AICORE', 'KERNEL_AIVEC', 'KERNEL_MIX_AIC', 'KERNEL_MIX_AIV'}


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_profile(folder):
    trace_paths = list(folder.rglob('trace_view.json'))
    if not trace_paths:
        trace_paths = list(folder.rglob('msprof_*.json'))
    assert len(trace_paths) == 1, trace_paths
    data = json.loads(trace_paths[0].read_text())
    events = data if isinstance(data, list) else data['traceEvents']
    hardware_pids = {e['pid'] for e in events if e.get('name') == 'process_name'
                     and e.get('args', {}).get('name') == 'Ascend Hardware'}
    assert len(hardware_pids) == 1, hardware_pids
    operations = []
    for event in events:
        if event.get('ph') != 'X' or event['pid'] not in hardware_pids:
            continue
        start = float(event['ts'])
        duration = float(event['dur'])
        operations.append(dict(start=start, end=start+duration,
                               name=event['name'], kind=event.get('args',{}).get('Task Type'),
                               op_type=event['name'], device=event['pid']))
    communication = []
    waits = []
    for event in events:
        if event.get('ph') != 'X':
            continue
        name = event['name']
        start = float(event['ts'])
        end = start + float(event['dur'])
        if name.startswith('hcom_'):
            communication.append(dict(start=start, end=end, name=name, args=event.get('args', {})))
        elif name == 'Notify_Wait':
            waits.append((start, end))
    waits.extend((o['start'],o['end']) for o in operations if 'WAIT' in str(o['kind']))
    return operations, communication, waits, trace_paths[0]


def select_forwards(folder, rank):
    workload = json.loads((folder/'workload.json').read_text())
    forwards = load_jsonl(folder/f'rank{rank}-forwards.jsonl')
    diagnostics = {r['step']: r for r in load_jsonl(folder/f'rank{rank}-synchronized-forwards.jsonl')}
    selected = []
    for case in workload['rows']:
        if case['phase'] != 'measured':
            continue
        candidates = [i for i, f in enumerate(forwards)
                      if f['extends'] == case['batch'] and f['bs'] == case['batch']
                      and f['tokens'] == case['input_length']*case['batch']
                      and case['start_ns'] <= f['start_ns'] < case['end_ns']]
        assert len(candidates) == 1, (case, candidates)
        first = candidates[0]
        sampled = forwards[first:first+8]
        assert len(sampled) == 8
        assert all(f['bs'] == case['batch'] and f['extends'] == 0 for f in sampled[1:])
        for index, forward in enumerate(sampled):
            selected.append(dict(diagnostics[forward['step']], input_length=case['input_length'],
                                 batch=case['batch'], phase='prefill' if index==0 else 'decode'))
    assert len(selected) == 32
    return selected


def analyze_run(tp, rank):
    folder = ROOT / TP_DIRS[tp]
    operations, communication, waits, trace = load_profile(folder/f'profile-rank{rank}')
    selected = select_forwards(folder, rank)
    all_forwards = load_jsonl(folder/f'rank{rank}-forwards.jsonl')
    gemms = sorted((op for op in operations if op['name'] == '_ascend_mm_kernel'), key=lambda op:op['start'])
    assert len(gemms) == len(all_forwards)*129, (len(gemms),len(all_forwards))
    step_to_index = {f['step']:i for i,f in enumerate(all_forwards)}
    embeddings = sorted((c for c in communication if 'allReduce' in c['name'] and c['args'].get('data_type')=='BFP16'),key=lambda c:c['start'])
    if tp > 1:
        assert len(embeddings) == len(all_forwards)
        assert all(int(c['args']['count']) == f['tokens']*4096 for c,f in zip(embeddings,all_forwards,strict=True))
    rows = []
    snippets = []
    for forward in selected:
        index = step_to_index[forward['step']]
        forward_gemms = gemms[index*129:(index+1)*129]
        # Identical model boundaries at both TP sizes, measured on the device:
        # first input projection begins -> LM-head GEMM completes. This avoids
        # clipping asynchronous collectives with a CPU submission timestamp.
        window = (forward_gemms[0]['start'], forward_gemms[-1]['end'])
        local = [op for op in operations if op['start'] >= window[0] and op['end'] <= window[1]]
        assert local, (forward, trace)
        compute = [(op['start'],op['end']) for op in local if op['kind'] in CORE_TYPES
                   and 'hccl' not in op['name'].lower() and not op['name'].startswith('hcom')]
        collectives = [c for c in communication if window[0] <= c['start'] < window[1]]
        if tp > 1:
            end = embeddings[index+1]['start'] if index+1<len(embeddings) else float('inf')
            entire = [c for c in communication if embeddings[index]['start'] <= c['start'] < end]
            counts = collections.Counter(c['name'].split('__')[0] for c in entire)
            assert counts == {'hcom_allGather':129, 'hcom_allReduce':1, 'hcom_broadcast':1}, (forward,counts)
            assert len(collectives)==128, (forward,len(collectives))
            assert all(c['args'].get('data_type') in ('BFP16','BF16') for c in collectives)
            assert sum(int(c['args']['count'])*2 for c in collectives) == 786432*forward['tokens']
        values = account(compute, [(c['start'],c['end']) for c in collectives], waits, window)
        wait_only = measure(intersect(waits,[window])) - measure(intersect(waits,compute+[(c['start'],c['end']) for c in collectives]))
        values['wait_only_us'] = wait_only
        values['other_gap_us'] = values['uncovered_us'] - wait_only
        values['gemm_us'] = measure([(op['start'],op['end']) for op in forward_gemms])
        assert values['uncovered_us'] >= -1e-3
        values.update(tp=tp, rank=rank, input_length=forward['input_length'], batch=forward['batch'],
                      phase=forward['phase'], step=forward['step'], tokens=forward['tokens'],
                      span_us=window[1]-window[0], event_us=1000*forward['event_ms'],
                      compute_count=len(compute), collective_count=len(collectives),
                      allocated_bytes=forward['allocated_bytes'],reserved_bytes=forward['reserved_bytes'])
        rows.append(values)
        if forward['input_length'] == 2048 and forward['batch'] == 2 and forward['phase'] == 'prefill':
            snippets.append(dict(rank=rank, phase='prefill', origin_us=window[0], window_us=values['span_us'],
                                 compute=[(s-window[0],e-window[0]) for s,e in merge(compute)],
                                 waits=[(s-window[0],e-window[0]) for s,e in intersect(waits,[window])],
                                 communication=[dict(name=c['name'],start=c['start']-window[0],end=c['end']-window[0],args=c['args']) for c in collectives]))
        if forward['input_length'] == 128 and forward['batch'] == 2 and forward['phase'] == 'decode' and not any(s['phase']=='decode' for s in snippets):
            snippets.append(dict(rank=rank, phase='decode', origin_us=window[0], window_us=values['span_us'],
                                 compute=[(s-window[0],e-window[0]) for s,e in merge(compute)],
                                 waits=[(s-window[0],e-window[0]) for s,e in intersect(waits,[window])],
                                 communication=[dict(name=c['name'],start=c['start']-window[0],end=c['end']-window[0],args=c['args']) for c in collectives]))
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row['input_length'],row['batch'],row['phase'])].append(row)
    summaries = []
    metrics = ('span_us','event_us','compute_us','communication_us','communication_wait_us','overlap_us','uncovered_us','wait_only_us','other_gap_us','gemm_us')
    for (length,batch,phase), group in sorted(groups.items()):
        summary = dict(tp=tp,rank=rank,input_length=length,batch=batch,phase=phase,forwards=len(group))
        for metric in metrics:
            summary[metric] = statistics.mean(r[metric] for r in group)
        summary['communication_percent'] = 100*summary['communication_us']/summary['span_us']
        summary['span_stdev_us'] = statistics.stdev(r['span_us'] for r in group) if len(group)>1 else 0
        summaries.append(summary)
    return rows,summaries,snippets


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='Directory containing tp1 and tp2 profiler outputs')
    parser.add_argument('--tp1-dir', default='tp1-final')
    parser.add_argument('--tp2-dir', default='tp2-clean')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    global ROOT, TP_DIRS
    ROOT = root
    TP_DIRS = {1: args.tp1_dir, 2: args.tp2_dir}
    rows,summary,snippets = [],[],[]
    for tp,rank in ((1,0),(2,0),(2,1)):
        r,s,t = analyze_run(tp,rank)
        rows.extend(r);summary.extend(s)
        if tp == 2:
            snippets.extend(t)
    # Compare every requested output from both freshly profiled deployments.
    a=json.loads((root/args.tp1_dir/'workload.json').read_text())
    b=json.loads((root/args.tp2_dir/'workload.json').read_text())
    assert len(a['rows']) == len(b['rows']) == 8
    assert all(x['output_ids']==y['output_ids'] for x,y in zip(a['rows'],b['rows'],strict=True))
    for name,records in (('forwards',rows),('summary',summary)):
        (output/f'{name}.json').write_text(json.dumps(records,indent=2))
        with (output/f'{name}.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (output/'timeline-snippets.json').write_text(json.dumps(snippets,indent=2))
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
