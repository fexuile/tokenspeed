#!/usr/bin/env bash
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

set -eo pipefail
if [[ $# != 4 ]]; then
    echo "Usage: bash serve-qwen35.sh <model-path> <1|2> <physical-chip-ids> <enabled|disabled>" >&2
    exit 2
fi
MODEL="$1"
TP="$2"
CHIPS="$3"
PREFIX_CACHE="$4"
if [[ "$PREFIX_CACHE" != enabled && "$PREFIX_CACHE" != disabled ]]; then
    echo "Select prefix caching explicitly: enabled or disabled" >&2
    exit 2
fi
if [[ ! "$TP" =~ ^[12]$ || ! "$CHIPS" =~ ^[0-9]+(,[0-9]+)?$ ]]; then
    echo "Expected TP=1 or 2 and a comma-separated physical chip list" >&2
    exit 2
fi
IFS=, read -ra CHIP_IDS <<< "$CHIPS"
if [[ ${#CHIP_IDS[@]} != "$TP" || ( "$TP" == 2 && "${CHIP_IDS[0]}" == "${CHIP_IDS[1]}" ) ]]; then
    echo "Select exactly TP distinct physical chips" >&2
    exit 2
fi
export REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${PYTHON_VENV:-}" ]]; then
    source "$PYTHON_VENV/bin/activate"
fi
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH="$REPO_ROOT/python:$REPO_ROOT/tokenspeed-kernel/python:$REPO_ROOT/tokenspeed-kernel-npu/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENSPEED_KERNEL_BACKEND=npu
export ASCEND_RT_VISIBLE_DEVICES="$CHIPS"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
python "$REPO_ROOT/test/ascend_hw/tp_validation.py" inspect --model "$MODEL" --tp "$TP"
npu-smi info
command=(python -B -m tokenspeed.cli serve "$MODEL"
    --device npu --dtype bfloat16 --world-size "$TP" --attn-tp-size "$TP"
    --dense-tp-size "$TP" --lm-head-tp-size "$TP" --data-parallel-size 1
    --language-model-only --kv-cache-dtype auto --attention-backend mha
    --sampling-backend greedy --mamba-ssm-dtype float32 --enforce-eager
    --disable-prefill-graph --disable-pdl --disable-autotune
    --disable-kvstore --retraction-snapshot-ratio 0
    --max-model-len 4096 --max-num-seqs 2 --max-total-tokens "${MAX_TOTAL_TOKENS:-16384}"
    --chunked-prefill-size 4096 --prefix-granularity 128
    --host 127.0.0.1 --port "${SERVER_PORT:-31905}")
if [[ "$PREFIX_CACHE" == disabled ]]; then
    command+=(--disable-prefix-caching)
else
    command+=(--enable-prefix-caching)
fi
printf 'LAUNCH %q ' "${command[@]}"
printf '\n'
exec "${command[@]}"
