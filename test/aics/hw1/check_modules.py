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

"""Ascend numerical checks against independent CPU FP32 calculations."""

import json
import math

import torch
import torch.nn.functional as F
import torch_npu
from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill, gdn_decode_step
from tokenspeed_kernel.ops.attention.gdn.triton import (
    build_causal_conv1d_prefill_metadata,
)
from tokenspeed_kernel.ops.conv import causal_conv1d_fn, causal_conv1d_update
from tokenspeed_kernel.ops.layernorm import gated_rmsnorm
from tokenspeed_kernel_npu.ops.mha import mha_prefill

from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm

torch.set_num_threads(8)
torch.manual_seed(20261023)
records = []


def compare(name, actual, reference, atol, rtol):
    a, b = actual.detach().float().cpu(), reference.detach().float().cpu()
    error = (a - b).abs()
    result = dict(
        name=name,
        shape=list(a.shape),
        atol=atol,
        rtol=rtol,
        max_abs=float(error.max()) if error.numel() else 0,
        rmse=float(error.square().mean().sqrt()) if error.numel() else 0,
    )
    try:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        result["passed"] = True
    except AssertionError as exc:
        result.update(passed=False, error=str(exc)[:1500])
    records.append(result)
    print(json.dumps(result), flush=True)


def rand(*shape, scale=1):
    return (torch.randn(*shape) * scale).bfloat16()


def recurrence(q, k, v, g, beta, state, norm_round):
    q, k, v = q.float(), k.float(), v.float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True).clamp_min(1e-6))
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True).clamp_min(1e-6))
    if norm_round:
        q, k = q.bfloat16().float(), k.bfloat16().float()
    h = state.float().clone()  # [B,H,V,K]
    outs = []
    for t in range(q.shape[1]):
        h = h * g[:, t].exp()[..., None, None]
        delta = (v[:, t] - torch.einsum("bhvk,bhk->bhv", h, k[:, t])) * beta[
            :, t, :, None
        ]
        h = h + delta[..., None] * k[:, t, :, None, :]
        outs.append(torch.einsum("bhvk,bhk->bhv", h, q[:, t]) / math.sqrt(q.shape[-1]))
    return torch.stack(outs, dim=1), h


@torch.inference_mode()
def main():
    for rows in (0, 1, 65):
        x, r, w = rand(rows, 1024), rand(rows, 1024), rand(1024, scale=0.1)
        norm = GemmaRMSNorm(1024, eps=1e-6).to(device="npu", dtype=torch.bfloat16)
        norm.weight.copy_(w.to("npu"))
        for residual in (False, True):
            total = (x + r) if residual else x
            ref = (
                total.float()
                * torch.rsqrt(total.float().square().mean(-1, keepdim=True) + 1e-6)
                * (1 + w.float())
            )
            got = norm(x.to("npu"), r.to("npu") if residual else None)
            if residual:
                compare(f"gemma_residual_{rows}", got[1], total, 0, 0)
                got = got[0]
            # CPU/NPU FP32 reductions can cross a BF16 rounding boundary.
            compare(f"gemma_{rows}_{residual}", got, ref.bfloat16(), 1e-6, 2**-7)
    for rows in (1, 65):
        x, z, w = rand(rows, 2048), rand(rows, 2048), rand(128, scale=0.1)
        got = gated_rmsnorm(
            x.to("npu"),
            w.to("npu"),
            z.to("npu"),
            eps=1e-6,
            group_size=128,
            norm_before_gate=True,
            sigmoid_gate=False,
            fp8_scale=None,
            weights_independent=False,
        )
        xx = x.float().view(rows, 16, 128)
        ref = (
            xx * torch.rsqrt(xx.square().mean(-1, keepdim=True) + 1e-6) * w.float()
        ).reshape(rows, -1) * F.silu(z.float())
        compare(f"gated_norm_{rows}", got, ref.bfloat16(), 2**-7, 2**-7)
    for length in (1, 63, 64, 65, 129):
        q, k, v = rand(length, 8, 256), rand(length, 2, 256), rand(length, 2, 256)
        bounds = torch.tensor([0, length], dtype=torch.int32, device="npu")
        got = mha_prefill(
            q.to("npu"), k.to("npu"), v.to("npu"), bounds, [0, length], length
        )
        scores = (
            torch.einsum("qhd,khd->hqk", q.float(), k.float().repeat_interleave(4, 1))
            / 16
        )
        scores.masked_fill_(
            torch.ones(length, length, dtype=torch.bool).triu(1), -torch.inf
        )
        ref = torch.einsum(
            "hqk,khd->qhd", scores.softmax(-1), v.float().repeat_interleave(4, 1)
        )
        compare(f"mha_D256_L{length}", got, ref.bfloat16(), 0.02, 0.02)
    for length in (1, 63, 64, 65, 130, 257):
        q, k, v = (
            rand(1, length, 16, 128),
            rand(1, length, 16, 128),
            rand(1, length, 16, 128, scale=0.5),
        )
        g = F.logsigmoid(torch.randn(1, length, 16))
        beta = torch.rand(1, length, 16).bfloat16()
        h = torch.randn(1, 16, 128, 128) * 0.01
        result = gdn_chunk_prefill(
            q.to("npu"),
            k.to("npu"),
            v.to("npu"),
            g.to("npu"),
            beta.to("npu"),
            scale=128**-0.5,
            initial_state=h.to("npu"),
            cu_seqlens=torch.tensor([0, length], dtype=torch.int32, device="npu"),
            qk_l2norm=True,
            output_final_state=True,
            solution="triton",
        )
        ref, hs = recurrence(q, k, v, g, beta.float(), h, True)
        compare(f"gdn_prefill_L{length}", result.out, ref.bfloat16(), 0.02, 0.02)
        compare(f"gdn_prefill_state_L{length}", result.final_state, hs, 0.03, 0.02)
    for state_dtype in (torch.float32, torch.bfloat16):
        q, k, v = (
            rand(2, 32, 16, 128),
            rand(2, 32, 16, 128),
            rand(2, 32, 16, 128, scale=0.5),
        )
        a, b = torch.randn(2, 32, 16), torch.randn(2, 32, 16)
        alog, bias = torch.randn(16) * 0.2, torch.randn(16) * 0.2
        initial = (torch.randn(5, 16, 128, 128) * 0.01).to(state_dtype)
        pool = initial.to("npu")
        refstate = initial[[1, 3]].float()
        g = -alog.exp() * F.softplus(a + bias)
        refs = []
        outputs = []
        for t in range(32):
            out = gdn_decode_step(
                q[:, t : t + 1].to("npu"),
                k[:, t : t + 1].to("npu"),
                v[:, t : t + 1].to("npu"),
                A_log=alog.to("npu"),
                a=a[:, t : t + 1].to("npu"),
                dt_bias=bias.to("npu"),
                b=b[:, t : t + 1].to("npu"),
                initial_state=pool,
                initial_state_indices=torch.tensor(
                    [1, 3], dtype=torch.int32, device="npu"
                ),
                scale=128**-0.5,
                use_qk_l2norm=True,
                solution="triton",
            )
            ref, refstate = recurrence(
                q[:, t : t + 1],
                k[:, t : t + 1],
                v[:, t : t + 1],
                g[:, t : t + 1],
                b[:, t : t + 1].sigmoid(),
                refstate,
                False,
            )
            refstate = refstate.to(state_dtype).float()
            outputs.append(out.cpu())
            refs.append(ref)
            if t in (0, 7, 31):
                compare(
                    f"gdn_decode_state_{state_dtype}_{t+1}",
                    pool[[1, 3]],
                    refstate,
                    0.03,
                    0.02,
                )
        compare(
            f"gdn_decode_32_{state_dtype}",
            torch.cat(outputs, 1),
            torch.cat(refs, 1).bfloat16(),
            0.02,
            0.02,
        )
        compare(
            f"gdn_untouched_{state_dtype}", pool[[0, 2, 4]], initial[[0, 2, 4]], 0, 0
        )
    for channels in (257, 6144):
        lengths = [1, 7, 8, 9, 65]
        x, weight = rand(sum(lengths), channels), rand(channels, 4, scale=0.1)
        initial = rand(7, channels, 3, scale=0.1)
        state = initial.to("npu")
        bounds_cpu = [0]
        for n in lengths:
            bounds_cpu.append(bounds_cpu[-1] + n)
        bounds = torch.tensor(bounds_cpu, dtype=torch.int32, device="npu")
        meta = build_causal_conv1d_prefill_metadata(
            bounds, torch.tensor(lengths, dtype=torch.int32), 8
        )
        indices = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device="npu")
        history = [False, True, False, True, False]
        got = causal_conv1d_fn(
            x.T.to("npu"),
            weight.to("npu"),
            None,
            state,
            bounds,
            cache_indices=indices,
            has_initial_state=torch.tensor(history, device="npu"),
            activation="silu",
            prefill_metadata=meta,
        )
        refparts = []
        for i, n in enumerate(lengths):
            prefix = initial[i + 1] if history[i] else torch.zeros(channels, 3)
            seq = torch.cat(
                [prefix.float(), x[bounds_cpu[i] : bounds_cpu[i + 1]].T.float()], 1
            )
            ref = F.silu(F.conv1d(seq[None], weight.float()[:, None], groups=channels))[
                0
            ]
            refparts.append(ref)
            compare(
                f"conv_prefill_state_C{channels}_L{n}", state[i + 1], seq[:, -3:], 0, 0
            )
        compare(
            f"conv_prefill_C{channels}",
            got,
            torch.cat(refparts, 1).bfloat16(),
            0.02,
            0.02,
        )
        reference_state = state.cpu().clone()
        untouched = state[[0, 2, 4, 6]].clone()
        for t in range(32):
            xx = rand(2, channels)
            out = causal_conv1d_update(
                xx.to("npu"),
                state,
                weight.to("npu"),
                activation="silu",
                conv_state_indices=torch.tensor(
                    [1, 3], dtype=torch.int32, device="npu"
                ),
                parent_indices=None,
            )
            seq = torch.cat(
                [reference_state[[1, 3]].float(), xx[:, :, None].float()], -1
            )
            ref = F.silu((seq * weight.float()[None]).sum(-1))
            reference_state[[1, 3]] = seq[:, :, 1:].bfloat16()
            if t in (0, 7, 31):
                compare(
                    f"conv_decode_C{channels}_step{t+1}",
                    out,
                    ref.bfloat16(),
                    0.02,
                    0.02,
                )
                compare(
                    f"conv_decode_state_C{channels}_step{t+1}",
                    state[[1, 3]],
                    reference_state[[1, 3]],
                    0,
                    0,
                )
        compare(f"conv_untouched_C{channels}", state[[0, 2, 4, 6]], untouched, 0, 0)
    torch.npu.synchronize()
    assert all(
        row["passed"] for row in records
    ), "Numerical comparison failed; see terminal output"
    print(f"ALL {len(records)} checks passed", flush=True)


if __name__ == "__main__":
    main()
