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
export REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PYTHON_VENV:-$HOME/tokenspeed/.venv}/bin/activate"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd "$REPO_ROOT/test/ascend_hw"
export PYTHONPATH="$REPO_ROOT/python:$REPO_ROOT/tokenspeed-kernel/python:$REPO_ROOT/tokenspeed-kernel-npu/python"
export PYTHONDONTWRITEBYTECODE=1
export TOKENSPEED_KERNEL_BACKEND=npu
export ASCEND_RT_VISIBLE_DEVICES="${NPU_CHIP_ID:-6}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
python - <<'PY'
import os
from pathlib import Path
import tokenspeed, tokenspeed_kernel, tokenspeed_kernel_npu
repo = Path(os.environ["REPO_ROOT"]).resolve()
for module in (tokenspeed, tokenspeed_kernel, tokenspeed_kernel_npu):
    print(module.__name__, module.__file__, flush=True)
    assert Path(module.__file__).resolve().is_relative_to(repo)
PY
exec python -B -m tokenspeed.cli serve "$HOME/models/Qwen3.5-0.8B" \
    --served-model-name qwen3.5-0.8b \
    --device npu --dtype bfloat16 --world-size 1 --language-model-only \
    --kv-cache-dtype auto --attention-backend mha --sampling-backend greedy \
    --mamba-ssm-dtype float32 --enforce-eager --disable-prefill-graph \
    --disable-pdl --disable-autotune --disable-prefix-caching --disable-kvstore \
    --retraction-snapshot-ratio 0 \
    --max-model-len 4096 --max-num-seqs 2 --max-total-tokens 8192 \
    --chunked-prefill-size 4096 --prefix-granularity 128 \
    --host 127.0.0.1 --port "${SERVER_PORT:-31905}"
