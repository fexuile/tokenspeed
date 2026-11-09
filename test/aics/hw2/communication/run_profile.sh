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

set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
: "${PYTHON_VENV:?Set PYTHON_VENV to the TokenSpeed virtualenv}"
source "$PYTHON_VENV/bin/activate"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH="$REPO_ROOT/test:$REPO_ROOT/python:$REPO_ROOT/tokenspeed-kernel/python:$REPO_ROOT/tokenspeed-kernel-npu/python"
export TOKENSPEED_KERNEL_BACKEND=npu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HCCL_NPU_SOCKET_PORT_RANGE=auto
MODEL="${AICS_MODEL:-$HOME/models/Qwen3.5-9B}"
OUTPUT_ROOT="${AICS_OUTPUT_ROOT:-$REPO_ROOT/outputs/aics-hw2-communication}"
for HW2_ANALYSIS_TP in 1 2; do
    export HW2_ANALYSIS_TP
    if [[ "$HW2_ANALYSIS_TP" == 1 ]]; then
        export ASCEND_RT_VISIBLE_DEVICES=1
    else
        export ASCEND_RT_VISIBLE_DEVICES=0,1
    fi
    export HW2_ANALYSIS_OUT="$OUTPUT_ROOT/tp${HW2_ANALYSIS_TP}"
    test ! -e "$HW2_ANALYSIS_OUT"
    mkdir -p "$HW2_ANALYSIS_OUT"
    timeout 600s python -m aics.hw2.communication.profile_workers --model "$MODEL" --tp "$HW2_ANALYSIS_TP" --output "$HW2_ANALYSIS_OUT" > "$HW2_ANALYSIS_OUT/run.log" 2>&1
    echo "DONE tp=$HW2_ANALYSIS_TP"
done
