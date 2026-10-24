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
if [[ $# -lt 2 ]]; then
    echo "Usage: PYTHON_VENV=/path/to/venv bash run.sh <modules|regression|unit|reference|server|compare|requests|benchmark|memory> <physical-chip-id> [test-options]"
    exit 2
fi
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$REPO_ROOT/test/ascend_hw"
: "${PYTHON_VENV:?Set PYTHON_VENV to an existing Ascend Python environment}"
source "${PYTHON_VENV}/bin/activate"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd "$TEST_ROOT"
export PYTHONPATH="$REPO_ROOT/python:$REPO_ROOT/tokenspeed-kernel/python:$REPO_ROOT/tokenspeed-kernel-npu/python"
export ASCEND_RT_VISIBLE_DEVICES="$2"
export TOKENSPEED_KERNEL_BACKEND=npu
export PYTHONDONTWRITEBYTECODE=1
unset OMP_NUM_THREADS
export TOKENIZERS_PARALLELISM=false
unset FORWARD_DIAGNOSTICS
TASK="$1"
shift 2
case "$TASK" in
    modules|regression|unit)
        if [[ $# -ne 0 ]]; then echo "Unexpected options for $TASK: $*" >&2; exit 2; fi
        case "$TASK" in
            modules) python -B "$TEST_ROOT/check_modules.py" ;;
            unit) python -B -m pytest -q -p no:cacheprovider "$TEST_ROOT/test_console_helpers.py" ;;
            regression) python -B -m pytest -q -p no:cacheprovider "$REPO_ROOT/tokenspeed-kernel-npu/test/test_mha.py" "$REPO_ROOT/tokenspeed-kernel-npu/test/test_layernorm.py" "$REPO_ROOT/tokenspeed-kernel-npu/test/test_rotary_embedding.py" "$REPO_ROOT/tokenspeed-kernel-npu/test/test_triton.py" ;;
        esac ;;
    reference|server|compare) python -B "$TEST_ROOT/verify-qwen35.py" "$TASK" "$@" ;;
    requests) python -B "$TEST_ROOT/check_request_isolation.py" "$@" ;;
    benchmark|memory) python -B "$TEST_ROOT/bench_engine.py" "$TASK" "$@" ;;
    *) echo "Unknown task: $TASK" >&2; exit 2 ;;
esac
