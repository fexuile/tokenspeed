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

"""Print model reference, service output, or in-memory comparison."""

import argparse
import json
import sys

sys.dont_write_bytecode = True
from model_checks import (
    compare_results,
    default_model,
    prepare_cases,
    print_case,
    reference_results,
    server_result,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["reference", "server", "compare"])
    parser.add_argument("--model", default=default_model())
    parser.add_argument("--url", default="http://127.0.0.1:31905")
    parser.add_argument("--served-model", default="qwen3.5-0.8b")
    args = parser.parse_args()
    tokenizer, cases = prepare_cases(args.model)
    reference = None
    if args.mode in ("reference", "compare"):
        reference = reference_results(args.model, tokenizer, cases)
    if args.mode in ("server", "compare"):
        actual = []
        for case in cases:
            row = server_result(case, args.url, args.served_model)
            actual.append(row)
            print_case("SERVER", row)
        if args.mode == "compare":
            result = compare_results(actual, reference)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            assert result["passed"], "Model comparison failed; see terminal output"
            print(f"PASS {len(cases)}/{len(cases)} model comparisons", flush=True)


if __name__ == "__main__":
    main()
