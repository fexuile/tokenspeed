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

"""Check interleaved/concurrent requests against references held in memory."""

import argparse
import concurrent.futures
import sys

sys.dont_write_bytecode = True
from model_checks import (
    default_model,
    prepare_cases,
    print_case,
    reference_results,
    server_result,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=default_model())
    parser.add_argument("--url", default="http://127.0.0.1:31905")
    parser.add_argument("--served-model", default="qwen3.5-0.8b")
    args = parser.parse_args()
    tokenizer, cases = prepare_cases(args.model)
    expected = reference_results(args.model, tokenizer, cases)
    rows = []

    def request(index):
        return server_result(cases[index], args.url, args.served_model)

    for indices in ([0, 1], [2, 3], [0, 1]):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            rows.extend(executor.map(request, indices))
    rows.append(request(0))
    for row in rows:
        passed = row["text"] == expected[row["index"]]["text"]
        print_case("ISOLATION", row)
        print(f"case={row['index']} exact_reference_text={passed}", flush=True)
    passed = sum(row["text"] == expected[row["index"]]["text"] for row in rows)
    print(f"ISOLATION {passed}/{len(rows)} passed", flush=True)
    assert passed == len(rows), "Request isolation comparison failed"


if __name__ == "__main__":
    main()
