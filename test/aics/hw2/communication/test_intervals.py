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

"""Check overlap accounting against hand-computed timelines."""

import unittest

from aics.hw2.communication.intervals import account, intersect, measure, merge


class TimelineTests(unittest.TestCase):
    def test_unsorted_nested_touching_and_empty_spans(self):
        self.assertEqual(merge([(8, 8), (3, 6), (1, 4), (6, 7), (2, 3)]), [(1, 7)])
        self.assertEqual(measure([(1, 4), (2, 6), (9, 10)]), 6)

    def test_intersection_handles_disjoint_and_nested_spans(self):
        self.assertEqual(
            intersect([(1, 5), (7, 10)], [(2, 3), (4, 8)]), [(2, 3), (4, 5), (7, 8)]
        )
        self.assertEqual(intersect([], [(1, 2)]), [])

    def test_overlap_and_wait_are_not_added_twice(self):
        result = account([(1, 5), (4, 7)], [(5, 9)], [(6, 8)], (0, 10))
        self.assertEqual(
            result,
            dict(
                compute_us=6,
                communication_us=4,
                overlap_us=2,
                communication_wait_us=2,
                uncovered_us=2,
            ),
        )
        self.assertEqual(
            result["compute_us"]
            + result["communication_us"]
            - result["overlap_us"]
            + result["uncovered_us"],
            10,
        )

    def test_clipping_excludes_other_forwards(self):
        result = account([(-5, 2), (8, 15)], [(1, 4), (9, 20)], [(3, 30)], (0, 10))
        self.assertEqual(
            result,
            dict(
                compute_us=4,
                communication_us=4,
                overlap_us=2,
                communication_wait_us=2,
                uncovered_us=4,
            ),
        )


if __name__ == "__main__":
    unittest.main()
