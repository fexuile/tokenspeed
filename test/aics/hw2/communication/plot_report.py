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

"""Render submission figures from the validated device trace analysis."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(ROOT / "matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch

FONT = "/usr/share/fonts/google-droid-fonts/DroidSansFallback.ttf"
font_manager.fontManager.addfont(FONT)
plt.rcParams.update(
    {
        "font.family": [
            "DejaVu Sans",
            font_manager.FontProperties(fname=FONT).get_name(),
        ],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "path",
        "axes.unicode_minus": False,
    }
)
OUT = None
DATA = None
LOOKUP = {}
BLUE, ORANGE, GREY, PURPLE = "#31688e", "#e69a36", "#cdd3d8", "#8054a3"
CASES = [(128, 1), (128, 2), (2048, 1), (2048, 2)]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def timing():
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.7), layout="constrained")
    for ax, phase, title in zip(
        axes, ("prefill", "decode"), ("(a) Prefill", "(b) Decode"), strict=True
    ):
        x = np.arange(4)
        one = [LOOKUP[1, 0, l, b, phase]["span_us"] / 1000 for l, b in CASES]
        two = [
            max(LOOKUP[2, r, l, b, phase]["span_us"] for r in (0, 1)) / 1000
            for l, b in CASES
        ]
        for offset, values, color, label in (
            (-0.19, one, BLUE, "TP=1"),
            (0.19, two, ORANGE, "TP=2（两 rank 较大值）"),
        ):
            bars = ax.bar(
                x + offset, values, width=0.35, color=color, label=label, zorder=3
            )
            ax.bar_label(
                bars,
                labels=[
                    f"{v:.0f}" if phase == "prefill" else f"{v:.1f}" for v in values
                ],
                padding=3,
                fontsize=9,
            )
        ax.set_xticks(x, [f"L={l}\nB={b}" for l, b in CASES])
        ax.set_ylabel("主干跨度 / ms")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylim(0, max(one + two) * 1.18)
        ax.grid(axis="y", alpha=0.2, zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    save(fig, "tp-timing")


def breakdown():
    fig, (ax, trace) = plt.subplots(
        2,
        1,
        figsize=(10.8, 6.2),
        gridspec_kw={"height_ratios": [1, 1.1]},
        layout="constrained",
    )
    cases = [
        (2048, 2, "prefill", 0),
        (2048, 2, "prefill", 1),
        (128, 2, "decode", 0),
        (128, 2, "decode", 1),
    ]
    for y, (length, batch, phase, rank) in enumerate(cases):
        row = LOOKUP[2, rank, length, batch, phase]
        span = row["span_us"]
        left = 0
        for field, color in (
            ("compute_us", BLUE),
            ("communication_us", ORANGE),
            ("uncovered_us", GREY),
        ):
            width = 100 * row[field] / span
            ax.barh(
                y,
                width,
                left=left,
                color=color,
                height=0.65,
                edgecolor="white",
                linewidth=0.6,
            )
            if width > 10:
                label = {
                    "compute_us": "K",
                    "communication_us": "C",
                    "uncovered_us": "G",
                }[field]
                ax.text(
                    left + width / 2,
                    y,
                    f"{label} {row[field]/1000:.1f} ms",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if field == "compute_us" else "#202a33",
                )
            if field == "communication_us":
                ax.barh(
                    y,
                    100 * row["communication_wait_us"] / span,
                    left=left,
                    height=0.65,
                    facecolor="none",
                    edgecolor=PURPLE,
                    hatch="////",
                    linewidth=0,
                )
            left += width
    ax.invert_yaxis()
    ax.set_yticks(
        range(4), [f"{phase.capitalize()} {l}/{b} · r{r}" for l, b, phase, r in cases]
    )
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("(a) 主干时间分解（斜线等待已含于 C）", loc="left", fontweight="bold")
    ax.set_xlabel("主干跨度占比")
    handles = [
        Patch(facecolor=BLUE, label="K 计算/布局"),
        Patch(facecolor=ORANGE, label="C 集合（含等待）"),
        Patch(facecolor=GREY, label="G 未覆盖间隔"),
        Patch(
            facecolor="none", edgecolor=PURPLE, hatch="////", label="与 C 重合的流等待"
        ),
    ]
    fig.legend(
        handles=handles, loc="outside upper center", ncol=4, frameon=False, fontsize=9
    )
    snippets = [
        s
        for s in json.loads((ROOT / "timeline-snippets.json").read_text())
        if s["phase"] == "decode"
    ]
    assert len(snippets) == 2
    origin = min(s["origin_us"] for s in snippets)
    limit = 6.0
    for s in snippets:
        base = 3 if s["rank"] == 0 else 0
        delta = 0  # Each rank has its own device-clock origin.
        series = [
            (s["compute"], base + 2, BLUE),
            ([(c["start"], c["end"]) for c in s["communication"]], base + 1, ORANGE),
            (s["waits"], base, PURPLE),
        ]
        for intervals, y, color in series:
            bars = []
            for start, end in intervals:
                a = max(0, delta + start / 1000)
                b = min(limit, delta + end / 1000)
                if b > a:
                    bars.append((a, b - a))
            trace.broken_barh(
                bars, (y - 0.27, 0.54), facecolors=color, edgecolors="none"
            )
    trace.set_yticks(
        [5, 4, 3, 2, 1, 0],
        ["r0 计算", "r0 集合", "r0 等待", "r1 计算", "r1 集合", "r1 等待"],
    )
    trace.set_ylim(-0.6, 5.7)
    trace.set_xlim(0, limit)
    trace.axhline(2.5, color="#8d959c", linewidth=0.6)
    trace.grid(axis="x", alpha=0.18)
    trace.set_xlabel("各 rank 自首个 GEMM 起的设备时间 / ms（分别归零）")
    trace.set_title(
        "(b) 128/2 的一个 decode 样本：前 6 ms", loc="left", fontweight="bold"
    )
    save(fig, "communication-breakdown")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ROOT = args.analysis_dir.resolve()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    DATA = json.loads((ROOT / "summary.json").read_text())
    LOOKUP = {
        (r["tp"], r["rank"], r["input_length"], r["batch"], r["phase"]): r for r in DATA
    }
    timing()
    breakdown()
    print(OUT)
