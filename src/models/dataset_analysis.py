"""
Dataset Analysis Script — OEP Ground Truth Explorer
=====================================================
Parses all gt.txt files in the OEP dataset, then produces:
  • Per-subject label distribution table
  • Dataset-wide summary (frame counts, durations, class imbalance)
  • Console bar-chart of label distribution
  • Saved bar + pie charts  →  <reports_path>/dataset_analysis/

Defaults are read from src/configs/app.yaml (model: section).
CLI args override YAML values for one-off runs.

Usage:
  python src/models/dataset_analysis.py
  python src/models/dataset_analysis.py --dataset-path /path/to/OEP/database
  python src/models/dataset_analysis.py --no-plot   # text-only output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ── Load config (same helper as cnn_bi_lstm_train.py) ──────────────────────── #

def _load_model_config():
    try:
        _root = Path(__file__).resolve().parent.parent.parent
        from src.utils.config import Config
        return Config(str(_root / "src" / "configs" / "app.yaml")).model
    except Exception:
        try:
            from src.utils.config import Config
            return Config().model
        except Exception:
            return None


# ── Argument parsing ───────────────────────────────────────────────────────── #

def parse_args() -> argparse.Namespace:
    _cfg = _load_model_config()

    p = argparse.ArgumentParser(
        description="OEP dataset ground-truth analyser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset-path",
        default=_cfg.dataset_path if _cfg else "data/raw/database",
        help="Root folder containing subjectN directories",
    )
    p.add_argument(
        "--output-dir",
        default=str(Path(_cfg.reports_path if _cfg else "data/reports") / "dataset_analysis"),
        help="Where to save the output plots",
    )
    p.add_argument(
        "--fps", type=float, default=30.0,
        help="Video FPS used to convert frame counts → seconds",
    )
    p.add_argument(
        "--no-plot", action="store_true",
        help="Skip saving plots (text output only)",
    )
    return p.parse_args()


# ── GT parser ────────────────────────────────────────────────────────────────  #

def parse_gt(gt_path: Path) -> pd.DataFrame | None:
    for sep in ["\t", r"\s+"]:
        try:
            df = pd.read_csv(
                gt_path, sep=sep, header=None,
                names=["Start", "End", "Label"],
                engine="python", on_bad_lines="skip",
            )
            df = df.dropna(subset=["Label"])
            df["Start"]  = pd.to_numeric(df["Start"],  errors="coerce").fillna(0).astype(int)
            df["End"]    = pd.to_numeric(df["End"],    errors="coerce").fillna(0).astype(int)
            df["Label"]  = pd.to_numeric(df["Label"],  errors="coerce")
            df = df.dropna(subset=["Label"])
            df["Label"]  = df["Label"].astype(int)
            df["Frames"] = df["End"] - df["Start"] + 1
            if len(df) > 0:
                return df
        except Exception:
            continue
    return None


# ── Scan all subjects ─────────────────────────────────────────────────────── #

def scan_dataset(dataset_path: Path, fps: float) -> tuple[pd.DataFrame, dict]:
    subjects = sorted([
        d for d in dataset_path.iterdir()
        if d.is_dir() and d.name.startswith("subject")
    ])

    if not subjects:
        print(f"[ERROR] No subject* directories found in: {dataset_path}")
        sys.exit(1)

    print(f"Found {len(subjects)} subject(s) in {dataset_path}\n")

    rows = []
    missing_gt = []

    for subject_dir in subjects:
        gt_path = subject_dir / "gt.txt"
        if not gt_path.exists():
            missing_gt.append(subject_dir.name)
            continue

        df = parse_gt(gt_path)
        if df is None or len(df) == 0:
            missing_gt.append(subject_dir.name)
            continue

        for _, row in df.iterrows():
            rows.append({
                "Subject": subject_dir.name,
                "Label":   int(row["Label"]),
                "Start":   int(row["Start"]),
                "End":     int(row["End"]),
                "Frames":  int(row["Frames"]),
            })

    if missing_gt:
        print(f"⚠  {len(missing_gt)} subject(s) with missing/unreadable gt.txt: {missing_gt}\n")

    if not rows:
        print("[ERROR] No labelled segments could be parsed.")
        sys.exit(1)

    full_df = pd.DataFrame(rows)
    full_df["Seconds"] = full_df["Frames"] / fps

    meta = {"fps": fps, "num_subjects": len(subjects), "missing_gt": missing_gt}
    return full_df, meta


# ── Reporting ─────────────────────────────────────────────────────────────── #

def console_bar(value: int, max_value: int, width: int = 30) -> str:
    filled = int(round(value / max_value * width)) if max_value else 0
    return "█" * filled + "░" * (width - filled)


def print_summary(df: pd.DataFrame, meta: dict) -> None:
    fps            = meta["fps"]
    sep            = "─" * 74
    total_segments = len(df)
    total_frames   = df["Frames"].sum()
    total_seconds  = total_frames / fps

    # Discover all labels present in data
    all_labels = sorted(df["Label"].unique())

    print(sep)
    print("  OEP DATASET — GROUND TRUTH ANALYSIS")
    print(sep)
    print(f"  Subjects parsed  : {df['Subject'].nunique()} / {meta['num_subjects']}")
    print(f"  Labels found     : {all_labels}")
    print(f"  Total segments   : {total_segments:,}")
    print(f"  Total frames     : {total_frames:,}")
    print(f"  Total duration   : {total_seconds/60:.1f} min  ({total_seconds:.0f} s)")
    print(sep)

    # ── Class distribution ─────────────────────────────────────────────────── #
    print("\n  CLASS DISTRIBUTION (all subjects combined)\n")
    print(f"  {'Label':<8}  {'Segments':>9}  {'Frames':>10}  {'Duration':>10}  {'Share':>7}  Chart")
    print("  " + "─" * 66)

    grp = df.groupby("Label").agg(
        Segments=("Label",   "count"),
        Frames  =("Frames",  "sum"),
        Seconds =("Seconds", "sum"),
    ).reset_index()

    max_segs = grp["Segments"].max()

    for _, row in grp.iterrows():
        lbl   = int(row["Label"])
        segs  = int(row["Segments"])
        frames = int(row["Frames"])
        secs  = row["Seconds"]
        share = segs / total_segments * 100
        bar   = console_bar(segs, max_segs)
        print(f"  {lbl:<8}  {segs:>9,}  {frames:>10,}  {secs:>8.1f}s  {share:>6.1f}%  {bar}")

    print()

    # ── Per-subject table ──────────────────────────────────────────────────── #
    print(sep)
    print("  PER-SUBJECT BREAKDOWN  (frames per label)\n")

    col_w = max(10, *(len(str(l)) + 2 for l in all_labels))
    header = f"  {'Subject':<14}" + "".join(f"  {'Lbl '+str(l):>{col_w}}" for l in all_labels) + f"  {'Total':>9}"
    print(header)
    print("  " + "─" * (14 + (col_w + 2) * len(all_labels) + 11))

    for subject in sorted(df["Subject"].unique()):
        sub_df = df[df["Subject"] == subject]
        counts = sub_df.groupby("Label")["Frames"].sum()
        row_str = f"  {subject:<14}"
        total = 0
        for l in all_labels:
            val = int(counts.get(l, 0))
            total += val
            row_str += f"  {val:>{col_w},}"
        row_str += f"  {total:>9,}"
        print(row_str)

    print()
    print(sep)

    # ── Imbalance summary ──────────────────────────────────────────────────── #
    minority = grp.loc[grp["Segments"].idxmin()]
    majority = grp.loc[grp["Segments"].idxmax()]
    ratio    = majority["Segments"] / max(minority["Segments"], 1)

    print(f"\n  Imbalance ratio  : {ratio:.1f}x")
    print(f"  Most common      : Label {int(majority['Label'])}  ({int(majority['Segments'])} segments)")
    print(f"  Least common     : Label {int(minority['Label'])}  ({int(minority['Segments'])} segments)")

    if ratio > 5:
        print("\n  ⚠  High class imbalance detected (>5×). Consider:")
        print("      • Weighted loss function during training")
        print("      • Oversampling minority classes")
        print("      • Augmenting frames from under-represented labels")

    print(sep + "\n")


# ── Plotting ──────────────────────────────────────────────────────────────── #

# Generate a consistent colour palette from the labels found in the data
def _make_palette(labels: list[int]) -> dict[int, str]:
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(labels))
    return {lbl: cmap(i) for i, lbl in enumerate(labels)}


def plot_charts(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_labels = sorted(df["Label"].unique())
    palette    = _make_palette(all_labels)
    colors     = [palette[l] for l in all_labels]

    grp = df.groupby("Label").agg(
        Segments=("Label",   "count"),
        Seconds =("Seconds", "sum"),
    ).reset_index()

    label_strs = [f"Label {l}" for l in all_labels]
    seg_counts = [int(grp.loc[grp["Label"] == l, "Segments"].values[0]) if l in grp["Label"].values else 0
                  for l in all_labels]
    sec_counts = [float(grp.loc[grp["Label"] == l, "Seconds"].values[0]) if l in grp["Label"].values else 0.0
                  for l in all_labels]

    # ── 1. Stacked per-subject bar chart ────────────────────────────────────── #
    subjects = sorted(df["Subject"].unique())
    pivot = df.pivot_table(index="Subject", columns="Label",
                           values="Frames", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(subjects)

    fig, ax = plt.subplots(figsize=(max(10, len(subjects) * 0.7), 6))
    bottom = np.zeros(len(subjects))
    for lbl in all_labels:
        vals = pivot[lbl].values.astype(float) if lbl in pivot.columns else np.zeros(len(subjects))
        ax.bar(subjects, vals, bottom=bottom,
               label=f"Label {lbl}", color=palette[lbl],
               edgecolor="white", linewidth=0.4)
        bottom += vals

    ax.set_title("Frame Count per Subject — Stacked by Label", fontsize=13, fontweight="bold")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Frames")
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "per_subject_stacked.png", dpi=150)
    plt.close(fig)

    # ── 2. Overall distribution bar charts ──────────────────────────────────── #
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bars = ax.bar(label_strs, seg_counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("Segment Count by Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of segments")
    ax.bar_label(bars, padding=3, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    ax = axes[1]
    bars = ax.bar(label_strs, sec_counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("Total Duration by Label (seconds)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Seconds")
    ax.bar_label(bars, fmt="%.0f s", padding=3, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    fig.suptitle("OEP Dataset — Label Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "label_distribution_bars.png", dpi=150)
    plt.close(fig)

    # ── 3. Pie chart ─────────────────────────────────────────────────────────── #
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        seg_counts, labels=label_strs, colors=colors,
        autopct="%1.1f%%", startangle=140,
        pctdistance=0.75, wedgeprops=dict(edgecolor="white", linewidth=1),
    )
    for t in autotexts:
        t.set_fontsize(10)
    ax.set_title("Segment Share by Label", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "label_pie_chart.png", dpi=150)
    plt.close(fig)

    print(f"  Charts saved to: {output_dir}/")
    print(f"    • per_subject_stacked.png")
    print(f"    • label_distribution_bars.png")
    print(f"    • label_pie_chart.png\n")


# ── Main ──────────────────────────────────────────────────────────────────── #

def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    output_dir   = Path(args.output_dir)

    if not dataset_path.exists():
        print(f"[ERROR] Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    df, meta = scan_dataset(dataset_path, args.fps)
    print_summary(df, meta)

    if not args.no_plot:
        plot_charts(df, output_dir)


if __name__ == "__main__":
    main()
