"""
05_split_dataset.py

Splits labels.csv into train/val/test sets, grouped by PERFORMANCE.

Why performance and not video_id: video_id was the wrong unit. "Thrikketta
2015 | 1am Kaalam" and "Thrikketta 2015 | 4am Kaalam" are two video_ids but
one performance — same ensemble, same day, same microphone, same upload
session. One landed in train and the other in test, and grouping by video_id
could not see it. performance_groups.csv maps video_id -> performance_id;
videos sharing a performance_id are never split apart.

Why grouped at all: multiple clips come from the same source video (a single
performance). If clips from the same video end up in both train and test,
the model can partly "memorize" that specific recording's noise/room
acoustics rather than learning general melam patterns, inflating test
accuracy artificially. Splitting by video_id keeps every clip from one
video entirely in one split.

Why NOT plain GroupShuffleSplit: it balances splits by NUMBER OF VIDEOS,
not number of clips. With source video length varying widely (~10 min to
3 hours), one long video can single-handedly dominate whichever split it
lands in — observed in practice as test=193 clips / val=1997 clips for
Panchari in one run, and the reverse skew for Panchavadyam, purely from
which few long videos happened to land where. That makes the resulting
test-set metrics unreliable, not just noisy. Instead, this does a greedy
largest-video-first assignment per class, always adding the next video to
whichever split is furthest below its clip-count target — this keeps the
same no-leakage guarantee (every clip from one video stays in one split)
while keeping actual clip counts balanced.

Requires: pandas

Usage:
    python 05_split_dataset.py
"""

import csv
import os

import numpy as np
import pandas as pd

import config


def load_performance_groups():
    """Returns {video_id: performance_id}. Unlisted videos are their own group.

    A performance is the real leakage unit. Re-uploads of one show, or one
    show uploaded in parts, share a room, a microphone and an ensemble — a
    model that sees part 1 in train and part 2 in test is being graded on a
    recording it has already heard.
    """
    path = getattr(config, "PERFORMANCE_GROUPS_CSV", None)
    if not path or not os.path.exists(path):
        print("No performance_groups.csv found — grouping by video_id alone. "
              "Re-uploads and multi-part performances will NOT be caught.")
        return {}
    groups = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            groups[row["video_id"].strip()] = row["performance_id"].strip()
    print(f"Performance groups loaded for {len(groups)} videos.")
    return groups


def assign_videos_to_splits(df, train_ratio, val_ratio, test_ratio, seed):
    """Returns {performance_id: split_name}, balancing CLIP COUNT per class
    per split (not performance count) via greedy largest-first bin-packing,
    while keeping every performance's clips together in one split."""
    assignments = {}

    for cls, group in df.groupby("class"):
        video_counts = group.groupby("performance_id").size()
        total_clips = video_counts.sum()
        targets = {
            "train": train_ratio * total_clips,
            "val": val_ratio * total_clips,
            "test": test_ratio * total_clips,
        }
        current = {"train": 0, "val": 0, "test": 0}

        # Largest videos first (classic LPT bin-packing heuristic) — placing
        # the biggest, hardest-to-balance items first gives a much tighter
        # final balance than placing them last. Shuffle within same-size
        # ties so the choice among equal videos isn't alphabetical/arbitrary.
        shuffled_ids = video_counts.sample(frac=1.0, random_state=seed).index
        video_ids = video_counts.loc[shuffled_ids].sort_values(ascending=False).index

        for vid in video_ids:
            n = video_counts[vid]
            # Assign to whichever split is currently furthest below its
            # target — this is what keeps per-class clip counts balanced.
            deficits = {s: targets[s] - current[s] for s in current}
            best_split = max(deficits, key=deficits.get)
            current[best_split] += n
            assignments[vid] = best_split

    return assignments


def main():
    if not os.path.exists(config.LABELS_CSV):
        raise SystemExit(f"Missing {config.LABELS_CSV} — run 03_generate_spectrograms.py first.")
    if not os.path.exists(config.SEGMENTS_METADATA_CSV):
        raise SystemExit(f"Missing {config.SEGMENTS_METADATA_CSV} — run 02_segment_audio.py first.")

    labels = pd.read_csv(config.LABELS_CSV)
    segments = pd.read_csv(config.SEGMENTS_METADATA_CSV)[["clip_id", "video_id"]]

    df = labels.merge(segments, on="clip_id", how="left")
    missing_video_id = df["video_id"].isna().sum()
    if missing_video_id:
        print(f"Warning: {missing_video_id} clips have no matching video_id "
              f"(likely augmented clips) — dropping them from splitting.")
        df = df.dropna(subset=["video_id"])

    assert abs(config.TRAIN_SPLIT + config.VAL_SPLIT + config.TEST_SPLIT - 1.0) < 1e-6, \
        "TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT must sum to 1.0 in config.py"

    perf_map = load_performance_groups()
    df["performance_id"] = df["video_id"].map(lambda v: perf_map.get(v, v))

    collapsed = df["video_id"].nunique() - df["performance_id"].nunique()
    if collapsed:
        print(f"{collapsed} video(s) collapsed into shared performances — "
              f"these will not be split across train/test.")

    perf_to_split = assign_videos_to_splits(
        df, config.TRAIN_SPLIT, config.VAL_SPLIT, config.TEST_SPLIT, config.RANDOM_SEED
    )
    out = df.copy()
    out["split"] = out["performance_id"].map(perf_to_split)

    # video_id stays in the output — it is still what per-video reporting in
    # 07 aggregates over — but performance_id is what decided the split.
    out = out[["clip_id", "class", "npy_path", "video_id", "performance_id", "split"]]
    out.to_csv(config.SPLITS_CSV, index=False)

    # Assert the guarantee rather than trusting it. This is exactly the check
    # that would have caught the Thrikketta 2015 leak.
    leaked = out.groupby("performance_id")["split"].nunique()
    leaked = leaked[leaked > 1]
    if len(leaked):
        raise SystemExit(
            f"LEAK: {len(leaked)} performance(s) appear in more than one "
            f"split: {list(leaked.index)[:5]}")
    print("Leakage check passed: no performance spans more than one split.")

    print(f"\nSplits written to {config.SPLITS_CSV}\n")
    print("Clips per split:")
    print(out["split"].value_counts(), "\n")
    print("Class balance within each split:")
    print(out.groupby(["split", "class"]).size().unstack(fill_value=0))
    print("\nUnique PERFORMANCES per split per class (this is your real n):")
    print(out.groupby(["split", "class"])["performance_id"].nunique().unstack(fill_value=0))

    test_perf = out[out.split == "test"].groupby("class")["performance_id"].nunique()
    thin = test_perf[test_perf < 5]
    if len(thin):
        print("\nWARNING: these classes are tested on fewer than 5 performances:")
        for cls, n in thin.items():
            print(f"  {cls:15s} {n} performance(s) — per-class recall here has "
                  f"error bars tens of points wide")


if __name__ == "__main__":
    main()