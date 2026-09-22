"""
02_segment_audio.py

Takes raw downloaded audio (01_raw_audio/<class>/*.wav) and:
  1. Loads + resamples to config.SAMPLE_RATE
  2. Removes silence using librosa.effects.split
  3. Cuts remaining audio into 5-10 sec clips
  4. Drops low-energy (likely junk/silence/crowd-noise-only) clips
  5. Saves clips to 02_segments/<class>/ and writes segments_metadata.csv

Requires: librosa, soundfile, numpy, pandas, tqdm

Usage:
    python 02_segment_audio.py
"""

import argparse
import csv
import os

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

import config


def load_corrections():
    """Returns (dropped_ids, relabel_map) from LABEL_CORRECTIONS_CSV.

    The class labels originally came from the YouTube search query that
    found each video, not from anyone listening. This file is where a
    human overrides that.
    """
    path = getattr(config, "LABEL_CORRECTIONS_CSV", None)
    if not path or not os.path.exists(path):
        print("No label_corrections.csv found — using labels as collected.")
        return set(), {}

    dropped, relabel = set(), {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["video_id"].strip()
            action = row["action"].strip().lower()
            if action == "drop":
                dropped.add(vid)
            elif action == "relabel":
                new = row["new_class"].strip()
                if new not in config.CLASSES:
                    raise SystemExit(
                        f"label_corrections.csv: '{new}' is not in config.CLASSES")
                relabel[vid] = new
    print(f"Corrections loaded: {len(dropped)} videos dropped, "
          f"{len(relabel)} relabelled.")
    return dropped, relabel

def ensure_dirs():
    os.makedirs(config.SEGMENTS_DIR, exist_ok=True)
    for cls in config.CLASSES:
        os.makedirs(os.path.join(config.SEGMENTS_DIR, cls), exist_ok=True)


def merge_intervals(intervals, sr, max_gap_sec):
    """Join non-silent intervals separated by only a brief gap.

    librosa.effects.split cuts at every dip below the threshold, and melam
    is full of them — the space between strokes in a slow kalam, a breath in
    the kuzhal, a lull in the crowd. At CLIP_MIN_SEC=5 those fragments were
    mostly still long enough to use. At 20s they are not, and without
    merging, this function discards nearly everything: the smoke test on a
    slow click-train produced zero clips.

    A gap of a fraction of a second inside a continuous performance is part
    of the music, not a boundary between takes.
    """
    if len(intervals) == 0:
        return intervals
    max_gap = int(max_gap_sec * sr)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], end)  # overlap-chunk pieces
        else:                                        # can arrive out of order
            merged.append([start, end])
    return merged


def _silent_split_chunked(y, sr, chunk_sec, overlap_sec=2.0):
    """librosa.effects.split on long audio in bounded-memory chunks.

    librosa.effects.split frames the ENTIRE signal into a (frame_length,
    n_frames) matrix before it looks for silence. For a ~92-minute clip at
    22050 Hz that matrix alone is 2.85 GiB, and MAX_VIDEO_DURATION_SEC in
    this project allows up to 3 hours — which would need far more RAM than
    that. This crashed partway through a real run once clips got long
    enough to make memory pressure visible, but the underlying problem
    exists at any clip length; it was only a matter of time.

    Splitting into fixed-size chunks bounds the matrix size regardless of
    total file length. Chunks overlap slightly so a non-silent interval
    that straddles a chunk boundary doesn't get cut in two; merge_intervals
    then joins the resulting duplicate/adjacent pieces back together.
    """
    chunk_len = int(chunk_sec * sr)
    overlap = int(overlap_sec * sr)
    n = len(y)

    raw = []
    pos = 0
    while pos < n:
        end = min(pos + chunk_len, n)
        for s, e in librosa.effects.split(y[pos:end], top_db=config.SILENCE_TOP_DB):
            raw.append((s + pos, e + pos))
        if end >= n:
            break
        pos = end - overlap

    raw.sort(key=lambda iv: iv[0])
    return raw


def split_into_clips(y, sr):
    """Split a long audio signal into a list of (start_sample, end_sample) clip
    boundaries, roughly config.CLIP_TARGET_SEC long, using non-silent intervals."""
    chunk_sec = getattr(config, "SILENCE_SPLIT_CHUNK_SEC", 300.0)
    intervals = _silent_split_chunked(y, sr, chunk_sec)
    intervals = merge_intervals(intervals, sr,
                                getattr(config, "SILENCE_MERGE_GAP_SEC", 1.0))

    clip_len = int(config.CLIP_TARGET_SEC * sr)
    min_len = int(config.CLIP_MIN_SEC * sr)

    clips = []
    for start, end in intervals:
        seg_len = end - start
        if seg_len < min_len:
            continue  # too short a non-silent chunk to bother with
        # Walk through this non-silent interval in clip_len-sized windows
        pos = start
        while pos + min_len <= end:
            clip_end = min(pos + clip_len, end)
            clips.append((pos, clip_end))
            pos = clip_end
    return clips


def rms_energy(y):
    return float(np.sqrt(np.mean(y ** 2)))


def scan_file(filepath, energy_pool):
    """Pass 1 worker. Returns a list of (start, end, idx, energy) tuples —
    sample-index BOUNDARIES and their energy, never the audio itself.

    The previous version returned the actual waveform for every clip, and
    main() kept every video's clips in memory for the whole run before
    writing anything. At CLIP_TARGET_SEC=30 (4.3x the array size of the
    original 7s design) that meant holding the audio for nearly the entire
    ~180-video, ~17,000-clip dataset in RAM at once — tens of gigabytes.
    That's what was failing: allocations of only a few hundred MB were
    getting refused late in the run because everything before them had
    already eaten the available memory. This function keeps only numbers;
    save_file (pass 2) reloads the audio and slices it.
    """
    try:
        y, sr = librosa.load(filepath, sr=config.SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"  ! Failed to load {filepath}: {e}")
        return None

    clip_bounds = split_into_clips(y, sr)
    out = []
    for i, (start, end) in enumerate(clip_bounds):
        energy = rms_energy(y[start:end])
        energy_pool.append(energy)
        out.append((start, end, i, energy))
    return out


def save_file(filepath, bounds_info, threshold, out_dir, cls, video_id, writer):
    """Pass 2 worker. Reloads the audio ONCE for this file and writes only
    the clips whose cached energy clears the threshold.

    A second load per file costs real time (disk read + resample happen
    twice overall), but it is what keeps peak memory bounded to one file's
    audio at a time instead of the whole dataset's.
    """
    try:
        y, sr = librosa.load(filepath, sr=config.SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"  ! Failed to reload {filepath} in pass 2: {e}")
        return 0, 0

    kept = dropped = 0
    for start, end, idx, energy in bounds_info:
        if energy < threshold:
            dropped += 1
            continue
        clip = y[start:end]
        clip_id = f"{video_id}_{idx:03d}"
        out_path = os.path.join(out_dir, f"{clip_id}.wav")
        sf.write(out_path, clip, sr)
        writer.writerow({
            "clip_id": clip_id,
            "class": cls,
            "video_id": video_id,
            "start_sec": round(start / sr, 2),
            "end_sec": round(end / sr, 2),
            "duration_sec": round((end - start) / sr, 2),
            "rms_energy": round(energy, 6),
            "filepath": out_path,
        })
        kept += 1
    return kept, dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing", action="store_true",
                        help="Proceed even when more than 20%% of the videos "
                             "listed in raw_metadata.csv have no audio on disk.")
    args = parser.parse_args()

    ensure_dirs()

    if not os.path.exists(config.RAW_METADATA_CSV):
        raise SystemExit(f"Missing {config.RAW_METADATA_CSV} — run 01_download_videos.py first.")

    with open(config.RAW_METADATA_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    dropped_ids, relabel_map = load_corrections()

    # Pass 1: scan everything, collect BOUNDARIES + energies only (no audio)
    # to compute a global threshold. See scan_file's docstring for why.
    all_bounds = []  # (row, bounds_info)
    energy_pool = []
    missing = []
    load_failed = []
    skipped_by_correction = 0
    print(f"Pass 1/2: scanning at CLIP_TARGET_SEC={config.CLIP_TARGET_SEC}s...")
    for row in tqdm(rows):
        vid = row["video_id"]

        if vid in dropped_ids:
            skipped_by_correction += 1
            continue

        # Relabel BEFORE segmenting, so clips are written into the correct
        # class directory and carry the corrected class downstream.
        if vid in relabel_map:
            row["class"] = relabel_map[vid]

        filepath = row["filepath"]
        if not os.path.exists(filepath):
            # Record rather than silently skip. This bare `continue` is how
            # 107 of 180 videos disappeared without a warning.
            missing.append(vid)
            continue

        bounds_info = scan_file(filepath, energy_pool)
        if bounds_info is None:
            # Loaded-but-failed (e.g. the OOM this function used to cause).
            # Distinct from "missing": the audio file is fine, so simply
            # rerunning — now that pass 1 no longer hoards memory — should
            # recover it without needing --repair.
            load_failed.append(vid)
            continue
        all_bounds.append((row, bounds_info))

    if skipped_by_correction:
        print(f"Skipped {skipped_by_correction} video(s) marked 'drop' in "
              f"label_corrections.csv")

    if missing:
        pct = 100.0 * len(missing) / len(rows)
        report = os.path.join(config.DATA_ROOT, "missing_audio.txt")
        with open(report, "w", encoding="utf-8") as f:
            f.write("\n".join(missing))
        print("")
        print("=" * 72)
        print(f"  {len(missing)} of {len(rows)} videos ({pct:.0f}%) are listed in")
        print( "  raw_metadata.csv but have NO audio file on disk.")
        print("=" * 72)
        print(f"  Full list: {report}")
        print( "  Recover:   python 01_download_videos.py --repair")
        print( "  Then:      python 01_download_videos.py --prune")
        print("")
        if pct > 20 and not args.allow_missing:
            raise SystemExit(
                f"Refusing to continue: {pct:.0f}% of the collected dataset has "
                f"no audio. Segmenting now would quietly build a model on a "
                f"fraction of your data and every count downstream would be "
                f"wrong. Run --repair first, or pass --allow-missing if you "
                f"have decided those videos are gone for good.")

    if load_failed:
        report = os.path.join(config.DATA_ROOT, "load_failed.txt")
        with open(report, "w", encoding="utf-8") as f:
            f.write("\n".join(load_failed))
        print(f"\n{len(load_failed)} video(s) loaded but failed during "
              f"scanning (see {report}). These have audio on disk — just "
              f"rerun this script to retry them, no need for --repair.")

    if not energy_pool:
        raise SystemExit(
            f"No clips produced. With CLIP_MIN_SEC={config.CLIP_MIN_SEC}s, every "
            f"non-silent stretch came out shorter than that. Try raising "
            f"config.SILENCE_MERGE_GAP_SEC (currently "
            f"{getattr(config, 'SILENCE_MERGE_GAP_SEC', 1.0)}s) so brief dips "
            f"inside a performance stop being treated as boundaries, or lower "
            f"SILENCE_TOP_DB (currently {config.SILENCE_TOP_DB}).")

    energy_threshold = float(np.percentile(energy_pool, config.MIN_CLIP_ENERGY_PERCENTILE))
    print(f"Energy threshold (bottom {config.MIN_CLIP_ENERGY_PERCENTILE}th percentile): {energy_threshold:.5f}")

    # Pass 2: reload each file once, write only the clips that clear the
    # threshold. Peak memory here is one file's audio, not the dataset's.
    print("Pass 2/2: reloading + saving clips...")
    # Overwrite, not append: this script reprocesses ALL rows in
    # raw_metadata.csv every run (not just newly-added videos), so append
    # mode would duplicate every existing clip's row each time new source
    # videos are added and this is rerun (same bug we found and fixed in
    # 03_generate_spectrograms.py's labels.csv writer).
    with open(config.SEGMENTS_METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["clip_id", "class", "video_id", "start_sec", "end_sec", "duration_sec",
                      "rms_energy", "filepath"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        kept, dropped = 0, 0
        for row, bounds_info in tqdm(all_bounds):
            cls, video_id = row["class"], row["video_id"]
            out_dir = os.path.join(config.SEGMENTS_DIR, cls)
            k, d = save_file(row["filepath"], bounds_info, energy_threshold,
                             out_dir, cls, video_id, writer)
            kept += k
            dropped += d

    print(f"\nDone. Kept {kept} clips, dropped {dropped} low-energy/junk clips.")
    print(f"Metadata written to {config.SEGMENTS_METADATA_CSV}")


if __name__ == "__main__":
    main()