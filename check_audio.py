"""
check_audio.py — what is actually in data/01_raw_audio right now?

Safe to run while a download is in progress, in a second terminal.
Reports nothing, deletes nothing; just looks.

Usage:
    python check_audio.py
"""

import os
import csv
from collections import Counter

import config


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def wav_info(path):
    """Sample rate, channels and duration from the WAV header only.

    Reading the header avoids loading a 500 MB file into memory just to
    learn its format.
    """
    try:
        import wave
        with wave.open(path, "rb") as w:
            return w.getframerate(), w.getnchannels(), w.getnframes() / w.getframerate()
    except Exception:
        return None, None, None


def main():
    if not os.path.exists(config.RAW_METADATA_CSV):
        raise SystemExit(f"No {config.RAW_METADATA_CSV}")

    with open(config.RAW_METADATA_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    have, missing, empty = [], [], []
    for r in rows:
        fp = r["filepath"]
        if not os.path.exists(fp):
            missing.append(r)
        elif os.path.getsize(fp) == 0:
            empty.append(r)
        else:
            have.append(r)

    print(f"raw_metadata.csv lists {len(rows)} videos")
    print(f"  audio present : {len(have)}")
    print(f"  audio missing : {len(missing)}")
    if empty:
        print(f"  ZERO BYTES    : {len(empty)}  <- failed extracts, will be re-fetched")
    print()

    # Leftover downloads that never got converted or cleaned up.
    junk = []
    for root, _, files in os.walk(config.RAW_AUDIO_DIR):
        for fn in files:
            if not fn.lower().endswith(".wav"):
                junk.append(os.path.join(root, fn))
    if junk:
        total = sum(os.path.getsize(p) for p in junk)
        print(f"LEFTOVER non-WAV files: {len(junk)} ({human(total)})")
        print("These are interrupted downloads. yt-dlp normally deletes the .mp4")
        print("after extracting audio, but a Ctrl+C mid-download leaves them.")
        print("Safe to delete — the .wav is what the pipeline reads.")
        for p in junk[:10]:
            print(f"  {os.path.basename(p)}  {human(os.path.getsize(p))}")
        if len(junk) > 10:
            print(f"  ... and {len(junk) - 10} more")
        print()
    else:
        print("No leftover .mp4/.part files — yt-dlp cleaned up properly.\n")

    # Format check: are the new downloads honouring the postprocessor args?
    print("WAV format of downloaded files")
    print("-" * 58)
    fmts = Counter()
    total_bytes = 0
    sample_rows = []
    for r in have:
        fp = r["filepath"]
        sz = os.path.getsize(fp)
        total_bytes += sz
        sr, ch, dur = wav_info(fp)
        fmts[(sr, ch)] += 1
        sample_rows.append((os.path.basename(fp), sr, ch, dur, sz,
                            os.path.getmtime(fp)))

    for (sr, ch), n in sorted(fmts.items(), key=lambda x: -x[1]):
        tag = ""
        if sr == config.SAMPLE_RATE and ch == 1:
            tag = "  <- correct, postprocessor_args working"
        elif sr:
            tag = "  <- downloaded BEFORE the fix (still usable, just 4x bigger)"
        print(f"  {str(sr):>7} Hz  {ch} ch   {n:4d} files{tag}")
    print(f"\n  total on disk: {human(total_bytes)}")

    # The five most recent files tell you whether the CURRENT run is doing
    # the right thing, which the aggregate above can hide.
    print("\nFive most recently written files (is the current run correct?)")
    print("-" * 58)
    for name, sr, ch, dur, sz, _ in sorted(sample_rows, key=lambda x: -x[5])[:5]:
        mins = f"{dur/60:.0f} min" if dur else "?"
        print(f"  {name:<20} {str(sr):>7} Hz  {ch} ch  {mins:>7}  {human(sz)}")

    if missing:
        print(f"\n{len(missing)} still to fetch. Per class:")
        for cls, n in sorted(Counter(r["class"] for r in missing).items()):
            print(f"  {cls:15s} {n}")


if __name__ == "__main__":
    main()
