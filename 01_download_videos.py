"""
01_download_videos.py

Bulk-downloads audio from YouTube for each melam class using yt-dlp,
based on the search queries defined in config.py.

Requires: yt-dlp (pip install yt-dlp), ffmpeg installed on the system.

Usage:
    python 01_download_videos.py
    python 01_download_videos.py --class Panchari   # download just one class
    python 01_download_videos.py --repair           # re-fetch missing audio
    python 01_download_videos.py --prune            # drop rows with no audio
"""

import argparse
import csv
import os
import sys

from yt_dlp import YoutubeDL

import config


def ensure_dirs():
    os.makedirs(config.RAW_AUDIO_DIR, exist_ok=True)
    for cls in config.CLASSES:
        os.makedirs(os.path.join(config.RAW_AUDIO_DIR, cls), exist_ok=True)


def load_existing_ids():
    """Avoid re-downloading videos already recorded in the metadata CSV."""
    seen = set()
    if os.path.exists(config.RAW_METADATA_CSV):
        with open(config.RAW_METADATA_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.add(row["video_id"])
    return seen


def download_class(cls, seen_ids, writer):
    out_dir = os.path.join(config.RAW_AUDIO_DIR, cls)
    failures = []

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        # Write WAV at exactly what the pipeline loads: config.SAMPLE_RATE,
        # mono. Without this yt-dlp extracts at the source rate in stereo —
        # 44.1kHz 16-bit stereo over 162 hours of audio is 96 GB, and
        # librosa immediately downmixes and resamples all of it away in
        # 02_segment_audio.py. Zero information loss, 4x less disk.
        "postprocessor_args": {
            "extractaudio": ["-ac", "1", "-ar", str(config.SAMPLE_RATE)],
        },
        "noplaylist": True,
        "quiet": False,
        "ignoreerrors": True,
        "match_filter": _duration_filter,
        # Lets yt-dlp auto-fetch the JS challenge-solver script it runs via Deno.
        # Without this, YouTube's signature/n-parameter challenges fail even
        # with Deno installed, causing widespread 403 errors.
        "remote_components": {"ejs:github"},
        # Fallback list of player clients, tried in order. Reordered after
        # yt-dlp issue #17348 (Aug 2026): android_vr now requires a GVS PO
        # Token for everything except the legacy combined format 18, and
        # android has largely moved to a PO-token-only "SABR" streaming mode
        # too — so both are demoted below web/tv here, which are more likely
        # to have a usable format without a token server set up.
        "extractor_args": {"youtube": {"player_client": ["web", "tv", "android", "android_vr"]}},
    }

    # Optional auth — either sidesteps YouTube's PO Token requirement
    # without needing a token-generation server (see
    # https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide). Prefer
    # YT_DLP_COOKIE_FILE on Windows — reading the browser's live cookie DB
    # directly (YT_DLP_COOKIES_FROM_BROWSER) is unreliable there.
    if config.YT_DLP_COOKIE_FILE:
        ydl_opts["cookiefile"] = config.YT_DLP_COOKIE_FILE
    elif config.YT_DLP_COOKIES_FROM_BROWSER:
        ydl_opts["cookiesfrombrowser"] = (config.YT_DLP_COOKIES_FROM_BROWSER,)

    for query in config.SEARCH_QUERIES[cls]:
        search_target = f"ytsearch{config.MAX_VIDEOS_PER_QUERY}:{query}"
        print(f"\n[{cls}] Searching: {query!r}")

        with YoutubeDL({**ydl_opts, "quiet": True, "skip_download": True, "extract_flat": False}) as ydl:
            try:
                info = ydl.extract_info(search_target, download=False)
            except Exception as e:
                print(f"  ! Search failed for {query!r}: {e}")
                continue

        entries = info.get("entries", []) if info else []
        for entry in entries:
            if entry is None:
                continue
            vid = entry.get("id")
            if not vid or vid in seen_ids:
                continue

            duration = entry.get("duration") or 0
            if duration and not (config.MIN_VIDEO_DURATION_SEC <= duration <= config.MAX_VIDEO_DURATION_SEC):
                print(f"  - Skipping {vid} (duration {duration}s out of range)")
                continue

            print(f"  > Downloading {vid}: {entry.get('title', '')[:70]}")
            expected = os.path.join(out_dir, f"{vid}.wav")

            with YoutubeDL(ydl_opts) as ydl_dl:
                try:
                    ret = ydl_dl.download([f"https://www.youtube.com/watch?v={vid}"])
                except Exception as e:
                    print(f"  ! Download raised for {vid}: {e}")
                    ret = 1

            # ydl_opts sets ignoreerrors=True, which means download() does NOT
            # raise when a fetch fails — it logs, returns a non-zero code and
            # carries on. So the except block above almost never fires, and
            # before this check the code fell straight through to
            # writer.writerow() and recorded a video that was never written to
            # disk. That is how raw_metadata.csv came to list 180 videos when
            # only 73 audio files existed. The only trustworthy test is
            # whether the file actually landed.
            if ret != 0 or not os.path.exists(expected) or os.path.getsize(expected) == 0:
                print(f"  ! FAILED {vid} — no usable audio at {expected}")
                failures.append(vid)
                continue

            seen_ids.add(vid)
            writer.writerow({
                "video_id": vid,
                "title": entry.get("title", ""),
                "class": cls,
                "query": query,
                "duration_sec": duration,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "filepath": expected,
            })

    if failures:
        shown = ", ".join(failures[:8])
        more = f" (+{len(failures) - 8} more)" if len(failures) > 8 else ""
        print(f"\n[{cls}] {len(failures)} download(s) failed: {shown}{more}")
    return failures


def _build_opts(cls):
    """The same yt-dlp options download_class builds, for repair runs."""
    out_dir = os.path.join(config.RAW_AUDIO_DIR, cls)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        # Write WAV at exactly what the pipeline loads: config.SAMPLE_RATE,
        # mono. Without this yt-dlp extracts at the source rate in stereo —
        # 44.1kHz 16-bit stereo over 162 hours of audio is 96 GB, and
        # librosa immediately downmixes and resamples all of it away in
        # 02_segment_audio.py. Zero information loss, 4x less disk.
        "postprocessor_args": {
            "extractaudio": ["-ac", "1", "-ar", str(config.SAMPLE_RATE)],
        },
        "noplaylist": True,
        "quiet": False,
        "ignoreerrors": True,
        "remote_components": {"ejs:github"},
        "extractor_args": {"youtube": {"player_client": ["web", "tv", "android", "android_vr"]}},
    }
    if config.YT_DLP_COOKIE_FILE:
        opts["cookiefile"] = config.YT_DLP_COOKIE_FILE
    elif config.YT_DLP_COOKIES_FROM_BROWSER:
        opts["cookiesfrombrowser"] = (config.YT_DLP_COOKIES_FROM_BROWSER,)
    return opts


def missing_rows(rows):
    """Rows in raw_metadata.csv whose audio file is absent or empty."""
    out = []
    for row in rows:
        fp = row.get("filepath", "")
        if not fp or not os.path.exists(fp) or os.path.getsize(fp) == 0:
            out.append(row)
    return out


def repair(rows):
    """Re-fetch any video listed in raw_metadata.csv with no audio on disk.

    Downloads fail for ordinary reasons — throttling, a region lock, a video
    pulled between the search and the fetch. Repair is a normal part of
    collection, not an error path. Run it before anything else: it is the
    only step that ADDS data rather than removing it.
    """
    todo = missing_rows(rows)
    if not todo:
        print("Nothing to repair — every row has audio on disk.")
        return []

    print(f"{len(todo)} of {len(rows)} rows have no audio. Re-fetching...\n")
    still_missing = []
    for i, row in enumerate(todo, 1):
        vid, cls = row["video_id"], row["class"]
        print(f"  [{i}/{len(todo)}] {vid} ({cls}): {row['title'][:60]}")
        with YoutubeDL(_build_opts(cls)) as ydl:
            try:
                ydl.download([row["url"]])
            except Exception as e:
                print(f"    ! {e}")
        fp = row["filepath"]
        if not os.path.exists(fp) or os.path.getsize(fp) == 0:
            still_missing.append(vid)

    print(f"\nRepair complete. Recovered {len(todo) - len(still_missing)}, "
          f"{len(still_missing)} still unavailable.")
    if still_missing:
        print("Still missing: " + ", ".join(still_missing[:20]))
        print("These are likely gone for good. Run --prune to drop them so "
              "raw_metadata.csv describes what you actually have.")
    return still_missing


def prune(rows):
    """Rewrite raw_metadata.csv without rows whose audio is absent.

    Leaving dead rows in place makes the dataset look 2.5x bigger than it
    is, and every downstream count inherits that lie.
    """
    dead = {r["video_id"] for r in missing_rows(rows)}
    if not dead:
        print("Nothing to prune.")
        return
    kept = [r for r in rows if r["video_id"] not in dead]
    fieldnames = ["video_id", "title", "class", "query", "duration_sec", "url", "filepath"]
    with open(config.RAW_METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)
    print(f"Pruned {len(dead)} dead rows. raw_metadata.csv now lists {len(kept)} videos.")
    print("Per class: ")
    from collections import Counter
    for cls, n in sorted(Counter(r["class"] for r in kept).items()):
        print(f"  {cls:15s} {n}")


def _duration_filter(info_dict):
    """yt-dlp match_filter hook — return None to allow, a string to skip."""
    duration = info_dict.get("duration")
    if duration is None:
        return None
    if duration > config.MAX_VIDEO_DURATION_SEC:
        return "Video too long"
    if duration < config.MIN_VIDEO_DURATION_SEC:
        return "Video too short"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="single_class", choices=config.CLASSES, default=None,
                         help="Download only this class instead of all classes")
    parser.add_argument("--repair", action="store_true",
                        help="Re-fetch videos listed in raw_metadata.csv whose "
                             "audio file is missing, then exit.")
    parser.add_argument("--prune", action="store_true",
                        help="Remove rows from raw_metadata.csv whose audio is "
                             "missing, then exit. Run after --repair.")
    args = parser.parse_args()

    ensure_dirs()

    if args.repair or args.prune:
        if not os.path.exists(config.RAW_METADATA_CSV):
            raise SystemExit(f"No {config.RAW_METADATA_CSV} to work from.")
        with open(config.RAW_METADATA_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if args.repair:
            repair(rows)
        if args.prune:
            with open(config.RAW_METADATA_CSV, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            prune(rows)
        return

    seen_ids = load_existing_ids()

    write_header = not os.path.exists(config.RAW_METADATA_CSV)
    with open(config.RAW_METADATA_CSV, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["video_id", "title", "class", "query", "duration_sec", "url", "filepath"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        targets = [args.single_class] if args.single_class else config.CLASSES
        all_failures = []
        for cls in targets:
            all_failures.extend(download_class(cls, seen_ids, writer) or [])

    print(f"\nDone. Metadata written to {config.RAW_METADATA_CSV}")
    if all_failures:
        print(f"{len(all_failures)} video(s) failed to download and were NOT "
              f"written to metadata. Re-try them with --repair.")


if __name__ == "__main__":
    sys.exit(main())