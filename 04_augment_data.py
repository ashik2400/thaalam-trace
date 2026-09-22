"""
04_augment_data.py

Optional step — run this AFTER 02_segment_audio.py, BEFORE 03_generate_spectrograms.py
(or re-run 03 after this to spectrogram the new augmented clips too).

If any class has fewer clips than config.TARGET_CLIPS_PER_CLASS, this script
generates augmented copies (pitch shift / time stretch / noise injection) of
existing clips from that class until it hits the target — helps balance the
dataset instead of one melam class dominating training.

Requires: audiomentations, librosa, soundfile, numpy, tqdm

Usage:
    python 04_augment_data.py
"""

import csv
import os
import random
from collections import defaultdict

import librosa
import soundfile as sf
from audiomentations import Compose, PitchShift, TimeStretch, AddGaussianNoise
from tqdm import tqdm

import config

augment_pipeline = Compose([
    PitchShift(min_semitones=-2, max_semitones=2, p=0.6),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.6),
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.008, p=0.4),
])


def ensure_dirs():
    os.makedirs(config.AUGMENTED_DIR, exist_ok=True)
    for cls in config.CLASSES:
        os.makedirs(os.path.join(config.AUGMENTED_DIR, cls), exist_ok=True)


def main():
    ensure_dirs()

    if not os.path.exists(config.SEGMENTS_METADATA_CSV):
        raise SystemExit(f"Missing {config.SEGMENTS_METADATA_CSV} — run 02_segment_audio.py first.")

    with open(config.SEGMENTS_METADATA_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_class = defaultdict(list)
    for row in rows:
        by_class[row["class"]].append(row)

    write_header = not os.path.exists(config.SEGMENTS_METADATA_CSV)
    with open(config.SEGMENTS_METADATA_CSV, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["clip_id", "class", "video_id", "start_sec", "end_sec", "duration_sec",
                      "rms_energy", "filepath"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        for cls in config.CLASSES:
            existing = by_class.get(cls, [])
            n_existing = len(existing)
            n_needed = max(0, config.TARGET_CLIPS_PER_CLASS - n_existing)

            if n_needed == 0:
                print(f"{cls}: {n_existing} clips, already at/above target — skipping.")
                continue

            print(f"{cls}: {n_existing} clips, generating {n_needed} augmented clips...")
            out_dir = os.path.join(config.AUGMENTED_DIR, cls)

            for i in tqdm(range(n_needed)):
                src_row = random.choice(existing)
                y, sr = librosa.load(src_row["filepath"], sr=config.SAMPLE_RATE, mono=True)
                y_aug = augment_pipeline(samples=y, sample_rate=sr)

                clip_id = f"aug_{src_row['clip_id']}_{i:03d}"
                out_path = os.path.join(out_dir, f"{clip_id}.wav")
                sf.write(out_path, y_aug, sr)

                writer.writerow({
                    "clip_id": clip_id,
                    "class": cls,
                    "video_id": src_row["video_id"] + "_aug",
                    "start_sec": src_row["start_sec"],
                    "end_sec": src_row["end_sec"],
                    "duration_sec": src_row["duration_sec"],
                    "rms_energy": "",
                    "filepath": out_path,
                })

    print("\nDone. Augmented clips added to segments_metadata.csv — "
          "re-run 03_generate_spectrograms.py to spectrogram them.")


if __name__ == "__main__":
    main()
