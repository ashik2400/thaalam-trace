# ThaalamTrace — Dataset Automation Pipeline

Automates: YouTube search & download → audio segmentation → silence/junk filtering →
mel-spectrogram generation → labeled dataset ready for CNN training.

## Setup (do this first, ~10 min)

```bash
# 1. ffmpeg is required by yt-dlp for audio extraction
sudo apt-get install ffmpeg      # Linux
# brew install ffmpeg            # Mac

# 2. Python dependencies
pip install -r requirements.txt
```

## Pipeline order

```bash
python 01_download_videos.py          # downloads audio for all 4 classes
python 02_segment_audio.py            # splits into 5-10s clips, filters junk
python 04_augment_data.py             # OPTIONAL — balances under-filled classes
python 03_generate_spectrograms.py    # generates .npy (train) + .png (present) + labels.csv
python 05_split_dataset.py            # train/val/test split, grouped by video
python 06_train_cnn.py                # trains the CNN, saves best checkpoint
python 07_evaluate_model.py           # tests on held-out set, confusion matrix
```

Each script is resumable/incremental — re-running `01` skips videos already
recorded in `raw_metadata.csv`, so you can stop and restart across days
without losing progress or re-downloading.

Everything gets written under `data/`:

```
data/
  01_raw_audio/<class>/*.wav          full downloaded audio
  02_segments/<class>/*.wav           5-10 sec trimmed clips
  03_spectrograms/<class>/npy/*.npy   for CNN training
  03_spectrograms/<class>/png/*.png   for presentation slides
  04_augmented/<class>/*.wav          extra clips if augmentation was run
  raw_metadata.csv                    video-level metadata
  segments_metadata.csv               clip-level metadata
  labels.csv                          spectrogram-level labels
  splits.csv                          train/val/test assignment per clip

models/
  best_model.pt                       trained CNN checkpoint (best val accuracy)
  label_map.json                      class name <-> index mapping
  training_curves.png                 loss/accuracy over epochs
  confusion_matrix.png                test-set confusion matrix
```

## Model training notes

- **Splits are grouped by source video**, not by individual clip — this
  stops clips from the same performance leaking across train/val/test,
  which would otherwise let the model "memorize" a recording's specific
  room noise instead of learning general melam patterns.
- **Class imbalance is handled via weighted loss**, not by
  duplicating/deleting clips. Weights are computed automatically from the
  TRAIN split's class counts (`sklearn.utils.class_weight`) and passed into
  `CrossEntropyLoss` — mistakes on smaller classes (e.g. Pandi) cost more
  during training than mistakes on the largest class (Thayambaka).
- Training auto-stops early if validation accuracy hasn't improved in
  `EARLY_STOP_PATIENCE` epochs (default 6) — prevents overfitting and saves
  time. The best checkpoint (by val accuracy) is always what gets saved.
- `07_evaluate_model.py` only touches the TEST split, which the model has
  never seen in any form during training. Its printed accuracy is the
  number worth reporting to your mam — training/val accuracy alone can be
  misleadingly optimistic.

## 5-Day Plan

**Day 1 — Download**
Run `01_download_videos.py`. Adjust `SEARCH_QUERIES` and `MAX_VIDEOS_PER_QUERY`
in `config.py` if a class is coming up short. Aim for 15–20 usable source
videos per class — longer full-performance videos give you more clips per
download than short highlight clips.

**Day 2 — Manual spot-check**
Listen to the first/last ~30 sec of each downloaded file in
`data/01_raw_audio/<class>/`. Delete anything that's mostly talking,
crowd noise, or clearly mislabeled (e.g. a Panchavadyam video that's
actually Pandi). This is the one step worth doing by hand — a weak-labeled
dataset with 10-15% noise is fine, but wildly wrong labels aren't.

**Day 3 — Segment**
Run `02_segment_audio.py`. Fully automated — trims silence, splits into
clips, drops the bottom 10% by energy (near-silent/junk). Check the printed
kept/dropped counts.

**Day 4 — Spectrograms + balance**
Run `03_generate_spectrograms.py`. Check the class-balance printout at the
end. If one class is far behind the others, run `04_augment_data.py` first
to top it up, then re-run `03`.

**Day 5 — QA + present**
- Open a handful of `.png` spectrograms per class in `data/03_spectrograms/`
  — sanity check they look visually distinct between classes.
- Report to your mam: total clips, per-class counts, a few sample
  spectrogram images, and the pipeline diagram (download → segment → filter
  → spectrogram → label).

