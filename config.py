"""
config.py — Central config for the ThaalamTrace dataset pipeline.
Edit this file to tune classes, search queries, targets, and paths.
"""

import os

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
CLASSES = ["Panchari", "Pandi", "Panchavadyam", "Thayambaka"]

# Search queries per class. Add/remove queries to widen or narrow results.
# yt-dlp will pull the top N results for EACH query (see MAX_VIDEOS_PER_QUERY).
SEARCH_QUERIES = {
    "Panchari": [
        "Panchari melam full performance",
        "Panchari melam Thrissur Pooram",
        "Panchari melam Kerala temple festival",
        "പഞ്ചാരി മേളം",  # Malayalam script query — often surfaces different uploads
        "Panchari melam kalasam",
        "Panchari melam pooram",
    ],
    "Pandi": [
        "Pandi melam full performance",
        "Pandi melam Thrissur Pooram",
        "Pandi melam Kerala temple",
        "പാണ്ടി മേളം",
        # Pandi has the fewest unique source videos of any class — these
        # add spelling variants and specific event names rather than more
        # phrasings of the same query, to surface genuinely different
        # uploads instead of re-hitting the same handful of videos.
        "Pandimelam",  # single-word spelling, common in upload titles
        "പാണ്ടിമേളം",  # single-word Malayalam spelling
        "Pandi melam Guruvayur",
        "Pandi melam Kodungallur",
        "Pandi melam vela",
        "Pandi melam kalasam",
    ],
    "Panchavadyam": [
        "Panchavadyam full performance",
        "Panchavadyam Thrissur Pooram",
        "Panchavadyam temple festival Kerala",
        "പഞ്ചവാദ്യം",
    ],
    "Thayambaka": [
        "Thayambaka full performance",
        "Thayambaka Kerala temple",
        "Thayambaka solo chenda",
        "തായമ്പക",
    ],
}

# Browser-cookie-DB reading (YT_DLP_COOKIES_FROM_BROWSER) is unreliable on
# Windows — Chrome's cookie database is OS-encrypted and yt-dlp sometimes
# can't copy/decrypt it even with the browser closed (yt-dlp issue #7271).
# Left as None; use YT_DLP_COOKIE_FILE below instead, which reads a plain
# exported cookies.txt file and sidesteps that problem entirely.
YT_DLP_COOKIES_FROM_BROWSER = None

# Path to a Netscape-format cookies.txt file, exported from your browser
# while logged into YouTube. More reliable than YT_DLP_COOKIES_FROM_BROWSER
# on Windows since it's just a text file, not a live encrypted database.
# How to get one: install a browser extension like "Get cookies.txt LOCALLY"
# (Chrome/Firefox), open youtube.com while logged in, click the extension,
# export/download the cookies.txt file, and put its path here, e.g.:
#   YT_DLP_COOKIE_FILE = r"C:\Users\ashik\Documents\Sem 3 Projects\ThaalamTrace\cookies.txt"
# Set to None to skip.
YT_DLP_COOKIE_FILE = None

# How many results to pull per search query (per class this multiplies by
# len(SEARCH_QUERIES[class]), so 5 queries x 6 results = ~30 candidate videos)
MAX_VIDEOS_PER_QUERY = 16

# Skip videos longer than this (avoid multi-hour uploads that bloat storage)
MAX_VIDEO_DURATION_SEC = 60 * 180  # 3 hours
# Skip videos shorter than this (too short to be a real performance)
MIN_VIDEO_DURATION_SEC = 60*10

# ---------------------------------------------------------------------------
# Paths — everything lives under DATA_ROOT
# ---------------------------------------------------------------------------
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

RAW_AUDIO_DIR = os.path.join(DATA_ROOT, "01_raw_audio")       # full downloaded audio
SEGMENTS_DIR = os.path.join(DATA_ROOT, "02_segments")          # 5-10s trimmed clips
SPECTROGRAM_DIR = os.path.join(DATA_ROOT, "03_spectrograms")   # .npy + .png
AUGMENTED_DIR = os.path.join(DATA_ROOT, "04_augmented")        # extra augmented clips

RAW_METADATA_CSV = os.path.join(DATA_ROOT, "raw_metadata.csv")
SEGMENTS_METADATA_CSV = os.path.join(DATA_ROOT, "segments_metadata.csv")
LABELS_CSV = os.path.join(DATA_ROOT, "labels.csv")  # final file used for CNN training

# Video-level corrections applied at segmentation time (02). Two columns
# matter: action=relabel (with new_class) and action=drop. Sourced from
# expert review — see review_sheet.html.
LABEL_CORRECTIONS_CSV = os.path.join(DATA_ROOT, "label_corrections.csv")

# Maps video_id -> performance_id. Videos sharing a performance_id are the
# same performance (re-uploads, or one performance split into parts) and
# MUST stay in the same split. See Fix 2.
PERFORMANCE_GROUPS_CSV = os.path.join(DATA_ROOT, "performance_groups.csv")

# ---------------------------------------------------------------------------
# Audio processing params
# ---------------------------------------------------------------------------
SAMPLE_RATE = 22050          # standard for librosa-based audio ML

# Clip length. Raised 7.0 -> 30.0.
#
# Why: what separates Panchari from Pandi is the thalam cycle — 6 beats vs
# 7. In the slow kalams a single beat can last 2-3 seconds, so one cycle
# runs 15-20s. A 7-second window physically cannot contain one cycle, which
# made a large share of clips unclassifiable no matter what the model did.
# Segmentation cuts contiguous windows across the whole performance, so
# every video contributed kalam-1 clips (impossible) alongside kalam-5 clips
# (easy), all with the same label. That is irreducible label noise.
#
# 30s holds at least one full cycle through kalam 2, and several by kalam 3.
# Kalam 1 is still marginal. Probe with 11_probe_rhythm_features.py before
# committing to a full training run.
CLIP_MIN_SEC = 20.0
CLIP_MAX_SEC = 35.0
CLIP_TARGET_SEC = 30.0
SILENCE_TOP_DB = 30          # librosa.effects.split threshold (higher = stricter)

# Non-silent intervals separated by less than this are joined before
# windowing. Needed once CLIP_MIN_SEC went to 20s: librosa.effects.split
# cuts at every dip below threshold, and melam has many — the space between
# strokes in a slow kalam, a breath in the kuzhal. Without merging, almost
# every interval falls short of 20s and gets discarded.
SILENCE_MERGE_GAP_SEC = 1.0

# librosa.effects.split loads the WHOLE signal into one frame matrix before
# it looks for silence. For a long recording that matrix is enormous — a
# ~92-minute clip needs ~2.85 GiB just for this step, and this project
# allows videos up to 3 hours (MAX_VIDEO_DURATION_SEC). 02_segment_audio.py
# processes audio in chunks of this length instead, bounding memory use
# regardless of how long the source video is. 300s (5 min) keeps peak usage
# under ~300 MB; raise it only if you have RAM to spare and want fewer,
# slightly cheaper librosa calls.
SILENCE_SPLIT_CHUNK_SEC = 300.0

MIN_CLIP_ENERGY_PERCENTILE = 10  # discard clips in bottom 10% RMS energy (likely junk)

# Mel-spectrogram params
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Augmentation — target clip count per class (script tops up under-filled classes)
TARGET_CLIPS_PER_CLASS = 200

# ---------------------------------------------------------------------------
# CNN training params
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SPLITS_CSV = os.path.join(DATA_ROOT, "splits.csv")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
LABEL_MAP_JSON = os.path.join(MODEL_DIR, "label_map.json")
CONFUSION_MATRIX_PNG = os.path.join(MODEL_DIR, "confusion_matrix.png")
TRAINING_CURVES_PNG = os.path.join(MODEL_DIR, "training_curves.png")

# Every spectrogram gets padded/cropped to this many time-frames so clips of
# slightly different lengths can be batched together.
#
# DERIVED, never hardcoded. This was previously the literal 300 with a
# comment saying it came from CLIP_TARGET_SEC. It did not — the two could
# drift apart silently, and raising CLIP_TARGET_SEC while this stayed at 300
# would have cropped every clip back down to 7 seconds with no error and no
# warning. 03_generate_spectrograms.py asserts these agree.
FIXED_SPEC_FRAMES = int(CLIP_TARGET_SEC * SAMPLE_RATE / HOP_LENGTH) + 1

# Split ratios — MUST sum to 1.0. Splitting is done by video_id (grouped),
# not by individual clip, so clips from the same performance never end up
# in both train and test (that would leak information and inflate accuracy).
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Lowered 32 -> 8. At CLIP_TARGET_SEC=30 each input is 2 x 128 x 1293
# instead of 2 x 128 x 301 — 4.3x the memory per sample, and the BiGRU
# sequence lengthens in proportion. Raise it back if it fits.
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 6  # stop if val accuracy doesn't improve by EARLY_STOP_MIN_DELTA for this many epochs

# Only affects the early-stopping PATIENCE counter, not which checkpoint
# gets saved (06_train_cnn.py always keeps the literal best val_acc seen,
# however small the gain — see main()). This just decides when to give up.
# Lowered from 0.01 -> 0.003 after 0.01 proved too aggressive: it caused a
# genuine +0.56pp improvement (epoch 4's val_acc=0.6517 vs epoch 1's
# 0.6461) to not reset the patience counter, in a run where that distinction
# mattered.
EARLY_STOP_MIN_DELTA = 0.003

# Adam weight decay (L2 regularization). Tried raising this from 1e-4 to
# 3e-4 to fight overfitting (train_acc was hitting 99%+) — but in a seeded,
# controlled run, train_acc STILL hit 99.76% at 3e-4 (no real improvement)
# while Pandi/Thayambaka recall got meaningfully worse. That means 3e-4 was
# adding regularization cost without fixing the actual overfitting cause,
# which is video-fingerprint memorization (too few unique source videos),
# not weight magnitude — L2 was never going to fix that. Reverted to 1e-4.
WEIGHT_DECAY = 1e-4

# Manual multipliers applied on top of the frequency-balanced class
# weights (see main() in 06_train_cnn.py for the full reasoning). Directly
# targets the Thayambaka-as-catch-all bias seen in a fair test run: 1.4x
# for Pandi/Panchari (the two classes it was absorbing) makes misclassifying
# them costlier; 0.6x for Thayambaka makes a true-Thayambaka mistake
# cheaper, so the model has less incentive to default-guess it when unsure.
# Panchavadyam is already performing well (99% recall), left untouched.
# These are a first attempt, not tuned — adjust based on the next
# confusion matrix.
CLASS_WEIGHT_OVERRIDES = {
    "Pandi": 1.4,
    "Panchari": 1.4,
    "Thayambaka": 0.6,
}

# Model architecture: "crnn" (CNN + BiGRU, DEFAULT — see note below),
# "simple_cnn" (built from scratch, ~4 conv blocks), or "resnet18"
# (ImageNet-pretrained, fine-tuned).
#
# Why "crnn" is the default now: melam type is defined mainly by tempo /
# beat-cycle length (e.g. Panchari 96->48->24->12->6 vs Pandi 56->28->14->7),
# not by timbre — both use the same instruments. simple_cnn and resnet18
# both end in global average pooling, which collapses the entire time axis
# into one vector and throws away exactly the rhythmic information that
# distinguishes Pandi from Panchari. "crnn" pools frequency but preserves
# time, then runs a BiGRU over the sequence so tempo/cycle patterns can
# actually be learned. Keep "simple_cnn"/"resnet18" here for comparison runs.
MODEL_ARCHITECTURE = "crnn"
RESNET_FREEZE_BACKBONE = True  # only used when MODEL_ARCHITECTURE == "resnet18"

# Number of input channels the model expects. 2 = [mel-spectrogram,
# tempogram] (see 03_generate_spectrograms.py's make_spectrogram() for why
# a tempogram channel was added — melam type is a rhythm/tempo distinction,
# not a timbre one, and two mel-only architectures both failed to separate
# Pandi from Panchari). Set to 1 only if you regenerate .npy files as
# mel-only again.
N_INPUT_CHANNELS = 2

# ---------------------------------------------------------------------------
# Rhythm channel
# ---------------------------------------------------------------------------
# Which feature fills channel 2. All options output (N_MELS, frames) so they
# stack with the mel channel and 06_train_cnn.py needs no changes.
#
#   "tempogram_legacy"  What this pipeline used until now: win_length=N_MELS,
#                       which caps the lag window at 3.0s. Kept only as a
#                       baseline to compare against — do not ship with it.
#
#   "tempogram_wide"    Same feature, but the lag window is set by
#                       TEMPOGRAM_MAX_LAG_SEC and the lag axis is then
#                       resampled down to N_MELS rows. This decouples HOW FAR
#                       the feature looks from HOW MANY ROWS it outputs. The
#                       old code conflated the two, which is the entire
#                       reason the lag window was 3.0s.
#
#   "beat_norm_acf"     Autocorrelation with the lag axis rescaled into BEATS
#                       instead of seconds. A tempogram reports the dominant
#                       periodicity, which is the beat — and both melams have
#                       beats. The difference is how beats GROUP, 6 vs 7, and
#                       averaging a tempogram erases exactly that. Rescaling
#                       by the estimated beat period puts a 6-grouping at 6
#                       regardless of tempo, making the feature
#                       tempo-invariant so kalam-1 and kalam-5 clips of the
#                       same melam look alike.
#
# Validated on real segmented audio with 11_probe_rhythm_features.py:
#   mel timbre (control)                ~54%
#   tempogram, legacy win=128           ~52-55%
#   tempogram, ~20s lag window          ~50-52%  (wider window alone doesn't help)
#   beat-normalized ACF, full band      ~57-64%
#   beat-normalized ACF, cymbal band    ~66-70%  <- clear winner, and default below
# Chance is 50%. Numbers from clip lengths 7s/15s/30s; cymbal-band improves
# with longer clips as expected (more full cycles observable).
RHYTHM_FEATURE = "beat_norm_acf"

# Longest periodicity the tempogram can see, in seconds. 20s covers a
# 6-beat cycle down to roughly kalam 2. Only used by "tempogram_wide".
TEMPOGRAM_MAX_LAG_SEC = 20.0

# Beat-normalized ACF settings. Only used by "beat_norm_acf".
BEATNORM_MAX_BEATS = 9        # look out to 9 beats — enough to see 6 or 7
BEATNORM_MIN_BEAT_SEC = 0.08  # fastest plausible beat (kalam 5)
BEATNORM_MAX_BEAT_SEC = 4.0   # slowest plausible beat (kalam 1)

# Frequency band the onset envelope is restricted to before computing the
# beat-normalized ACF. Cymbals (ilathalam) sit above the chenda's
# fundamental and are the thalam's timekeeper — band-limiting to their
# range is what took this feature from ~57-64% to ~66-70% in the probe.
BEATNORM_CYMBAL_FMIN_HZ = 2000
BEATNORM_CYMBAL_FMAX_HZ = 9000

# --- Anti-memorization settings ---
# With only 8-19 unique source videos per class, a fast-fitting model can
# reach 95%+ train accuracy in 1-2 epochs by memorizing per-recording
# artifacts (mic/room/compression fingerprint) instead of general melam
# features — this is what happened in the first CRNN run (val_acc peaked at
# epoch 1, then degraded every epoch after).

# Max clips any single source video can contribute to TRAIN (see main() in
# 06_train_cnn.py). Prevents one long video (some contributed 1000+ clips)
# from dominating gradient updates over genuinely diverse recordings.
# Raised back up from 150 -> 250: at 150, Pandi (only 8 train videos, the
# fewest of any class) lost too large a share of its already-small clip
# count, and its test recall dropped to 0% in that run.
# Lowered 250 -> 60. Clips are now 30s rather than 7s, so 250 clips is
# 4.3x the audio it used to be. 60 x 30s = 30 minutes per video, which is
# roughly what 250 x 7s allowed before.
MAX_CLIPS_PER_VIDEO = 60

# Mixup: blends pairs of training spectrograms (and their labels) each
# batch. Turned OFF by default now — in testing it made Pandi recall go
# from 9% to 0%, most likely because linearly blending two DIFFERENT-tempo
# spectrograms doesn't produce a coherent "in-between" rhythm the way
# blending two images produces a coherent in-between image; it just
# produces two overlapping rhythms, actively destroying the tempo signal
# the new tempogram channel is meant to supply. Left here (off) in case
# it's worth revisiting for the mel channel specifically, but don't
# re-enable blindly.
USE_MIXUP = False
MIXUP_ALPHA = 0.2

# Gradient clipping — GRUs are prone to exploding gradients; without this
# the first CRNN run had a val_loss spike (1.7 -> 5.4) at epoch 5. Keep
# this on regardless of USE_MIXUP.
GRAD_CLIP_NORM = 5.0