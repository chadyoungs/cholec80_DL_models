# Cholec80 deep-learning pipelines

Modern, runnable PyTorch code for the Cholec80 laparoscopic cholecystectomy
dataset. One shared package (`cholec80_dl/`), one CLI per task, no duplicated
implementations.

| Task | Model | Entry point |
| --- | --- | --- |
| Phase (7-way) + tool (7-way multi-label) recognition, frame level | EndoNet: CNN backbone + dual heads | `scripts/train_endonet.py` |
| Phase recognition with temporal context | EndoNet + BiLSTM over frame sequences | `scripts/train_endonet.py --temporal lstm` |
| Clip-level phase / tool recognition | SlowFast (3-D two-pathway, ImageNet inflation) | `scripts/train_slowfast.py` |
| Label-efficient training (active learning) | 7 query strategies, warm-start rounds | `scripts/train_active.py` |
| Sim-to-real texture transfer (also plain unpaired domain transfer) | CycleGAN + instrument-mask protection + flow temporal consistency | `scripts/train_sim2real.py`, `scripts/translate_video.py` |
| Instrument detection / segmentation | GroundingDINO + SAM | `scripts/segment_instruments.py` |

Supporting CLIs: `prepare_dataset.py` (frames + index CSVs), `evaluate.py`
(metrics + raw-video inference), `smoke_test.py` (synthetic end-to-end check).

The original exploratory folders (`EndoNet/`, `SlowFast/`, `CycleGAN/`,
`active_learning/`) were removed after being fully audited and rewritten;
they remain recoverable from git history (commit `034f0ea` and earlier).
A summary of the bugs found in them is kept at the bottom of this README.

## 1. Environment (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# On a GPU server, install torch for your CUDA version first (example: cu121)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -e '.[dev]'     # this repo + pytest
```

Or simply `bash setup.sh` (set `TORCH_INDEX_URL` for GPU wheels).
CPU works for the self-checks (`--device cpu --no-amp`) but not for real training.

No timm / scipy / sklearn dependencies: average precision, random-projection
PCA, k-center distances and confusion matrices are implemented in plain numpy.

## 2. Dataset preparation

Place the official Cholec80 release as-is; no manual CSV building:

```
data/cholec80/
├── videos/video01.mp4 ... video80.mp4
├── phase_annotations/video01-phase.txt     # Frame, Phase (one row per frame, 25 fps)
└── tool_annotations/video01-tool.txt       # Frame, 7 tool columns (1 fps)
```

```bash
python scripts/prepare_dataset.py \
    --root data/cholec80 --stride 25 --size 256 --crop square --workers 8
```

Outputs:

- `data/frames/<video>/<frame:06d>.jpg` — extracted frames (stride 25 = 1 fps),
  square-cropped and resized to 256 by default, JPEG q92;
- `data/index/{train,val,test}.csv` — columns
  `video,frame,path,phase,Grasper,...,SpecimenBag`; `path` is **relative**
  (`video01/000000.jpg`), so the index is machine-independent;
- `data/index/split.json` — split membership plus per-split frame counts and
  phase/tool histograms (instant class-imbalance overview).

Split protocol: `--protocol ordered` (default) splits by video number — first
40 videos for training, last 40 for testing (the EndoNet paper protocol) — then
carves `--val-ratio 0.1` of the training videos into a validation set, so val
and test never overlap. `--protocol random` shuffles with `--seed`.

## 3. Phase + tool recognition (EndoNet)

```bash
# Frame level (resnet50 backbone, ImageNet pretrained by default)
python scripts/train_endonet.py --epochs 30 --batch-size 64 --workers 8

# Temporal variant: 16 consecutive frames -> BiLSTM -> phase loss at every step
python scripts/train_endonet.py --temporal lstm --seq-len 16 --seq-stride 4 \
    --rnn-hidden 256 --rnn-layers 2 --name endonet-lstm
```

Design notes:

- Models emit **logits only**; losses are `BCEWithLogitsLoss` (tools) +
  `CrossEntropyLoss` (phases). Tool `pos_weight` and phase class weights are
  estimated from the **train CSV only** (no val/test leakage).
- The phase head consumes `[features, sigmoid(tool_logits)]`, following the
  EndoNet paper's 4103-d design; the width is derived from the backbone.
- `--freeze-epochs 1` runs a linear-probe warm-up; the frozen backbone is kept
  in eval mode so its BatchNorm statistics are not polluted.
- AMP on CUDA, AdamW with a separate backbone LR multiplier, warmup-cosine
  schedule, gradient clipping, early stopping on `--monitor`.
- `--no-pretrained` runs fully offline (ablations / no network).

## 4. Clip-level recognition (SlowFast)

Temporal models need denser sampling. Build a second, 5-fps index (at 1 fps
the slow pathway sees almost no motion):

```bash
python scripts/prepare_dataset.py --root data/cholec80 --stride 5 --size 256 \
    --frames-dir data/frames_5fps --index-dir data/index_5fps

python scripts/train_slowfast.py --index-dir data/index_5fps --frames-dir data/frames_5fps \
    --clip-len 32 --tau 8 --alpha 4 --batch-size 8 --epochs 20
```

- Channel schedule follows the paper (slow `64/256/512/1024/2048`, fast at
  1/`beta_inv` width).
- By default both pathways are initialised by inflating ImageNet ResNet-50
  2-D kernels along time (`--no-inflate` to disable; required when offline).
- Each JPEG in a clip is decoded exactly once; the slow view is an index
  subset of the fast frames.
- If clips are longer than the videos, `clip_len` shrinks to a multiple of
  the fast step that fits the longest video (with a warning) instead of
  producing an empty dataset.

## 5. Active learning

```bash
python scripts/train_active.py --strategy entropy --n-rounds 8 --query-budget 200 \
    --initial-ratio 0.05 --epochs-per-round 5 --batch-size 32
```

Strategies: `random`, `least_confidence`, `margin`, `entropy`, `coreset`,
`badge`, `mc_dropout` (BALD).

- One cached forward pass over the **unlabeled pool** per round; strategies
  operate on the cached probs/features (no repeated inference).
- **Warm start** across rounds by default; `--fresh-model` re-initialises
  every round (cold-start ablation).
- Best-model selection uses the **validation** set, never the training set.
- The initial labeled set is phase-balanced (at least one frame per phase).
- BADGE projects gradient embeddings to 64 dims with a random projection
  before k-means++ — a documented approximation (the exact 7x4103-d
  embedding is memory-infeasible).
- Core-set greedy k-center uses chunked numpy distances against labeled
  features only.

Output: `runs/al-<strategy>-<tag>_<ts>/curve.csv` (labels per round, query
time, val metrics) + `final.pt`.

## 6. Sim-to-real texture transfer

Give simulated organs a real-tissue look. Domain A = simulated frames,
domain B = real frames (Cholec80 frames from `prepare_dataset.py` work
directly). Two surgery-specific guards on top of CycleGAN:

- **Instrument / semantics protection**: simulator-exported masks
  (white = protect) composite the original pixels back over every
  translation. The generator gets zero gradient on protected pixels, so
  semantic labels stay valid.
- **Flow-based temporal consistency**: translations of consecutive frames
  are aligned with Farneback optical flow (with forward-backward occlusion
  checking) and matched with an L1 loss — suppresses per-frame flicker.

```bash
# Train (mask dir mirrors a-dir layout, .png, white = keep original pixels)
python scripts/train_sim2real.py \
    --a-dir data/sim/frames --b-dir data/frames \
    --a-mask-dir data/sim/instrument_masks \
    --temporal-weight 2.0 --epochs 100 --size 256 --batch-size 4

# Translate an image folder or a raw video
python scripts/translate_video.py \
    --checkpoint runs/sim2real_<ts>/checkpoints/last.pt \
    --input data/sim/videos/run01.mp4 --out-dir output/translated \
    --temporal-smooth 0.5 --side-by-side --write-video
```

Omit `--a-mask-dir` and `--temporal-weight` and this is exactly a plain
CycleGAN trainer — it is the single image-translation entry point in this
repo (there is no separate `train_cyclegan.py`).

Learning path: read the loss derivations at the top of
`scripts/train_sim2real.py`, then `cholec80_dl/sim2real.py` (principle-level
comments on LSGAN, cycle consistency, backward warping conventions and
forward-backward occlusion checks), then the dataset and inference script.
`tests/test_sim2real.py` verifies the math with exact assertions
(zero-flow warp = identity, pure-translation flow, mask alignment).

## 7. Evaluation and inference

```bash
# Full test-split metrics + per-video accuracy / Jaccard
python scripts/evaluate.py --checkpoint runs/endonet_<ts>/checkpoints/best.pt --split test

# Raw-video inference: per-frame phase probabilities + 7 tool probabilities
python scripts/evaluate.py --checkpoint runs/endonet_<ts>/checkpoints/best.pt \
    --video data/cholec80/videos/video71.mp4
```

`evaluate.py` rebuilds the model from the checkpoint's `model_config`
(`pretrained=False`, no downloads) and handles frame / temporal / SlowFast
outputs automatically.

## 8. Instrument segmentation (optional)

```bash
pip install -e '.[seg]'
python scripts/segment_instruments.py --input data/cholec80/videos --out-dir output/masks \
    --stride 25 --prompt "surgical instrument . grasper . hook . scissors . clipper"
```

Outputs `mask/`, `foreground/`, optional `background/`, and `meta.jsonl`
(per-frame prompts, boxes, scores). Requires HuggingFace weight downloads.

## 9. Server runbook

```bash
bash setup.sh                       # venv + deps (set TORCH_INDEX_URL on GPU)
pytest                              # 50 unit tests
python scripts/smoke_test.py        # synthetic 6-video end-to-end check (~2 min, CPU)
```

The smoke test chains `prepare_dataset -> train_endonet -> evaluate ->
train_active -> train_slowfast` on synthetic data, so any CLI/interface
breakage surfaces before you download the real dataset.

Suggested settings (single 24 GB GPU):

| Run | batch | workers | Notes |
| --- | --- | --- | --- |
| EndoNet frame | 64 | 8–16 | 224 px, AMP |
| EndoNet temporal | 16 (x16 frames) | 8 | ~4x frame-level memory |
| SlowFast | 8–16 | 8 | 32x224² clips, ~12 GB @ 8 |
| Sim2real / CycleGAN | 4–8 | 4 | 256 px, 9 residual blocks |

Operational notes:

- Single-GPU implementation (no DDP branch, no DataParallel).
- Frame extraction is CPU/IO bound; keep `--workers` below half the core count.
- Every training script supports `--resume runs/<name>_<ts>/checkpoints/last.pt`.
- Every run directory contains `args.json`, `metrics.csv`,
  `checkpoints/{last,best}.pt` and `test_metrics.json` — experiments are
  comparable without guesswork.
- Deterministic algorithms are not forced; add
  `torch.use_deterministic_algorithms` yourself if you need bit-exact reruns.

## 10. Repository layout

```
cholec80_dl/              # the single runnable implementation
├── constants.py          # 7 phases / 7 tools ordering, fps conventions
├── annotations.py        # TSV parsing, frame extraction, index, splits
├── data/                 # FrameDataset / SequenceDataset / ClipDataset / Sim2RealDataset
├── models/               # endonet / slowfast / cyclegan / backbone wrapper
├── active_learning/      # 7 strategies + labeled pool
├── sim2real.py           # mask blending, optical-flow warping, occlusion checks
├── engine.py             # AMP, schedulers, CheckpointManager, resume
├── metrics.py            # confusion matrix, hand-written AP/mAP, video Jaccard
├── evaluation.py  inference.py  losses.py  utils.py
scripts/                  # one CLI per task, uniform run artifacts
tests/                    # pytest (50 cases)
licenses/                 # third-party licenses (CycleGAN reference: GPLv3)
```

## 11. Known limitations

- No DDP / multi-GPU; no TensorBoard dependency (plot `metrics.csv` yourself).
- BADGE is an approximation; core-set is greedy k-center, not exact.
- `evaluate.py` gives aggregate metrics for SlowFast; per-video / raw-video
  inference currently supports frame-level models only.
- `segment_instruments.py` needs `transformers` and network access to
  HuggingFace weights.
- The CycleGAN generator/discriminator derive from the GPLv3
  PyTorch-CycleGAN reference (license kept in `licenses/`); keep that in
  mind when redistributing.

## Appendix: bugs found in the original code (all fixed in this rewrite)

`EndoNet/`

- Referenced config keys that never existed (`phase_label_in_csv`,
  `threshold`, `cnn_encoder_params`, `rnn_decoder_params`) → AttributeError;
  `eval.py` imported a non-existent `CNNEncoder`.
- Checkpointing saved model1's weights under `model2_state_dict` and
  optimizer1 under `optimizer2_state_dict` — restoring corrupted both heads.
- Training loader had `shuffle=False`; validation ran model1 twice; training
  used `MultiLabelSoftMarginLoss` while validation used `BCELoss`; the
  criterion was re-created inside the batch loop.
- AlexNet applied `sigmoid` in `forward` and then fed a sigmoid-internal
  loss (double sigmoid); `EasyFCNet` hard-coded 4103.
- The dataloader rebuilt `transforms.Compose` on every `__getitem__`,
  hard-coded a `(50,0,804,480)` crop, and carried dead code.
- `utils.py` used `DataFrame.append` (removed in pandas 2+); `plotting`
  referenced an unimported `plt`.
- `data/make_train_test.py`: undefined `default_src_dir`, `size=0.5`
  overriding the CLI, O(n²) per-frame label lookup, inverted directory
  creation logic, and a blocking `input()`.
- `data/` and `test/` subfolders were leftovers from a hand-gesture project
  (`gesture_classes`, `super_trail_count` — meaningless for Cholec80).

`SlowFast/`

- No training script at all.
- Grouped clips by `os.path.dirname(file_loc)` → all videos in a split were
  treated as one sequence; clips crossed video boundaries.
- Lexical frame sorting put `video01_1000.jpg` before `video01_100.jpg`.
- Read a `Phase` column that does not exist in the EndoNet CSVs.
- Decoded 40 JPEGs per sample, twice (once per pathway).

`CycleGAN/`

- `Variable`, Visdom dependency (breaks headless servers); `--epoch` was
  actually the start epoch; `output/` was never created; no AMP, no resume.
- `data_prepared.py` (592 lines) hard-coded `/mnt/sdc/...` paths, ignored its
  `batch_size` argument, dropped scores/text labels, and had no CLI.

`active_learning/`

- Depended on the non-existent config keys above; rebuilt the model every
  round (losing warm start); selected the best model by **training**
  accuracy; core-set queried a `shuffle=True` labeled loader; BADGE used the
  full 7x4103 = 28721-d embedding (memory-infeasible).
