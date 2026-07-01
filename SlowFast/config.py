
# SlowFast configuration for Cholec80 dataset
# Reference: "SlowFast Networks for Video Recognition" (Feichtenhofer et al., ICCV 2019)

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_ROOT_DIR = r"/mnt/sdc/cholec80"

# Path to the directory containing train_set_info.csv and test_set_info.csv
# Defaults to using the same CSVs prepared for EndoNet
CSV_DIR = r"../EndoNet"

img_w = 224
img_h = 224

# ── SlowFast temporal settings ────────────────────────────────────────────────
# alpha: ratio of fast frames to slow frames (T_fast = alpha * T_slow)
alpha = 4
# T_slow: number of frames sampled for the slow pathway
T_slow = 8
# T_fast: number of frames sampled for the fast pathway
T_fast = alpha * T_slow      # 32
# tau: sampling stride for the slow pathway (i.e., slow takes 1 frame every tau)
tau = 8
# Total clip window length in number of frames: T_slow * tau
clip_length = T_slow * tau   # 64

# Stride (in frames) between consecutive clips during dataset construction.
# Overlap between clips = clip_length - clip_stride.
clip_stride = 32

# ── Model architecture ────────────────────────────────────────────────────────
# beta_inv: channel ratio inverse (fast pathway uses slow_channels / beta_inv)
beta_inv = 8
# Slow pathway base channels (same as ResNet-50 stem)
slow_base_channels = 64
# Fast pathway base channels (slow_base_channels // beta_inv = 8)
fast_base_channels = slow_base_channels // beta_inv

num_phases = 7   # surgical phase classes
num_tools = 7    # tool classes (multi-label binary)
dropout = 0.5

# ── Dataset loading ───────────────────────────────────────────────────────────
dataset_params = {
    'batch_size': 4,
    'shuffle': True,
    'num_workers': 4,
    'pin_memory': True,
}

# ── Labels ────────────────────────────────────────────────────────────────────
phase_label_in_csv = "Phase"
phase_labels = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]
tool_labels = [
    "Grasper", "Bipolar", "Hook", "Scissors",
    "Clipper", "Irrigator", "SpecimenBag",
]

# ── Training ──────────────────────────────────────────────────────────────────
learning_rate = 1e-3
momentum = 0.9
weight_decay = 1e-4
epochs = 50
log_interval = 10   # print every N batches
save_interval = 5   # save checkpoint every N epochs

# Phase-loss / tool-loss weighting in the combined objective
phase_loss_weight = 1.0
tool_loss_weight = 1.0

# Threshold for converting tool sigmoid outputs to binary predictions
threshold = 0.5
