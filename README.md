# Cholec80 deep-learning pipelines

Cholec80 腹腔镜胆囊切除视频的四个任务，全部重写成一套共享代码：

| 任务 | 模型 | 入口 |
| --- | --- | --- |
| 7 类手术阶段 + 7 种器械多标签 | EndoNet（CNN + 双头），可选 BiLSTM 时序版 | `scripts/train_endonet.py` |
| 片段级阶段/器械识别 | SlowFast（3D 双路径，ImageNet 膨胀初始化） | `scripts/train_slowfast.py` |
| 主动学习（少标注选帧） | 7 种 query 策略，warm-start 跨轮训练 | `scripts/train_active.py` |
| 无配对域迁移 / 器械分割 | CycleGAN；GroundingDINO + SAM | `scripts/train_cyclegan.py`、`scripts/segment_instruments.py` |
| 仿真→真实纹理迁移（sim2real） | CycleGAN + 器械 mask 保护 + 光流时序一致性 | `scripts/train_sim2real.py`、`scripts/translate_video.py` |

`EndoNet/`、`SlowFast/`、`CycleGAN/`、`active_learning/` 四个旧目录**原样保留**，只作为对照参考（它们跑不起来，见文末清单）。真正能用的是 `cholec80/` 包 + `scripts/`，确认不需要旧代码后可以直接删掉那四个目录。

## 1. 环境（venv，服务器通用）

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# GPU 机器先按 CUDA 版本装 torch（示例 cu121，按需换成 cu118/cu124）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -e .            # 装本仓库（纯 Python，秒级）
pip install -e '.[dev]'     # 加 pytest
```

或者直接跑 `bash setup.sh`（见第 7 节）。CPU 也能跑（`--device cpu`、`--no-amp`），但训练没意义，只用于自检。

不依赖 timm / scipy / sklearn：AP、PCA（随机投影）、k-center 距离、混淆矩阵都是 numpy 手写，省掉一堆编译型依赖。

## 2. 数据准备

原始数据按官方布局放好即可，不需要手工做 CSV：

```
data/cholec80/
├── videos/video01.mp4 ... video80.mp4
├── phase_annotations/video01-phase.txt     # Frame, Phase（25 fps 每帧一行）
└── tool_annotations/video01-tool.txt       # Frame, 7 个器械列（1 fps）
```

```bash
python scripts/prepare_dataset.py \
    --root data/cholec80 --stride 25 --size 256 --crop square --workers 8
```

产出：

- `data/frames/<video>/<frame:06d>.jpg`：按 stride 抽帧（stride 25 = 1 fps），默认方形裁切+缩放到 256，JPEG q92；
- `data/index/{train,val,test}.csv`：列 `video,frame,path,phase,Grasper,...,SpecimenBag`，`path` 是**相对**路径（`video01/000000.jpg`），换机器不用改表；
- `data/index/split.json`：split 明细 + 每个 split 的帧数、各 phase/器械计数，用来一眼看出类别不平衡。

划分口径：`--protocol ordered`（默认）按视频序号切，前 40 个训练、后 40 个测试（EndoNet 论文口径），再从训练视频里切 `--val-ratio 0.1` 做验证集，保证 val/test 不重叠；`--protocol random` 用 `--seed` 随机切。

## 3. EndoNet：阶段 + 器械

```bash
# 帧级（默认 resnet50 backbone，ImageNet 预训练）
python scripts/train_endonet.py --epochs 30 --batch-size 64 --workers 8

# 时序版：连续 16 帧 → BiLSTM → 每个时间步都算 phase loss
python scripts/train_endonet.py --temporal lstm --seq-len 16 --seq-stride 4 \
    --rnn-hidden 256 --rnn-layers 2 --name endonet-lstm
```

关键点：

- 模型只输出 **logits**，`BCEWithLogitsLoss` + `CrossEntropyLoss`，器械 `pos_weight` 和 phase class weight 都从 **train CSV** 统计（不会泄漏 val/test）；
- phase 头输入是 `[features, sigmoid(tool_logits)]`，沿用 EndoNet 论文的 4103 维设计，维度由 backbone 自动推导；
- `--freeze-epochs 1` 先线性探测，冻结期 backbone 置 eval，BN 统计不被污染；
- CUDA 上默认 AMP，AdamW + backbone 单独 lr 系数，warmup-cosine，梯度裁剪，early stopping 按 `--monitor`；
- `--no-pretrained` 完全离线可跑（做 ablation 或没网时用）。

## 4. SlowFast：片段级

时序模型需要更密的采样。建议**另建一套 5 fps 的索引**（1 fps 只能看 1 秒一帧，慢路径几乎没运动信息）：

```bash
python scripts/prepare_dataset.py --root data/cholec80 --stride 5 --size 256 \
    --frames-dir data/frames_5fps --index-dir data/index_5fps

python scripts/train_slowfast.py --index-dir data/index_5fps --frames-dir data/frames_5fps \
    --clip-len 32 --tau 8 --alpha 4 --batch-size 8 --epochs 20
```

- 通道算术与论文一致（slow `64/256/512/1024/2048`，fast 为 1/8 宽度，按 `beta_inv` 调）；
- 默认用 ImageNet ResNet-50 的 2D 卷积核沿时间维膨胀初始化 slow 和 fast 两条路径（`--no-inflate` 关闭；无网时加 `--no-inflate`）；
- 每个 clip 的 JPEG 只解码一次，slow 视图从 fast 帧里按索引取子集；
- clip 太长而视频不够时会把 `clip_len` 自动缩到最长视频的整数倍并 warning（小数据集/自检不会直接报 0 样本）。

## 5. 主动学习

```bash
python scripts/train_active.py --strategy entropy --n-rounds 8 --query-budget 200 \
    --initial-ratio 0.05 --epochs-per-round 5 --batch-size 32
```

可用策略：`random`、`least_confidence`、`margin`、`entropy`、`coreset`、`badge`、`mc_dropout`（BALD）。

- 每轮只在**未标注池**上跑一次前向，缓存 probs/features 供策略使用，不重复推理；
- 默认 **warm start**（跨轮复用权重），`--fresh-model` 才是每轮冷启动（复现旧实现的口径）；
- 选 best 看**验证集**，不看训练集；
- 初始标注集 `balance_initial` 保证每个 phase 至少一帧，避免小池子某类永远学不到；
- BADGE 的梯度嵌入先用随机投影降到 64 维再 k-means++，这是**文档化的近似**，不是原论文精确式（原式 7×4103 维内存不可行）；
- coreset 在 query 时只用 labeled 集特征，分块算距离，不 O(N·L) 爆内存。

结果：`runs/al-<strategy>-<tag>_<时间戳>/curve.csv`（每轮标注量、query 耗时、val 指标）+ `final.pt`。

## 6. 评估、推理、CycleGAN、分割

```bash
# test split 全量指标 + 逐视频
python scripts/evaluate.py --checkpoint runs/endonet_20260101-120000/checkpoints/best.pt --split test

# 直接吃原始视频（滑窗抽帧，输出每帧 phase 概率和 7 个器械概率）
python scripts/evaluate.py --checkpoint runs/endonet_.../checkpoints/best.pt \
    --video data/cholec80/videos/video71.mp4
```

`evaluate.py` 从 checkpoint 的 `model_config` 重建模型（`pretrained=False`，不联网），自动识别 frame/temporal/SlowFast 三种输出。

CycleGAN（无配对域迁移，例如把 A 中心的画质迁移到 B 中心）：

```bash
python scripts/train_cyclegan.py --a-dir data/frames/video01 --b-dir data/frames/video02 \
    --epochs 100 --decay-epoch 50 --size 256 --batch-size 4 --lambda-cycle 10 --lambda-identity 5
```

去掉 Visdom，改成 `samples/epoch_XXX.jpg` 网格 + `metrics.csv`，无头服务器可用；`--resume` 续训。

### 6.1 仿真→真实纹理迁移（sim2real）

给仿真器官加"血肉感"：A 域 = 仿真帧，B 域 = 真实帧（Cholec80 抽好的帧直接可用）。相对 vanilla CycleGAN 多两个手术专用保护：

- **器械/语义 mask 保护**：仿真器导出的器械 mask（白色=保护）在每次生成后把原像素贴回，生成器在保护区零梯度，语义标签不被翻译污染；
- **光流时序一致性**：相邻两帧的翻译结果经 Farneback 光流对齐后做 L1 约束（带 forward-backward 遮挡检测），抑制逐帧闪烁。

```bash
# 训练（mask 目录镜像 a-dir 的相对路径，.png，白=保留原像素）
python scripts/train_sim2real.py \
    --a-dir data/sim/frames --b-dir data/frames \
    --a-mask-dir data/sim/instrument_masks \
    --temporal-weight 2.0 --epochs 100 --size 256 --batch-size 4

# 推理：图片目录或原始视频，可选 mask 贴回 + 光流引导的时域平滑
python scripts/translate_video.py \
    --checkpoint runs/sim2real_<ts>/checkpoints/last.pt \
    --input data/sim/frames/video01 --out-dir output/translated \
    --mask-dir data/sim/instrument_masks/video01 \
    --temporal-smooth 0.5 --side-by-side --write-video
```

学习入口：`cholec80/sim2real.py`（每个函数都有原理级英文注释：为什么 LSGAN、cycle 为什么防幻觉、backward warping 的坐标约定、forward-backward 遮挡检测）、`scripts/train_sim2real.py` 顶部有四个 loss 的完整推导说明。单测 `tests/test_sim2real.py` 用零光流=恒等、纯平移光流、mask 对齐等精确断言验证这些数学性质。

器械分割（可选，需额外依赖）：

```bash
pip install -e '.[seg]'
python scripts/segment_instruments.py --input data/cholec80/videos --out-dir output/masks \
    --stride 25 --prompt "surgical instrument . grasper . hook . scissors . clipper"
```

输出 `mask/`、`foreground/`、可选 `background/` 和 `meta.jsonl`（每帧 prompt、box、score），替代旧 `CycleGAN/data_prepared.py` 那段写死路径的 DINO+SAM 脚本。

## 7. 在服务器上跑：自检顺序

```bash
bash setup.sh                       # 建 venv + 装依赖（GPU 机器设 TORCH_INDEX_URL）

pytest                              # 40 个单测：标注解析、三种 Dataset、模型前向、指标、AL 策略、ckpt
python scripts/smoke_test.py        # 合成 6 个视频，串跑 5 个 CLI，约 2-3 分钟，CPU 即可
python scripts/smoke_test.py --skip-slowfast
```

`smoke_test` 会依次跑 `prepare_dataset → train_endonet → evaluate → train_active → train_slowfast`，任何一个 CLI 参数/接口对不上都会立刻暴露，跑通再去下真数据。

真数据上的建议参数（单卡 24 GB 量级）：

| | batch | workers | 说明 |
| --- | --- | --- | --- |
| EndoNet frame | 64 | 8–16 | 224px，AMP，~1.5 h/epoch 取决于磁盘 |
| EndoNet temporal | 16 (×16 帧) | 8 | 显存约 4×帧级；`--seq-len 16` 起 |
| SlowFast | 8–16 | 8 | 32×224² clip；显存 ~12 GB @ 8 |
| CycleGAN | 4–8 | 4 | 256px，9 blocks |

其他服务器注意事项：

- 目前是**单卡**实现（`nn.DataParallel` 都不用），多卡请自行扩展，代码里没有 DDP 分支；
- 抽帧是 CPU/IO 密集，`--workers` 别超过 CPU 核数的一半，否则 JPEG 解码互相抢；
- 中断续训：所有训练脚本支持 `--resume runs/<name>_<ts>/checkpoints/last.pt`；
- 每个 run 目录都有 `args.json`（完整参数）+ `metrics.csv`（逐 epoch）+ `checkpoints/{last,best}.pt` + `test_metrics.json`，comparison 不用再靠记忆；
- 不做 deterministic 强制，需要严格复现实验自己加 `torch.use_deterministic_algorithms`。

## 8. 旧代码的问题（已在新实现里修掉）

`EndoNet/`

- `config.py` 里根本没有 `phase_label_in_csv`、`threshold`、`cnn_encoder_params`、`rnn_decoder_params`，`train.py`/`eval.py` 引用即 AttributeError；`eval.py` 还 import 不存在的 `CNNEncoder`；
- checkpoint 保存时 `model2_state_dict` 存的是 model1 的权重，`optimizer2_state_dict` 存的是 optimizer1，恢复后两个头互相污染；
- 训练 `shuffle=False`；验证对 model1 前向两次；训练用 `MultiLabelSoftMarginLoss`、验证用 `BCELoss`，指标口径不一致；criterion 在 batch 循环里每步新建；
- `model.py` 的 AlexNet 在 forward 里就 `sigmoid`，再喂 `MultiLabelSoftMarginLoss`（内部又 sigmoid），损失基本失效；`EasyFCNet` 硬编码 4103；
- `dataloader.py` 每次 `__getitem__` 重建 `transforms.Compose`，硬编码裁 `(50,0,804,480)`，`use_mem` 死代码；
- `utils.py` 用 `DataFrame.append`（pandas 2+ 已删）、`plotting` 里 `plt` 未导入；
- `data/make_train_test.py`：`default_src_dir` 未定义、`size=0.5` 覆盖命令行、`get_phase_label` 每帧扫全表（O(n²)）、建目录逻辑反了、末尾 `input()` 阻塞；
- `data/`、`test/` 两个子目录是从手势识别项目抄来的残稿（`gesture_classes`、`super_trail_count` 在 Cholec80 里不存在），已整体丢弃。

`SlowFast/`

- 没有训练脚本，只有 model + dataloader；
- 用 `os.path.dirname(file_loc)` 分组，同一 split 所有视频被当成一段，clip 跨视频；
- 按字符串排序，`video01_1000.jpg` 排在 `video01_100.jpg` 前面，时序错乱；
- `int(center["Phase"])` 取的列在 EndoNet CSV 里不存在（那里是 `label` 或 7 个 one-hot 列）；
- 每样本解码 40 张 JPEG 且 slow/fast 各解一遍。

`CycleGAN/`

- `Variable`、Visdom 依赖（无头服务器直接挂）；`--epoch` 其实是起始 epoch；`output/` 不自动创建；无 AMP、无 resume；
- `data_prepared.py`（592 行）把 dataset 路径写死 `/mnt/sdc/...`，`batch_size` 收了不用，`save_segmentation_results` 把 scores/text_labels 丢掉，无 CLI。

`active_learning/`

- 依赖上面不存在的 config 键；每轮重建模型（丢 warm start）；按**训练集**准确率选 best；coreset 在 query 时用 `shuffle=True` 的 labeled loader；BADGE 原始嵌入 7×4103=28721 维，内存不可行。

## 9. 目录

```
cholec80/                 # 唯一可运行实现
├── constants.py          # 7 phase / 7 tool 顺序、fps 约定
├── annotations.py        # TSV 解析、抽帧、索引、split
├── data/                 # FrameDataset / SequenceDataset / ClipDataset / CycleGAN dataset
├── models/               # endonet / slowfast / cyclegan / backbone 包装
├── active_learning/      # 7 种策略 + 标注池
├── engine.py             # AMP、scheduler、CheckpointManager、resume
├── metrics.py            # 混淆矩阵、手写 AP/mAP、视频级 Jaccard
├── evaluation.py inference.py losses.py utils.py
scripts/                  # 全部 CLI，统一 args.json / metrics.csv 产物
tests/                    # pytest
EndoNet/ SlowFast/ CycleGAN/ active_learning/   # legacy，仅对照
```

## 10. 已知限制

- 无 DDP/多卡，无 TensorBoard 依赖（要曲线自己读 `metrics.csv`）；
- BADGE 是近似实现；coreset 是贪心 k-center，非精确解；
- `evaluate.py` 对 SlowFast 只给聚合指标，逐视频/原始视频推理目前只支持帧级模型；
- `segment_instruments.py` 需要 `transformers` 且要能下 HuggingFace 权重，跑不通先检查网络和 `.[seg]` 是否装上；
- 旧目录不参与打包（`pyproject.toml` 只收 `cholec80*`）。
