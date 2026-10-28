
# ESAP-SAM：电网设施／背景二分类

根据提供的 `frontiers.pdf` 第 2 节、图 2–6 和第 3.1 节实现。默认采用 **SAM ViT-L**，保留 ViT-B；自动分割只需 RGB 图像，也支持可选点、框和掩码提示。

这是依据论文定义、结构图和已确认的补全方案实现的版本。论文没有给出若干模块的唯一实现，本文逐项记录补全选择；不声称它是作者原始源码，也不声称已经复现论文中的精度或运行时间。

## 1. 文件和依赖

```text
ESAP_SAM/
├── esap_sam.py          # SAM 组件、EMS、HP、特征解码器、BRA，以及模型自检命令
├── train_esap_sam.py    # 数据读取、训练、验证、独立测试、保存和断点恢复
├── README.md
└── requirements.txt
```

两个 Python 文件可独立组成工程，不依赖参考训练脚本所在工程。SAM 所需的 PyTorch 实现已经合入模型文件，不需要额外安装 `segment-anything`、`torchvision`、`timm`、OpenCV 或 Albumentations。运行时不自动联网下载权重。

### 目标环境与精确版本

| 项目 | 版本 | 用途 |
|---|---|---|
| Python | 3.13.12，conda-forge，GCC 14.3.0 | 用户截图中的 Linux 解释器；不是 pip 包 |
| torch | 2.10.0+cu129 | 模型、自动微分、优化器、数据加载 |
| CUDA runtime | 12.9 | 截图中的 Torch CUDA 构建运行时 |
| cuDNN | 91002（9.10.2） | 截图中的卷积运行时 |
| triton | 3.6.0 | 保留目标环境版本；本工程不编写 Triton 内核，不要求 `torch.compile` |
| numpy | 2.2.6 | 数据与标签数组 |
| Pillow | 11.1.0 | 图像和掩码读取、可选重采样 |
| tqdm | 4.67.1 | 进度条 |
| colorama | 0.4.6，仅 Windows | tqdm 在 Windows 上的条件依赖；目标 Linux 无需安装 |

`requirements.txt` 精确列出工程的所有第三方 Python 依赖及 Windows 条件依赖。Python 标准库无需安装。Torch 自身的传递依赖、GPU 驱动及 CUDA/cuDNN 二进制组件由截图中的现有环境提供；仅凭截图无法恢复它们的完整包锁定清单，因此这里不臆造这些包的版本。可在目标环境运行 `python -m pip freeze` 留存完整清单。

**安装时保留截图中的现有 Torch 构建。** [PyTorch 官方 2.10.0 安装列表](https://pytorch.org/get-started/previous-versions/#v2100)列出了 cu126/cu128/cu130，没有列出 cu129；因此本项目没有虚构一个 `2.10.0+cu129` 下载源。若该环境尚未配置，应先从截图环境的原始提供方取得对应构建，然后安装本工程依赖。

在已有截图环境内执行：

```bash
cd /path/to/ESAP_SAM
# --no-deps 保留现有 Torch 及其传递依赖；已满足精确版本的包不会重新安装。
python -m pip install --no-deps -r requirements.txt
python train_esap_sam.py --check-env --strict-env
```

若 Torch 或 Triton 不在现有环境里，以上安装可能因找不到该特定构建而失败，需要原环境的包来源；不要通过去掉 `+cu129` 来悄悄替换目标构建。没有完整现有环境时也不要使用 `--no-deps` 安装一个不完整的 Torch。

`--check-env` 打印解释器、Torch、CUDA runtime、cuDNN、Triton、GPU 和工程依赖版本；`--strict-env` 对截图中的五个版本项逐项校验。普通训练允许兼容环境运行，同时把版本差异写入 `environment.json`。CPU 可用于小尺寸检查。截图中的 CUDA 版本不是要求本机另行安装同版本 CUDA Toolkit；本工程没有需要本地编译的 CUDA 扩展。

依赖版本来源：[NumPy 2.2.6](https://pypi.org/project/numpy/2.2.6/)、[Pillow 11.1.0](https://pypi.org/project/pillow/11.1.0/)、[tqdm 4.67.1](https://pypi.org/project/tqdm/4.67.1/)。这些版本提供 Python 3.13 支持。

## 2. 论文定义与实现补充

### 模型信息流

```text
RGB [B,3,H,W]（0~1）
  → SAM 标准化，右侧/底部补齐到 16 的倍数
  → 冻结 Patch Embedding（stride=16）
      ├─ 冻结 ViT blocks + 位置编码 → F_ViT
      └─ 可训练 EMS Adapter → ΔF_EMS
  → F_ViT + ΔF_EMS（在 Ce 通道空间相加）
  → 冻结 SAM neck：Ce → 256
      ├─ 图像特征 ──────────────────────────┐
      └─ HP：dense + 可选 sparse/mask → prompts │
  → SAM 双向注意力 + 4×上采样 ←───────────────┘
  → 32 通道 decoded features → BRA → 1 通道 logits
  → 双线性上采样并移除 padding → [B,1,H,W]
```

| 项目 | 论文明确给出的定义 | 本实现的补充或处理 |
|---|---|---|
| SAM 主干 | Patch stride=16，Ce=1024，冻结 ViT | 默认 ViT-L，24 blocks / 16 heads；ViT-B 为可选对照，Ce=768、12 blocks / 12 heads。冻结整个 image encoder，包括 patch embedding、位置编码和 neck |
| EMS 并联 | Patch features 同时进入 ViT 与 EMS，在输出相加 | 在 neck 之前以 Ce 通道逐元素相加，随后用原生冻结 neck 映射至 256 通道，保证 SAM decoder 维度兼容 |
| EMS 压缩 | Cr=32，不进一步降低空间分辨率 | `1×1` down/up projection，均 stride=1；`--adapter-channels` 可调 |
| MSCM | `1×1 + 3×3`，GAP→MLP→sigmoid，再逐元素加权 | 通道注意力 MLP 使用 Cr→max(Cr/4,1)→Cr，中间 ReLU；3×3 padding=1 |
| MDCM | 水平、垂直、45°、135°，每路 BN+ReLU 后相加 | 默认核长 7，可用 `--directional-kernel` 改为其他不小于 3 的奇数；水平 1×k、垂直 k×1；对角为 k 个可学习抽头，仅在对角线上有支撑，并非普通稠密 k×k 卷积；各方向均保持网格尺寸 |
| EMS aggregation | 两分支相加、聚合、恢复通道 | aggregation=`1×1 Conv + BN + ReLU`，再 `1×1` 恢复 Ce |
| HP dense | `LN(Conv1×1(Fenc))+Epos` | 使用 neck 后的 256 通道 embedding，LN 对每个位置的通道进行归一化；Epos 采用冻结 SAM 随机傅里叶位置编码 |
| HP sparse | 可选点/框、冻结几何提示编码、拼接后两层 FFN | 复用冻结 SAM 坐标编码和点/框类型嵌入。正/负 sparse tokens 与 dense 特征的归一化点积分别平均，生成两个网格通道；框的两个角计入正提示。得到图 5 所示 C+2 通道；无提示时两通道置零。原 sparse tokens 同时进入 decoder 的 token stream |
| HP FFN | 两层 Linear，中间 GELU，输出 C | `(256+2)→512→256`，`--hp-hidden` 可调；在每个网格位置共享权重 |
| HP mask | 可选 mask guidance | 掩码先随图像 pad，再缩放到 4 倍 encoder grid；冻结 SAM mask downscaling 编码后加到 HP dense 特征中。自动路径不需要它 |
| Mask decoder | SAM 风格的双向 attention、输出 decoded features | 从预训练 SAM 复用两层双向注意力、IoU/mask query tokens、4×上采样层；保留其 state 对应权重。原生多掩码超网络、IoU 预测头，以及只更新最终 token 的最后一次 attention 不用于 BRA 特征输出，加载权重后不保留这些模块 |
| BRA | 初始 1×1 预测、Sobel 幅值、3×3+sigmoid、**加法**残差、融合及预测头 | 初始预测使用 raw logits；Sobel 固定、无参数，边界采用 replicate padding；幅值使用 `sqrt(gx²+gy²+1e-6)` 保证平坦区域反向传播有限；随后使用论文的加法残差和 Conv+BN+ReLU |
| BRA prediction head | 3×3 Conv+BN+ReLU，再 1×1 输出 | 中间宽度 32，输出一个设施前景通道；最后不带 sigmoid，训练 Dice 内部只做一次 sigmoid |
| 输入大小 | TTPLA 640×640 chips；PLD 540×360 图像 | 默认保留已准备图像尺寸，不强制放大到 1024；插值绝对位置编码，原生 SAM relative position 代码动态插值；底部/右侧 pad 到 patch 倍数，输出恢复输入尺寸 |

**梯度边界：**冻结 patch/ViT 的前向在 `no_grad` 下进行。EMS 直接处理冻结 patch 特征，并行输出可训练增量。冻结 neck 的参数不更新，但它的前向保留对输入的梯度，使 EMS 能收到梯度；不能把整个融合和 neck 都包在 `no_grad` 中。冻结几何编码路径不更新参数。EMS、HP、解码器实际使用的组件和 BRA 参与训练。

论文 Ce=1024 与其约 95.52M 的总参数表存在冲突；已按确认采用 ViT-L，程序会报告实际参数量，不人为凑齐论文的数值。当前实现的模型规模见第 9 节。

## 3. 预训练权重

使用官方 **SAM v1** 的匹配权重，不使用 SAM2/HQ-SAM 权重：

- 默认 ViT-L：[sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- 可选 ViT-B：[sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- 权重说明：[SAM 官方仓库](https://github.com/facebookresearch/segment-anything#model-checkpoints)

通过 `--sam-checkpoint /path/to/sam_vit_l_0b3195.pth` 传入。先以原生 SAM 的参数名和维度严格加载，再取本实现需要的模块；型号或文件不匹配会报错，不会用 `strict=False` 静默丢弃主干权重。不会下载或附送约 GB 级的权重文件。

`--allow-random-init` 仅用于无权重时检查流程。它会冻结随机主干，无法作为预训练模型复现实验；训练会显示警告，并将该状态写入 checkpoint。

## 4. 数据组织和标签

推荐使用预先划分好的数据：

```text
dataset/
├── train/
│   ├── images/scene_001.png
│   └── masks/scene_001.png
├── val/
│   ├── images/scene_002.jpg
│   └── masks/scene_002.png
└── test/                       # 可选；独立测试时需要
    ├── images/scene_003.png
    └── masks/scene_003.png
```

支持子目录；图像与掩码按各自目录中的**相对路径去扩展名**严格配对。例如 `images/a/001.jpg` 匹配 `masks/a/001.png`，不会只按排序索引配对。图像支持 PNG/JPEG/TIFF/BMP；mask 使用无损 PNG/TIFF/BMP。同一相对 stem 出现多个扩展名、缺失配对、原图和 mask 尺寸不一致均会报错。

也支持 `dataset/images/` 和 `dataset/masks/`：按 seed=42 随机划分 7:2:1，划分文件保存在 `splits.json`。小样本取整会使比例近似，并确保三个集合非空。**已经切成相邻 patches 的数据应先按原始场景划分，再放入 train/val/test，避免同一原图进入不同集合。**脚本不能从任意 patch 文件名自动恢复其源场景身份。

标签约定：

- 通道数固定为 1：背景=0；杆塔、电线及任务中指定的其他电网设施统一作为前景。
- 默认接受 0/1 或 0/255 二值 mask；不会将 255 自动当作 ignore。
- 若已标注多个设施 ID，例如 1、2、3，添加 `--foreground-values 1 2 3` 合并为前景。除 0、声明的前景 ID 以及可选 ignore ID 外，其他 ID 会报错，避免悄悄错误合并。
- 可用 `--ignore-index 255` 明确指定忽略区，但不能同时将其设为前景。
- 调色板 PNG 保留原始类别索引；RGB mask 只有在三个颜色通道相同时才按灰度处理。彩色语义标签需要先明确颜色与类别 ID 的映射。
- 图像读为 RGB、除以 255；模型内部按 SAM 均值/标准差归一化。不要在外部重复做 ImageNet/SAM 标准化。
- 默认不做增强或缩放。不同尺寸的 batch 在右侧/底部补零，mask 补 -1；padding 像素不参与 Dice、混淆矩阵和指标。变尺寸 batch 的语义上下文仍可能受 padding 影响，通常建议同尺寸 chips 或 batch size=1。
- 仅在明确需要时使用 `--image-size HEIGHT WIDTH`；图像双线性、mask 最近邻。该选项会改变原始像素尺度，应记录为实验设置。

论文 TTPLA 的非重叠 640×640 裁剪和质量筛选保留 1000 个样本应在训练前准备好。论文未给出质量筛选的完整准则及原始样本清单，脚本不臆造这一步；也不直接读取 TTPLA 原始 COCO/实例 JSON。请先将所需设施实例合并成与原图对齐的语义 mask。PLD 按准备后的原尺寸输入，宽 540、高 360，不需要改为 640×640。

## 5. 论文训练设置

| 参数 | 默认值 | 依据 |
|---|---|---|
| 任务 | 设施／背景二分类 | 已确认的任务范围 |
| 主干 | ViT-L，冻结 | 论文 Ce=1024 与已确认方案 |
| EMS Cr | 32 | 论文 §2.2 |
| loss | 单独 foreground soft Dice | 论文 §3.1 |
| optimizer | AdamW | 论文 §3.1 |
| epochs | 300 | 论文 §3.1 |
| learning rate | 1e-4 | 论文 §3.1 |
| weight decay | 1e-5 | 论文 §3.1 |
| AdamW betas | (0.9, 0.999) | 论文 momentum=0.9 对应 beta1；beta2 未给出，补全为 PyTorch 默认 |
| 学习率调度 | 无，固定学习率 | 论文未给出调度规则，未添加调度 |
| 数值精度 | FP32 | 论文未给出 AMP；BF16 仅作为用户显式选择的运行选项 |
| batch size | 2 | 论文未给出；实现默认，可按显存修改 |
| num workers / seed | 4 / 42 | 实现补充 |
| 数据增强 / 梯度裁剪 | 无 / 无 | 论文未给出，不添加参考脚本的策略 |
| 提示 | 训练和评估均为自动路径 | 无需人工提示；可选提示仅通过模型 API 传入 |

**没有使用参考脚本的 BCE+Dice、损失权重切换、EMA、余弦调度、辅助深监督或默认 BF16。**初始预测仅用于 BRA 的可微边缘路径，不添加额外监督损失。

Dice 的具体平滑和归约方式论文未明确给出。本实现对每张图像的有效像素计算

```text
p = sigmoid(logits)
loss_i = 1 - (2 * sum(p_i * y_i) + 1e-5) / (sum(p_i) + sum(y_i) + 1e-5)
loss = mean(loss_i)
```

全为 ignore 的图像不进入均值；全为 ignore 的 batch 报错。背景图像仍按同一 soft Dice 公式训练。

### 开始训练

```bash
# TTPLA：事先准备好 640×640 的 train/val/test chips
python train_esap_sam.py \
  --data-dir /data/TTPLA_semantic \
  --sam-checkpoint /weights/sam_vit_l_0b3195.pth \
  --backbone vit_l \
  --epochs 300 --batch-size 2 --num-workers 4 \
  --output-dir ./runs/ttpla_vitl

# PLD：保留原尺寸；根据实际标签决定是否需要 --foreground-values
python train_esap_sam.py \
  --data-dir /data/PLD_semantic \
  --sam-checkpoint /weights/sam_vit_l_0b3195.pth \
  --backbone vit_l --batch-size 2 \
  --output-dir ./runs/pld_vitl

# 可选 ViT-B 对照；需要匹配 ViT-B 权重
python train_esap_sam.py \
  --data-dir /data/TTPLA_semantic \
  --sam-checkpoint /weights/sam_vit_b_01ec64.pth \
  --backbone vit_b --output-dir ./runs/ttpla_vitb
```

默认 GPU 可用时用 CUDA。`--device cpu` 可做小尺寸验证。显存不够先降低 `--batch-size`；ViT-L 的冻结权重依然占显存，冻结不等于删除。若明确允许改变数值精度，可在支持 BF16 的 CUDA GPU 上显式加 `--precision bf16`；这不是论文默认配置。

使用 `python train_esap_sam.py --help` 查看全部参数。路径包含空格时使用引号；上面的多行命令是 Bash 写法，在 PowerShell 中可写成单行。

## 6. 日志、指标和 checkpoint

每次训练保存：

```text
runs/ttpla_vitl/
├── config.json            # 实际参数
├── environment.json       # 运行环境与目标版本差异
├── splits.json            # 可复用的相对路径划分
├── history.csv            # 每轮 train/val loss、指标、混淆矩阵、学习率与耗时
└── weights/
    ├── best.pt            # 验证集 mIoU 最高时的同一模型
    └── last.pt            # 最近一轮完整恢复点
```

`best.pt`/`last.pt` 包含完整模型、架构配置、AdamW 状态、epoch、最佳 mIoU、数据清单及 Python/NumPy/Torch/DataLoader RNG 状态。无需再次提供原始 SAM 文件即可从这些完整 checkpoint 恢复。文件包含冻结主干，因此 ViT-L 每份 checkpoint 会超过 1 GB，保存过程需要相应磁盘空间。保存采用临时文件加原子替换；非空输出目录不会被训练覆盖。

恢复到新的输出目录：

```bash
python train_esap_sam.py \
  --data-dir /data/TTPLA_semantic \
  --resume ./runs/ttpla_vitl/weights/last.pt \
  --epochs 300 --output-dir ./runs/ttpla_vitl_resumed
```

恢复时使用 checkpoint 中的架构、batch size、学习率、AdamW 参数、seed、图像缩放、标签映射、阈值和数据划分。可更改数据根目录位置（相对文件路径必须仍存在）、输出路径、device、workers、目标总轮数和精度；若改精度或硬件，不能要求逐位相同。为了保持完全相同的训练顺序，恢复后不要改变数据文件内容。PyTorch 的 cuDNN 确定性设置不能保证跨硬件、跨版本逐位一致。

只在最后独立评估 test 集，验证集只用于选择 checkpoint：

```bash
python train_esap_sam.py \
  --data-dir /data/TTPLA_semantic \
  --resume ./runs/ttpla_vitl/weights/best.pt \
  --eval-only --eval-split test \
  --output-dir ./runs/ttpla_vitl_test
```

输出 DiceLoss、前景 Dice/IoU、背景和前景的 mIoU、Acc、F1、Precision、Recall，以及 TP/FP/FN/TN；数值为 0~1，乘 100 后为百分比。预测阈值默认严格 `>0.5`。混淆矩阵按**整个集合累计**，不会平均各 batch 比率。mIoU 对有非零并集的类别取平均；若整个集合某一类别在预测和真值中均不存在，则排除该类别。分母为零的 Precision/Recall/F1 定义为 0。

**APLS 尚未纳入脚本。**论文显示的双向倒数式在两个方向相似度均不超过 1 时最高为 0.5，与表中超过 0.8 的结果不一致；图提取、节点匹配与采样细节也没有提供。代码不把骨架重合率等代理指标标成 APLS。严格比较论文 APLS 需要作者的评估代码或进一步确认指标定义。这不影响模型训练及已列出的像素级指标。

## 7. 模型调用与可选提示

模型返回 raw logits；训练损失中或推理后只做一次 sigmoid。输出与输入图像尺寸一致。

```python
import torch
from esap_sam import ESAPSAM

device = "cuda"
model = ESAPSAM(
    backbone="vit_l",
    sam_checkpoint="/weights/sam_vit_l_0b3195.pth",
).to(device).eval()
image = torch.rand(1, 3, 640, 640, device=device)  # 实际使用 RGB 图像，值域 0~1

with torch.inference_mode():
    probability = model.predict(image)            # [1,1,640,640]
    binary_mask = probability > 0.5

    # 可选：一个正点、一个负点。坐标顺序为 (x,y)，单位是当前输入图像像素。
    xy = torch.tensor([[[100., 120.], [300., 350.]]], device=device)
    labels = torch.tensor([[1, 0]], device=device)  # 1=设施，0=背景，-1=补齐点
    box = torch.tensor([[50., 50., 500., 600.]], device=device)  # x0,y0,x1,y1
    hint = torch.zeros(1, 1, 640, 640, device=device)           # 用户提供的软掩码
    prompted_logits = model(image, points=(xy, labels), boxes=box, mask_prompt=hint)
    debug = model(image, return_aux=True)
    # debug 含 logits、preliminary_logits、edge_attention，三者均恢复到输入空间。
```

提示参数相互独立，可以只传一个。点要求 `[B,N,2]` 和 `[B,N]`，框要求 `[B,4]`，每张图像至多一个框；mask 要求 `[B,1,H,W]`、值域 0~1。点/框使用传给模型的图像坐标，若事先缩放图像，必须同步缩放坐标和掩码。提示超出图像边界、batch 或形状不匹配会报错。

这些接口提供交互修正通路。默认自动训练不会生成或使用人工提示，因此其交互效果需要额外数据和实验检验，不能据接口存在就声称交互精度提升。

加载训练好的完整 checkpoint：

```python
checkpoint = torch.load("./runs/ttpla_vitl/weights/best.pt", map_location="cpu", weights_only=True)
model = ESAPSAM(**checkpoint["model_config"], allow_random_init=True)
model.load_state_dict(checkpoint["model"], strict=True)  # 随后完整覆盖随机初始化
model = model.to("cuda").eval()
```

这里的 `allow_random_init=True` 只是避免重复提供原始 SAM 文件；完整 checkpoint 必须立即严格加载，不能省略下一行。

## 8. 自检

```bash
# 严格检查截图环境
python train_esap_sam.py --check-env --strict-env

# 不下载权重，检查完整架构的输出和反向传播；尺寸仅为降低测试开销
python esap_sam.py --backbone vit_l --allow-random-init --device cuda --size 64 --batch-size 2 --backward
python esap_sam.py --backbone vit_b --allow-random-init --device cuda --size 64 --batch-size 2 --backward

# 使用真实权重检查初始化；保持 --backbone 与权重匹配
python esap_sam.py --backbone vit_l --sam-checkpoint /weights/sam_vit_l_0b3195.pth --device cuda --size 64
```

`--backward` 会确认 EMS、HP、decoder、BRA 的每个可训练参数均收到有限梯度，冻结主干和几何编码器没有梯度。小尺寸随机输入只用于检查张量维度和训练链路。

## 9. 本次交付验证

本次实测环境是 **Windows、Python 3.13.5、PyTorch 2.9.0+cu126、CUDA runtime 12.6、cuDNN 91002、RTX 2060**，不是截图中的目标 Linux 环境。工程依赖 NumPy/Pillow/tqdm 使用上表精确版本。目标环境尚未在本机实测，应在目标机器运行第 8 节的严格环境检查和自检。

| 模型 | 总参数 | 可训练参数 | 本次检查 |
|---|---:|---:|---|
| 默认 ViT-L | 311,966,912 | 3,682,420 | 完整架构，B=2、64×64，GPU 前向、反向及冻结边界通过 |
| 可选 ViT-B | 93,342,912 | 3,665,780 | 同上；额外完成合成数据训练、保存、恢复和测试 |

已经通过的功能检查：

- ViT-L / ViT-B 的输出维度、所有实际可训练参数的有限梯度、冻结主干无梯度。
- B=2、53×67 非方形输入；整 batch 与逐图自动推理结果在浮点容差内一致，排除了错误的 batch 重复。
- 点、框、mask 和组合提示均影响预测；组合提示反向传播可达 HP，几何编码器保持冻结；越界和非法提示被拒绝。
- 辅助初始预测、边缘注意力以及最终 logits 均恢复到输入尺寸；概率和注意力范围正确。
- 方向卷积的真实七抽头对角支撑及梯度；调色板 mask 保留类别 ID；缺失配对与不明前景 ID 被拒绝。
- ignore/padding 像素不参与损失和混淆矩阵；指标不随 batch 划分方式改变。
- 10 对合成样本按 seed=42 自动生成可重复、互不重叠的 7/2/1 划分；独立脚本的两个 Windows spawn 数据加载进程正常读取变尺寸 batch。
- 使用官方 SAM 参数布局的**合成 checkpoint**，严格初始化加载通过；没有取得实际官方预训练权重，因此不声称完成了真实权重或真实数据精度验证。
- ViT-B 在合成 train/val/test 上完成训练和 CPU 独立测试；完整 checkpoint 可用 `weights_only=True` 加载，随后从第 1 轮恢复训练到第 2 轮。
- 与连续训练两轮相比，恢复训练的验证 loss、验证混淆矩阵和各项验证指标一致，DataLoader/Torch RNG 状态一致；训练 loss 差小于 1e-6。可训练参数观察到最大约 2.52e-5 的浮点差异，**没有达到或承诺逐位一致**；冻结参数保持一致。

验证仅使用小尺寸合成数据和随机初始化／合成 SAM 布局权重，不是 TTPLA/PLD 的 300 轮实验，不代表论文精度、APLS、640×640 显存需求或推理耗时已经复现。

## 10. 来源与许可证

SAM 组件取自 [Meta 官方 Segment Anything](https://github.com/facebookresearch/segment-anything/tree/dca509fe793f601edb92606367a655c15ac00fdf)，固定提交 `dca509fe793f601edb92606367a655c15ac00fdf`，合入 `common.py`、`image_encoder.py`、`prompt_encoder.py`、`transformer.py` 和 `mask_decoder.py`。保留原始版权说明；主要修改为单文件整合、attention 类名区分、去除相对导入及 NumPy 常量依赖，再增加本项目适配模块。代码中的 `LICENSE` 引用所对应的原始许可证全文附在下方，随本工程分发。原论文中的性能数值不作为本实现的实测值。

<!-- UPSTREAM_LICENSE -->

## SAM 第三方代码许可证

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

```
