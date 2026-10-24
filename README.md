# RIFE_FAST

RIFE 4.25 的高速插帧流水线：**NVDEC 解码 → RIFE 推理（TensorRT）→ NVENC 编码**，
全部在 GPU 上完成颜色转换，端到端不落盘不转 RGB。

在 RTX 4090 上的实测（720p 2x，含解码与编码）：

| 指标 | 原始 `inference_video.py` | 本工具 | 倍数 |
|------|--------------------------|--------|------|
| 720p 2x | ~100 对/s | **274 对/s（548 fps 输出）** | **2.7×** |
| 1080p 2x | — | 116 对/s（232 fps） | — |
| 画质 | 基准 | 与原始 fp32 **51.1 dB**，`assemble` 档**逐位一致** | 无损 |

技术细节与所有实测数据见 [`docs/report.md`](docs/report.md)，
做过但放弃的方案见 [`docs/experiments.md`](docs/experiments.md)。

---

## 1. 安装

### 1.1 环境要求

| 组件 | 要求 |
|------|------|
| Python | ≥ 3.10 |
| GPU | NVIDIA RTX 20/30/40 系（需要 NVENC + NVDEC） |
| 驱动 | ≥ 525（能跑 CUDA 12.x 即可） |
| ffmpeg | 系统安装，**必须带 nvenc**：`ffmpeg -encoders \| grep nvenc` |
| 包管理 | [uv](https://docs.astral.sh/uv/)：`curl -LsSf https://astral.sh/uv/install.sh \| sh` |

### 1.2 下载模型（必需）

> ⚠️ **模型权重不在本仓库里**（避免仓库过大），需要手动下载一次。

1. 打开上游仓库：<https://github.com/hzwer/Practical-RIFE>
2. 在 **Model list** 里找到 **RIFE 4.25**（建议版本，见 §5），下载对应的
   `flownet.pkl`
3. 放到本项目的 `models/` 目录：

```bash
mkdir -p models
mv ~/Downloads/flownet.pkl models/flownet.pkl
```

放好后应该是：

```
models/
└── flownet.pkl          # 约 24 MB
```

程序的网络结构文件（`IFNet_HDv3.py`）**已经内置**在包里，通常不需要下载。
只有当你换用架构不同的新版本（如 4.26 若改了结构）时，才把该版本的
`IFNet_HDv3.py` 一起放进 `models/`——同目录下的文件会优先生效。

### 1.3 安装依赖

```bash
uv sync               # torch + TensorRT + ONNX + numpy + tqdm
uv sync --extra png   # 额外装 opencv（只有 --png 输出和 PNG 对比工具需要）
```

TensorRT 通过 pip 安装（`tensorrt` 包自带 CUDA 库），不需要另外装 NVIDIA 的 tarball。
首次运行时会自动导出 ONNX 并构建引擎（约 2 分钟/分辨率）。

---

## 2. 快速开始

```bash
# 720p 2x —— 默认就是实测最快的配置
uv run rife-fast -i in.mp4 -m 2 -o out.mp4

# 加上 TensorRT（再快 ~20%）
uv run rife-fast -i in.mp4 -m 2 --trt-engine auto -o out.mp4

# 4x / 8x
uv run rife-fast -i in.mp4 -m 4 -o out4x.mp4
uv run rife-fast -i in.mp4 -m 8 -o out8x.mp4

# 更小的文件
uv run rife-fast -i in.mp4 -m 2 --trt-engine auto --quality 26 -o small.mp4
```

`uv run rife-fast` 与 `uv run python -m rife_fast` 等价；如果你已 `uv sync`，
也可以在 `.venv` 激活后用 `python -m rife_fast`。

---

## 3. 参数说明

### 3.1 基本参数（日常只用这些）

| 参数 | 说明 |
|------|------|
| `-i, --input` | 输入视频（必填） |
| `-o, --output` | 输出视频；默认 `输入名_2X_48fps.mp4` |
| `-m, --multi` | 倍率 2/4/8（`--exp 3` 等价于 `-m 8`） |
| `--codec` | `auto`（默认）：≤1080p 用 h264，>1080p 用 h265。也可强制 `h264`/`h265`/`x264`/`x265` |
| `--quality`（别名 `--cq`） | 画质档 18–30，越小越清晰、文件越大。默认 **22** |
| `--bitrate 4M` | 改用固定码率（放弃按画质） |
| `--fps` | 输出帧率，默认 源帧率 × 倍率 |
| `--trt-engine auto` | 用 TensorRT，按分辨率/批次/权重自动找或建引擎 |
| `--png` | 输出 PNG 序列而不是视频 |
| `--no-audio` | 不合并音轨 |

### 3.2 高级参数（一般不用动）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--batch-pairs` | 自动 | 每次前向的帧对数（720p 2x→4、4x→2、8x→1；1080p→4；2K→2） |
| `--mode` | `pyramid` | `pyramid`=最快 / `assemble`=与原实现逐位一致 / `ref`=原始实现 |
| `--scale` | 1.0 | 内部工作分辨率倍数，4K 可用 0.5 降算力 |
| `--fp32` | 关 | 关闭 fp16（默认 CUDA 上开启） |
| `--no-cudagraph` | 关 | 关闭 CUDA Graph（默认开启，捕获失败会自动降级） |
| `--compile` | off | `torch.compile`（与 cudagraph 二选一） |
| `--encoder` | auto | `nvenc` / `x264`（CPU）/ `null`（只测流水线不编码） |
| `--nvenc-preset` `--tune` `--nvenc-bf` | 按 codec 自动 | 手动覆盖编码器档位 |
| `--model` | `models/` | 权重目录（也可用环境变量 `RIFE_MODELS`） |
| `--pad-mult` | 64 | H/W 向上对齐的倍数 |
| `--bench --width --height --iters` | — | 只测模型吞吐，不读写视频 |

### 3.3 画质档位怎么选

实测（Y 平面 PSNR，参考为无损；「人眼不可闻」≈ PSNR ≥ 50 dB 且差异像素 < 2.5%）：

| `--quality` | 720p 码率 | PSNR | 判断 | 2 小时电影 |
|------|-----------|------|------|-----------|
| 20 | 6.93 Mbps | 51.5 dB | 视觉无损 | 6.2 GB |
| **22（默认）** | **5.97 Mbps** | **50.2 dB** | **人眼不可闻** | **5.4 GB** |
| 24 | 4.89 Mbps | 48.4 dB | 高质量 | 4.4 GB |
| 26 | 4.08 Mbps | 46.9 dB | 平衡 | 3.7 GB |
| 30 | 2.76 Mbps | 43.8 dB | 小体积 | 2.5 GB |

速度与 `--quality` 无关（编码器按像素计费）。

### 3.4 h264 还是 h265？看分辨率

等质量（等 PSNR）实测：

| 分辨率 | h265 相对 h264 | 建议 |
|--------|---------------|------|
| 720p | **+8%（更费）** | h264 |
| 1080p | −9% | 都行，求小用 h265 |
| 1440p | **−25%** | h265 |

所以默认 `--codec auto`：≤1080p 用 h264，>1080p 用 h265。
h265 的编码速度约为 h264 的 75%（1080p：87 vs 116 对/s）。

---

## 4. 目录结构

```
RIFE_FAST/
├── models/            # flownet.pkl（自己下载，git 忽略）
├── engines/           # TensorRT 引擎（自动生成，git 忽略）
├── assets/            # 自检用的两张示例图
├── rife_fast/
│   ├── cli.py         # 命令行入口
│   ├── net.py         # 前向实现（ref / assemble / pyramid 三档）
│   ├── engine.py      # ONNX 导出 / 引擎构建 / TRT 运行时
│   ├── weights.py     # 权重与网络结构加载
│   ├── validate.py    # 换权重后的数值自检
│   ├── warplayer.py   # 反向 warp（与上游逐位一致）
│   ├── IFNet_HDv3.py  # 内置网络结构（可被 models/ 里的同名文件覆盖）
│   └── tools/         # 开发期工具（bench/profile/quality/...）
├── docs/
│   ├── report.md      # 性能报告与所有实测数据
│   └── experiments.md # 已放弃方案的归档
└── pyproject.toml
```

---

## 5. 更换 RIFE 版本

> 当前内置/验证的是 **RIFE 4.25**（上游 README 推荐"大多数场景用 4.25"）。

**TensorRT 引擎里烘焙了权重**，所以换 `flownet.pkl` 后必须重建引擎。
本工具用**权重指纹**自动处理：引擎文件名带 `flownet.pkl` 的 md5 前 8 位，
换权重后文件名自动变化，下次运行自动重建，不会误用旧引擎。

步骤：

```bash
# 1) 覆盖权重
cp new/flownet.pkl models/flownet.pkl
#    若该版本改了网络结构，把它的 IFNet_HDv3.py 也放进 models/

# 2) 自检（30 秒，必跑）
uv run python -m rife_fast.validate
```

四个判据：

| 输出 | 期望 | 含义 |
|------|------|------|
| `[ref]` | **恰好 `0.000e+00`** | 权重 ↔ 前向逐位一致（key 对不上会立刻报错） |
| `[assemble]` | **≥ 120 dB** | 5 个 block 的 feat/mask/flow 拼接顺序未变 |
| `[pyramid]` | ≥ 40 dB | 低分辨率 warp 近似仍成立 |
| `[fold]` | ≥ 80 dB | beta 折叠的数值代价 |

`[assemble]` 若掉到 40 dB 量级 → 新版本改了前向结构，请先用 `--mode ref`，
再改 `rife_fast/net.py` 的对应分支。

```bash
# 3) 直接跑，引擎自动重建（720p≈2min / 1080p≈4min）
uv run rife-fast -i in.mp4 -m 2 --trt-engine auto -o out.mp4
```

---

## 6. 常见问题

**Q: 报 `flownet.pkl not found`？**
见 §1.2，去 <https://github.com/hzwer/Practical-RIFE> 下载 4.25 的 `flownet.pkl` 放进 `models/`。

**Q: 输出文件太大？**
默认 `--quality 22` 约 6 Mbps。改用 `--quality 26`（约 −32% 体积）或
`--bitrate 4M`。结束时会打印实际码率，超过源码率 1.8 倍还会告警。

**Q: 显存不够？**
2K 用 `--batch-pairs 2`，4K 用 `--scale 0.5`。720p 峰值约 1.2 GiB。

**Q: 换分辨率要重建引擎吗？**
要，引擎绑定 `(W, H, batch)`。用 `--trt-engine auto` 会自动处理。
注意**改 `--batch-pairs` 也会触发重建**。

**Q: 为什么 4x/8x 的输出 fps 和 2x 差不多？**
流水线受"输出帧数"限制（编码器/IO），不是帧对数。倍率越高帧对/s 按比例下降。

**Q: 能不能不用 TensorRT？**
可以，不加 `--trt-engine auto` 就走 PyTorch（fp16 + CUDA Graph），慢约 25%。
TensorRT 已经是必需依赖，不需要时可从 `pyproject.toml` 删掉。

**Q: `--png` 报缺 cv2？**
`uv sync --extra png`。

---

## 7. 开发工具

```bash
uv run python -m rife_fast.validate           # 换权重必跑
uv run python -m rife_fast.tools.bench_gpu    # 各配置逐项对比
uv run python -m rife_fast.tools.profile_gpu  # kernel 级瓶颈
uv run python -m rife_fast.tools.model_cost   # FLOPs/访存成本模型
uv run python -m rife_fast.tools.quality      # 画质 A/B
uv run python -m rife_fast.tools.compare_videos a.mp4 b.mp4 1280 720
```

环境变量 `RIFE_STATS=1` 会打印流水线分段计时（read/pin/h2d/fwd/bufwait）。

---

## 8. 引用与许可

- RIFE: *Real-Time Intermediate Flow Estimation for Video Frame Interpolation* (ECCV 2022)
- 上游实现：[hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE)（MIT）
- 本仓库代码同为 MIT；模型权重版权归上游所有，请遵循其许可。
