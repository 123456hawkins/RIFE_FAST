# 双 RTX 3090 适配与多卡调度

> 本文只保留结论、命令和可复现数据；架构细节见 `optimization.md`。

## 0. 环境与结论

| 项 | 值 |
|---|---|
| GPU | 2 × RTX 3090 24 GB（sm_86，82 SM，PCIe 4.0 ×8） |
| CPU | 32 核 / 125 GB |
| torch / TRT | 2.14.1+cu130 / 11.4.0.106 |
| ffmpeg | 系统 apt 4.4.2（h264_nvenc / hevc_nvenc） |
| 对照 | 本仓库 4090 文档（同代码） |

**3090 单卡 720p 2x**：端到端 **222 对/s**，模型侧 **258 对/s**，PyTorch 端到端 169 对/s。
**双 3090 720p 2x**：9600 帧 **384 对/s（1.74×）**；2400 帧 289 对/s（1.30×）。
**双 3090 1080p 2x**：1800 帧 **158 对/s（1.32×）**。

结论：

1. 3090 约为 4090 的 50~65%，瓶颈是 SM 数/频率，不是显存带宽。
2. 3090 上 TRT 相对 torch：模型 +29%、端到端 +32%，比 4090 收益更大。
3. 单卡 3090 的 NVENC（1 个 7 代编码器）不是唯一瓶颈；多卡同时增加模型与编码吞吐。
4. 双卡加速受每卡约 2 s 启动和末尾 concat 限制：长素材接近 2×，短视频约 1.3×。

## 1. 适配修复

| 问题 | 修法 |
|---|---|
| `ModuleNotFoundError: cv2` | 新增 `src/imgio.py`（stdlib PNG 解码）；validate/quality/compare/bench 不再依赖 opencv |
| ffmpeg 不在 PATH | 系统 apt 安装；`src/ffmpeg_util.py` 支持 `FFMPEG` 兜底并给出可操作错误 |
| 4090(sm89) 引擎被 3090(sm86) 误用 | 引擎名加 `sm<arch>_trt<ver>`；反序列化失败回退 PyTorch，不再 `sys.exit` |
| 换 `--model` 仍用默认权重指纹 | 指纹改为实际权重目录的 md5 |
| 双 worker 同时建引擎 | `flock` 串行 + write+rename 原子落盘 |
| 工具脚本旧 bug | `quality.py` 路径/`module.*`/`.inference`；`bench_gpu.py` 前缀/`merged[-1]` 已修 |

旧的无 arch tag 引擎会被忽略并重建（一次性）。

## 2. 多卡调度

每张卡一个 `python -m src._worker` 子进程（独立 CUDA context / TRT 引擎 / CUDA Graph）。
父进程写任务清单，worker 用 `flock` 原子领取 `next`：

- **完成即取下一个**，不是静态分配；空闲卡自动补位。
- 每任务写 `status/task_XXXXX.json`；失败/缺失（含 worker 被 kill）只重试该任务一次。
- 单输入无法 `ffprobe` 时从本批排除，其他文件继续，最终退出码 1。
- 分段任务实际解码帧对数不符时直接失败；拼接后再严格校验总帧数。
- `--encoder null` 也走队列（可用于多卡模型吞吐测试），不写文件、不拼接。

`--gpu-strategy auto`：

| 输入 | 选择 |
|---|---|
| 1 个视频 | `segments`（短视频 1 段/卡，≥120 s 每卡 2 段） |
| ≥2 个，时长均衡，N 是 GPU 数整数倍 | `files`（一卡一视频，无 part/concat） |
| ≥2 个，N 不是 GPU 数整数倍 | `segments`，按估时重新均分（等长时每视频 `ceil(N/G)` 段） |
| ≥2 个，最长视频明显超过“总时长/GPU 数” | `segments`，只拆长视频 |

分段按帧对切：解码 `[p0,p1]`（多 1 帧），输出 `(p1-p0)*multi+1` 帧，后续段丢首帧；
拼接后严格等于 `(N-1)*multi+1`。段间无卡间搬运，父进程只做 `ffmpeg -f concat -c copy`。
音轨只在最终文件复制一次。

## 3. 正确性与回归

| 用例 | 结果 |
|---|---|
| `validate.py` | ref 0，assemble 120.84 dB，pyramid 43.25 dB，fold 133.30 dB |
| 720p 2x 无损单卡 vs 双卡 | 479 帧，PSNR 99.00 dB，max\|d\|=0 |
| 29.97 fps + 短 GOP | 599 帧，PSNR 99.00 dB，max\|d\|=0 |
| 4x / 8x | 237 / 473 帧，公式一致 |
| 1 / 2 / 3 帧素材 | 1 / 3 / 5 帧 |
| AAC / h265 / 路径空格 / rgb24 | 均正常 |
| 损坏输入混批次 | 正常文件完成、坏文件报错、退出码 1 |
| worker 被 `kill -9` | 另一卡偷走剩余分段，只重试丢失的 1 段；19199 帧、退出码 0 |
| 双 worker 首次建同一引擎 | 一个 built、一个 waiting；共用 plan，输出正确 |
| 3 个等长视频 / 2 卡 | 6 个任务均衡分配；输出正确 |
| 单元测试 | `python -m unittest discover -s tests` 15/15 |

有损编码下单卡/双卡不会逐字节相同（分段 GOP 起点不同）；插帧结果本身逐位一致（无损可证）。

## 4. 性能

单卡 720p 2x（9600 帧长跑）：

| 指标 | 4090 文档 | 3090 |
|---|---:|---:|
| 端到端 | 375 对/s | **222 对/s** |
| 模型侧 | 507 对/s | **258 对/s** |
| PyTorch 端到端 | 305 对/s | 169 对/s |

双卡（本机）：

| 场景 | 单卡 | 双卡 | 加速 |
|---|---:|---:|---:|
| 720p 2x，9600 帧 | 222 | **384** | **1.74×** |
| 720p 2x，2400 帧 | 222 | 289 | 1.30× |
| 1080p 2x，1800 帧 | 120 | **158** | **1.32×** |
| 两个 2400 帧文件（files） | 24.8 s 串行 | **13.6 s** | **1.82×** |
| 100 s + 10 s（auto） | 13.1 s（files） | **9.4 s** | **−28%** |

启动约 1.5~2 s/卡，末尾 concat 约 0.2 s；素材越长越接近 2×。
本机 GPU1 满载约 93 ℃、SM 1322 vs 1584 MHz，热降频约 7%；改善散热优先于继续调软件。

## 5. 优化调研结论

| 方向 | 结论 |
|---|---|
| `--batch-pairs` | 3090 仍以 4 最优；6/8 模型 +2~3%，端到端 <1%，不改默认 |
| `--pad-mult 32` | 模型 +3.4%，但输出对 64 仅 39.45 dB / 1.2% 像素 >2 级 → **否决** |
| TRT 跨卡 plan | 同架构可跑，性能差在噪声内；需干净日志用 `--engine-per-gpu` |
| Head 跨 chunk 复用 | 2x 仅省约 2% 模型，且破坏指针别名，收益 <1% |
| 更多分段 | 长视频 2×GPU 段已用；短视频更多段只增启动/IDR |
| FP8 / INT8 | Ampere 无 FP8；INT8 画质风险，不做 |
| 同卡多 context | 3090 只有 1 个 NVENC；同卡两进程争 context，不做 |

## 6. 复现与限制

```bash
sudo apt install -y ffmpeg          # 与 4090 相同
uv sync
uv run python -m src.validate
uv run python -m unittest discover -s tests -v

# 单视频：自动分段
uv run python -m src -i in.mp4 -m 2 -o out --gpus 0,1

# 多文件：共享队列，完成即取下一个
uv run python -m src -i a.mp4 b.mp4 c.mp4 -m 2 -o out --gpus all

# 强制整文件 / 强制分段
uv run python -m src -i a.mp4 b.mp4 --gpus 0,1 --gpu-strategy files
uv run python -m src -i in.mp4 --gpus 0,1 --gpu-strategy segments

# 无损正确性对比
uv run python -m src -i in.mp4 -o s --encoder x264 --quality 0
uv run python -m src -i in.mp4 -o d --encoder x264 --quality 0 --gpus 0,1
uv run python -m src.tools.compare_videos s/in_2X.mp4 d/in_2X.mp4 1280 720
```

限制：

- auto 是估时启发式，不是全局最优调度；短素材分段收益可能被启动开销吃掉。
- 分段会生成 `.rife_queue_*` 临时 part；成功删除，失败保留。磁盘约等于最终输出。
- 严格帧校验依赖 `ffprobe` 帧数；未知/近似时退化为整文件任务。
- 同型号双卡共用 plan 有 TensorRT 警告；需要时 `--engine-per-gpu`。
- 仅支持单机多卡；跨主机需要共享存储/任务服务。
