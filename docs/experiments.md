# 评估过但没采用的方案（归档）

这份文档记录"为什么不做 X"，避免以后重复踩坑。所有数字都是在同一台 4090 上实测的。

## 1. 提案：全显存驻留的 C++ 流水线（NVDEC→TRT→NVENC，零 Host Copy）

结论：**方向对，但收益被高估；换语言本身不产生加速。**

| 提案里的说法 | 实测/核查 |
|---|---|
| "特征复用省 30–45%" | 2x 只有 **+5.4%**；8x 才 **+30.6%**。Encoder 只占 13.6% FLOPs |
| "零拷贝，PCIe 占用接近 0" | 成立，但只值 **+5.6%**（见 §2） |
| "CUDA Graph 让间隙缩短 80%" | 实测 **+16%**（`torch.compile max-autotune` +27%） |
| "FP8 吞吐翻倍" | 算力只用了 fp16 峰值的 **6.7%**，翻倍无意义 |
| "L2 persistence 再提 15–25%" | 无法确认：中间张量依次产生依次消费，没有热点 |
| "4090 4K 实时" | 114 fps 输出 → 能实时处理 4K57 源，4K60 勉强 |
| "4060 1080p 130–160 fps" | 偏乐观，按 4090 线性外推约 **100 fps** |

另外提案漏掉的最大一块：**CPU 侧的 RGB→YUV（swscale 单线程，345 fps）**，
在纯 CUDA 架构里天然没有，但用 ffmpeg 管道时会卡死整条流水线。
把颜色转换搬到 GPU（NV12）后，端到端 155 → 226 对/s（**+45%**）。

## 2. CUDA IPC 零拷贝编码（已实现、已测、已删除）

实现方式：C++ 进程 `cuMemAlloc` 一块 NV12 显存环形池，`cuIpcGetMemHandle` 导出；
Python 用 `cuIpcOpenMemHandle` 映射，`cuMemcpyDtoD` 把模型输出直接写进显存（不过 PCIe），
发命令给 C++ 侧 NVENC 编码。

| 路径（720p 2x，同编码参数） | 对/s | 输出 fps |
|---|---|---|
| ffmpeg 管道（现状） | 181.4 / 182.6 | 363 / 365 |
| **CUDA IPC 零拷贝** | **192.7 / 192.6** | 385 |
| IPC 但跳过编码（只测交接） | ≈300 | 600 |
| IPC + TensorRT | 193.8 | 388（**不如** 管道的 284） |

**只有 +5.6%。** 分段计时显示：编码器一进来就吃掉 7.5 ms/chunk，
模型和编码器没有真正并行；真正的天花板是 NVENC 本身（真实内容 650–700 fps），不是 D2H。
而且它和 TensorRT 不叠加（IPC 路径每 chunk 要做一次设备同步，把模型提速全吃掉了）。

三个必须知道的坑：

1. **`cuIpcOpenMemHandle` 的 flags 不能传 0**，必须传
   `CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS`，否则返回 `CUDA_ERROR_INVALID_VALUE`；
2. 进程退出前要显式 `nvEncDestroyEncoder()`，否则内部队列没排空，进程退出挂死
   （`cuCtxSynchronize` 同样会挂）；
3. `cuda-python` 里要先让主 context 成为 current（`torch.zeros(1,device='cuda')`），
   否则 `cuMemAlloc` 报 `CUDA_ERROR_INVALID_CONTEXT`。

## 3. 纯 C++ 只做编码（探针，已删除）

用 `nv-codec-headers` + `nvEncodeAPI.h` 手写了 `CUdeviceptr` 直喂的零拷贝编码器：

| 输入路径 | 内容 | fps |
|---|---|---|
| ffmpeg 管道（`nv12 → hevc_nvenc p3 hq cq26 bf0`） | 真实帧 | **938** |
| C++ `CUdeviceptr` 零拷贝（同步 lock/write） | 真实帧 | **643** |
| C++ 零拷贝 | 纯色帧 | 1030 |

**C++ 反而更慢 32%**：探针是"编码一帧→立刻 lock→fwrite"，CPU 与编码器完全串行；
ffmpeg 内部是多缓冲异步的。**语言不是收益来源，流水线结构才是**。

两个坑：驱动 NVENC API 是 **13.0**，最新头文件 13.1 → 必须 checkout `n13.0.19.1`；
`lookahead>0` + 无条件 `nvEncLockBitstream` 会死锁（编码器要攒够 20 帧才出第一帧输出）。

## 4. 其他

- **channels_last**：K=1 快 8%，K=4 反慢 1%。省下的 0.72 ms 布局转换被
  0.39 ms 碎片化 elementwise + 0.197 ms NHWC padding 抵消。只有 `Head` 单项受益（285→177 µs）。
- **双 NVENC 分流**：4090/4080 有 2 个编码引擎，按奇偶帧切分理论 ~2×；用户明确排除，未做。
- **轻量模型（4.25.lite）**：画质有损，要求保效果，排除。
- **TensorRT 动态 shape**：静态 shape 每层能独立选 tactic（更快），代价是换分辨率要重建引擎；
  现在由 `--trt-engine auto` 按 `rife_<W>x<H>_b<batch>_<权重指纹>.engine` 自动处理。

## 5. 双 NVENC 分流（实测不可行）

硬件层面确实能并行：同一批帧喂两个 ffmpeg 进程，

| 场景 | 吞吐 |
|------|------|
| 单进程 h264_nvenc | 1031 fps |
| **两进程并发** | 787 + 773 = **1560 fps（1.41×）** |

但**整条流水线跑不通**：视频是单一有序流，只能按时间分段拆成两个独立流水线，
而每个流水线都要自己跑一遍模型：

| 方案 | 结果 |
|------|------|
| 1 个进程 | 267 对/s |
| **2 个进程各跑一半时间轴** | 79.9 + 78.9 = **159 对/s（慢 40%）** |
| 3 个进程 | 更慢 |

原因不是编码器，而是**两个 CUDA context 争抢同一块 GPU**：模型每 chunk 要 10.4 ms，
两份一起跑时每份只剩 ~1/3 的吞吐（比均分还差，context 切换 + cache 抖动）。
输出核对过：分段拼接后帧数正确（959 帧），只是慢。

真正的双 NVENC 用法是**空间切片**（每帧上下半各交给一个 encoder，再把 slice NAL 拼回
同一个 picture）。这需要 NVENC 的 slice mode + 手写码流拼接，ffmpeg 完全不支持，
属于大型 C++ 工程，且收益上限也就是把编码从 14 ms 降到 ~8 ms/chunk。

**结论：单流场景放弃。** 代码已回滚，只保留这份记录。

## 6. 解码 NV12 / 原帧直通 / pinned readinto（评估后不做）

提案：解码直接出 NV12、原始帧零拷贝进编码器、读线程预分配 pinned 缓冲 `readinto`。

按实测的分段数据，`wall = 15.08 ms/chunk` 里：

| 项 | 时间 | 占 wall |
|----|------|---------|
| 主线程 CPU 全部（read+pin+h2d+fwd） | 4.36 ms | **29%**（且与 GPU 重叠） |
| 其中 `pin`（拷进 pinned） | 1.30 ms | 8.6% |
| 读线程（解码 + swscale） | 不在关键路径 | — |
| **编码器** | **14.02 ms** | **瓶颈** |

- `readinto` 预分配能省掉 pin 的一半（~0.65 ms/chunk）→ 上限 **+4%**
- 解码改 NV12 省掉的是 swscale 的 CPU 时间，而读线程本来就有余量 → **0**
- 原帧直通能省掉一半 RGB→NV12 转换（在 GPU 上，~1.1 ms/chunk）→ 但 GPU 不是瓶颈 → **0**

**结论：收益 <5%，且要改解码/上传/编码三条链路，性价比低。**
如果将来 CPU 成为瓶颈（比如 4K 或更慢的 CPU），这套改造才有意义。
