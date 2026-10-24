# 架构级性能优化：从 v1 到 v2

> 本文记录 RIFE_FAST 从 v1（单引擎 + RGB24 管道 + eager epilogue）到
> v2（双引擎 + NV12 全链路 + Triton 融合内核）的改动、实测收益和踩过的坑，
> 并给出剩下的空间。
>
> 与其它文档的分工：
> - [`report.md`](report.md)：v1 的优化阶梯、画质/体积数据、编码器选型；
> - [`experiments.md`](experiments.md)：试过但不做的方案归档；
> - **本文**：v2 的架构、逐项实测、上游（vs-rife / vs-mlrt）源码对照。
>
> 除标注「估算」外所有数字都是本机实测。环境：RTX 4090 / i9-13900K /
> torch 2.14.1+cu130 / TensorRT 11.4.0 / 驱动 595.91 / ffmpeg 4.4.2。

---

## 0. 一页结论

**端到端（720p，2400 帧素材，含解码与编码）**

| 场景 | v1 | **v2** | 提升 |
|------|----:|-------:|-----:|
| 720p 2x | 307 对/s（614 fps） | **352 对/s（705 fps）** | **+15%** |
| 720p 4x | 109 对/s（438 fps）¹ | **145 对/s（580 fps）** | **+33%** |
| 720p 8x | 52 对/s（415 fps）¹ | **68 对/s（542 fps）** | **+30%** |
| 1080p 2x | 148 对/s（296 fps） | **186~187 对/s（372 fps）** | **+26%** |
| 720p 2x 模型侧（`--encoder null`） | 388 对/s | **504 对/s** | **+30%** |

¹ 4x/8x 的 v1/v2 都测自同一段 20 秒素材（`report.md` §6 口径）；2400 帧素材上
v2 实测 146 对/s（4x）/ 68（8x）。2x 与 1080p 行用 2400/1920 帧素材。

**四项改动，按实测收益排序**

| # | 改动 | 收益 | 说明 |
|---|------|------|------|
| 1 | **Head / IFNet 拆两个 engine，捕进同一张 CUDA Graph**，2x 时 body 输入用**指针别名**指向 head 输出 | 模型 8.04→6.93 ms（**−13%**） | §3 |
| 2 | **NV12 全链路**：解码直接出 NV12，源帧原样透传，模型输入在 GPU 转 RGB | H2D 0.70→0.35 ms，源帧**逐位不变** | §4 |
| 3 | **Triton 融合 epilogue**：33 个 kernel → 3 个 | 0.68→**0.04 ms**（−94%） | §5 |
| 4 | **编码器参数扫描** | **0**（所有档位在噪声内） | §6 |

**一句话**：v1 里"模型和编码器各占 8~13 ms、互相拖累"的格局，v2 变成了
**编码器 11.1 ms / 模型 8.1 ms（同一个 chunk），流水线 11.3 ms —— 已经贴住编码器**。
模型侧还能再快，但端到端不会再快多少，除非减少要编码的像素。

---

## 1. 测量口径

| 项 | 值 |
|---|---|
| GPU / CPU | RTX 4090 24G / i9-13900K（24C32T） |
| 720p 素材 | `demo720_long.mp4` ×5 = 2400 帧 1280×720@24fps，600 chunk |
| 1080p 素材 | `demo1080p.mp4` ×20 = 1920 帧 1920×1080@24fps |
| 输出 | 2x，h264_nvenc `p1 ull constqp22 -bf0 -g600`，NV12 直通 |
| chunk | K=4 对帧 = 8 个输出帧（720p/1080p 2x）；4x 用 K=2、8x 用 K=1 |

三个**口径陷阱**：

1. **不要用 `RIFE_STATS=1` 的数当基线**：它每 chunk 插两个 event 并
   `synchronize()`，`wall` 偏高 2%~9%。
2. **素材长度显著影响绝对值**：同一配置 480 帧 vs 2400 帧差 5%~10%
   （启动、引擎反序列化、Triton 编译摊到更少的 chunk 上）。
3. **GPU 状态会整体漂移 ±5%**：同一份代码同一天内 1080p 从 187 掉到 179
   （核心时钟 2730→2520 MHz），但 torch/TRT 的**比值稳定在 +15% / +25%**。
   跨表只比同一时段测的相对值。
3. **编码器的"空载 fps"要用长样本测**：480 帧测出 960 fps，1920 帧测出
   **1500 fps** —— 差的全是启动开销。用短样本会误判"编码器是瓶颈"。

---

## 2. v2 的流水线

### 2.1 拓扑

```
ffmpeg (nvdec)                                  ┌──────────── 主线程 ─────────────┐
   │ nv12, pipe                               │                                 │
   ▼                                          │  pin_in[i].copy_(nv12 frame)     │
FrameReader ──queue(8)──► window[K+1] ────────┤  .to(cuda, non_blocking) 6.9 MB  │
   (线程)                                       │                                 │
                                               │  [Triton] nv12 -> RGB fp16       │
                                               │       [K+1,3,PH,PW]，pad 区常零  │
                                               │                                 │
                                               │  ┌── 一张 CUDA Graph ─────────┐  │
                                               │  │ head.engine (K+1 帧)       │  │
                                               │  │   -> feats[K+1,4,PH,PW]    │  │
                                               │  │ (n>1 时图内 index_select；  │  │
                                               │  │  2x 时纯指针别名)           │  │
                                               │  │ body.engine (B 个样本)      │  │
                                               │  └────────────────────────────┘  │
                                               │                                 │
                                               │  [Triton] RGB -> NV12：          │
                                               │    源帧   = 直接拷解码器 NV12    │
                                               │    插值帧 = 网络 luma + 2x2 chroma│
                                               │      -> D2H (11.1 MB, copy 流)   │
                                               └────────────────┬────────────────┘
                                                                ▼
                                                   FrameWriter（线程）→ ffmpeg nvenc
```

### 2.2 一个 chunk 的实测构成（720p 2x，8 输出帧）

| 段 | v1 | v2 | 备注 |
|----|----:|----:|------|
| H2D（pinned→dev） | 0.70 ms | **0.35** | 13.8→6.9 MB（NV12 是 rgb24 的一半） |
| 模型输入转换 | 0.08 | **0.01** | `u8→f16/255` → NV12→RGB 融合内核 |
| Head + IFNet（GPU） | 8.04 | **6.93** | 双引擎 + 单图 + 别名 |
| epilogue（GPU） | 0.68 | **0.04** | 33 kernel → 3 kernel |
| D2H（copy 流） | 0.62 | 0.62 | 11.1 MB，与下一 chunk 重叠 |
| **模型侧合计** | 10.31 | **8.05** | `--encoder null` 388→504 对/s |
| 编码器（同一 chunk 内） | 13.0 | **11.1** | 有效 707 fps（空载 1500 fps） |
| **端到端 wall** | 13.03 | **11.32** | 307→352 对/s |

两级仍然只重叠约 2/3：`sum = 8.05 + 8.3(空载编码) = 16.4`、`max = 8.3`，
实测 11.3 → 重叠率 `(16.4−11.3)/(16.4−8.3) = 63%`。缺的 1/3 是**显存带宽争用**
（§6），不是 CPU、不是 PCIe（每 chunk 共 22 MB，约 1.1 ms @20 GB/s）。

---

## 3. 改动 1：双引擎 + 单图 + 指针别名

### 3.1 问题：Head 在引擎里被算了 2B 次

v1 导出的 ONNX 有 1007 个节点，`Conv=56`：5 个 IFBlock 贡献 50 个，
**多出来的 6 个正好是两个 Head**（`IFNetFast.forward` 里 `encode(img0)`、
`encode(img1)` 各导出一次）。批里每个 sample 都带一份自己的 `img0/img1`，
所以 batch=B 的引擎实际做了 **2B 次单帧编码**：

| 倍率 | B | 引擎实际 Head 次数 | 真正需要 | 浪费 |
|------|--:|------------------:|---------:|-----:|
| 2x (K=4) | 4 | 8 | 5 | +60% |
| 4x (K=2) | 6 | 12 | 3 | +300% |
| 8x (K=1) | 7 | 14 | 2 | +600% |

Head 单帧实测 0.16~0.21 ms（1280×768 fp16），占模型 GPU 时间 18~19%。

### 3.2 做法

`src/engine.py` 导出两个静态引擎：

```
rife_<W>x<H>_k<K>m<M>_<tag>.head.engine   [K+1,3,PH,PW] -> [K+1,4,PH,PW]
rife_<W>x<H>_k<K>m<M>_<tag>.body.engine   i0,i1,f0,f1,t  -> out[B,3,PH,PW]
```

`TRTNet` 同时建两个 `IExecutionContext`，把它们**按顺序捕进同一张 CUDA Graph**
（`torch.cuda.graph` 里依次 `execute_async_v3`），每 chunk 只 `replay()` 一次。

`multi == 2` 时 `B == K`，body 的四个输入正好是 `x` / `feats` 的连续切片，
于是**直接把 body 的输入指针绑到 head 的缓冲区上**：

```python
bctx.set_tensor_address("i0", x.data_ptr())          # 帧 0..K-1
bctx.set_tensor_address("i1", x.data_ptr() + o3)     # 帧 1..K
bctx.set_tensor_address("f0", feats.data_ptr())      # 特征 0..K-1
bctx.set_tensor_address("f1", feats.data_ptr() + o4) # 特征 1..K
```

一个 chunk 的 Python 侧只剩 `x.copy_(new_frames)` + `graph.replay()` ——
原来那 110 MB（`i0/i1/f0/f1` 的 `repeat_interleave` + `copy_`）全部消失。
`multi > 2` 时 `B != K`，改为在**图内** `torch.index_select(..., out=buf)` 展开，
同样零 Python 开销。

### 3.3 实测

| 倍率 | v1 一体引擎 | 双引擎（2 次 launch） | **双引擎 + 单图 + 别名** |
|------|-----------:|---------------------:|------------------------:|
| 2x | 8.04 ms | 7.37 | **6.93（−13.2%）** |
| 4x | 12.18 ms | 10.26 | **10.13（−16.8%）** |
| 8x | 14.16 ms | 11.75 | **11.62（−18.0%）** |

> **反直觉的一点**：把 head+body 塞进**同一个 engine**（共享 head 的融合图）
> 反而更慢 —— 实测 7.67 ms（2x），比"两个引擎"的 7.37 慢 4%。
> TRT 对"一个大图"的融合/tactic 选择不如"两个小图"。
> 正确形态是 **两个引擎 + 一张图**，不是"一个引擎 + 一张图"。

数值：与 v1 引擎逐位对比 **PSNR 66.4 dB / max|Δ| 0.033**（`assets` 静帧）、
**52~56 dB**（demo 视频帧，`|Δ|>4/255` 的像素 0.04~0.13%），比 `report.md` 里
TRT 对 fp32 的 51.07 dB 还小 —— 属于 fp16 算子重排的正常水平。

### 3.4 踩过的坑（值得单独记）

1. **`set_tensor_address` 必须给每一个 tensor 都绑**。漏了 `t`/`o` 时 TRT 只在
   stderr 打一行 `Address is not set for input tensor t`，`enqueueV3` 返回 false，
   但**流水线会照样跑完**：输出全黑、编码快 60% —— 看起来像"性能大突破"。
   现在 `_Ctx.run()` 检查返回值并抛异常。
2. **输入 `copy_` 和 `execute_async_v3` 必须在同一条流**（或 `wait_stream`）。
   分开时引擎可能读到半份输入，PSNR 会随机从 66 dB 掉到 24 dB。
   原型阶段这里踩了两次。

---

## 4. 改动 2：NV12 全链路

### 4.1 为什么

v1 的管道是 rgb24：解码器做 YUV420→RGB（swscale，CPU），我们上传 13.8 MB，
再在 GPU 上把 8 个输出帧 RGB→NV12。两个后果：

- **源帧被无谓地转换了两次**（解码器 YUV→RGB、我们 RGB→YUV），chroma 被
  重采样两次，实测与解码器原始 NV12 差 **47.3 dB**（mean 0.79 级）。
- 上传字节数是 NV12 的两倍。

### 4.2 做法

- `FrameReader` 直接 `-pix_fmt nv12`，队列里是 `[h*3/2, w]`。
- 模型输入：Triton 内核 `nv12 -> RGB fp16`（BT.601 limited），写进
  `[K+1,3,PH,PW]` 的常驻缓冲，pad 区只在分配时清零一次。
- 输出：**源帧原样拷贝解码器的 NV12**（零色彩运算、逐位不变）；
  插值帧 luma 由网络输出算，chroma 取插值 RGB 的 2×2 均值。
- `--pix-fmt rgb24`（调试用）在最后多一步 NV12→RGB，不影响主路径。

### 4.3 实测

| 项 | v1 (rgb24) | v2 (nv12) |
|----|-----------:|----------:|
| H2D / chunk | 13.8 MB / 0.70 ms | **6.9 MB / 0.35 ms** |
| 模型输入转换 | 0.08 ms（u8→f16/255） | 0.01 ms（融合内核） |
| 源帧 | YUV→RGB→YUV，47.3 dB | **逐位等于解码器输出** |
| 输出色彩 | 8 帧 RGB→NV12，0.68 ms | 4 帧 luma+chroma，0.04 ms |

色彩矩阵：ffmpeg 对 `yuv420p→rgb24` 默认用 **BT.601 limited**（实测 601 逆变换
对 ffmpeg 的 RGB 有 49.16 dB；换 709 只有 24.30 dB，一眼偏色）。所以逆变换必须
用 601，且与输出端的 601 矩阵自洽。

> **踩过的坑**：一开始把插值帧的 chroma 取成"两个源帧 chroma 的时域平均"，
> 理论上 RGB→YUV 是线性的、应该等价 —— 但插值帧的 RGB 不是两帧的线性混合，
> 而是**运动补偿**后的结果。运动场景下 luma 边缘和时域平均的 chroma 边缘对不齐，
> 码率从 5.97 涨到 8.62 Mbps（+44%）。改回"从插值 RGB 算 chroma"后恢复。

---

## 5. 改动 3：Triton 融合 epilogue

### 5.1 问题

v1 的 epilogue 在图外，33 个 kernel、0.68 ms/chunk：
`×255 → clamp → to(uint8) → crop → cat → stack(u,v) → float → avg_pool →
clamp → to(uint8) → permute → reshape → cat`，其中 `permute+reshape` 还有一次
全尺寸拷贝。

### 5.2 做法

`src/kernels.py` 三个 Triton 内核（无 Triton 时自动退回 torch 实现）：

| 内核 | 工作量 | 实测 |
|------|--------|-----:|
| `nv12 → rgb` | 读 1.5 B/px，写 3×fp16/px | **0.014 ms** |
| `rgb → luma` + `rgb → chroma`（2 个） | 读 3×fp16/px，写 1 B/px + 2 B/4px | **0.025 ms** |
| 源帧拷贝 | `copy_` 到 strided 视图 | ~0.005 ms |

关键点：**3D 网格 `(col_block, row, frame)`**，热循环里没有整除/取模；
`BLOCK=512` 而不是 32768。

> **性能教训**：第一版用 1D 网格 + `i // W` / `i % W` + `BLOCK=32768`，
> 同一个内核要 **0.379 ms**（慢 27 倍）：每个元素两次 64 位整除，外加
> 每 program 128 KB 的 float32 累加器把寄存器打爆。
> 换成"一行一块"后直接到 0.014 ms。

### 5.3 为什么不把色彩转换塞进 engine

可以（ONNX 末尾加 `Mul 255 + Clip + Cast`，TRT 会融进最后一层），但输出要拼
NV12（源帧原样 + 插值帧交织）仍需额外 kernel 或多次 strided copy，
2×2 均值还要 `AveragePool` + 多一路输出。
现在 3 个 kernel 合计 0.04 ms，不值得再改。

---

## 6. 改动 4：编码器参数扫描（结论：没有空间）

同一段真实 NV12、同一条 pipe、8 帧一次 write，1920 帧 × 3 次重复：

| 配置 | fps |
|------|----:|
| **当前**（`p1 ull constqp22 bf0 g600 multipass0`） | **1506 / 1531 / 1485** |
| `-async_depth 2` / `32`（默认 4） | 1530 / 1537 |
| `-tune ll` | 1523 / 1535 / 1527 |
| `-preset p2`、`-rc vbr -cq 22`、`-g 120`、4 帧一次 write、`-pix_fmt yuv420p` | 全部 ±3% 以内 |

**没有一个档位带来 >3% 的差异**（都在重复测量的噪声里）。"换 ffmpeg 参数"
这条路走完了，当前参数已在最优区间。

真正的限制在别处：**编码器在模型满载时从 1500 fps 掉到 520 fps（−65%）**。

| 状态 | NVENC 8 帧 | 有效 fps |
|------|-----------:|---------:|
| 空载（只有编码器） | 5.3 ms | 1500 |
| 模型满载（`--encoder null` 并行） | 15.4 ms | **520（−65%）** |
| v2 流水线内 | 11.1 ms | ~707 |

两边抢的是 **L2/HBM 带宽**：RIFE 720p 是访存型负载（43.7 GFLOP 只跑出 24 TFLOPS，
带宽 ~380 GB/s），NVENC 的帧间预测要读参考帧，排队时间被拉长。
PCIe 不是瓶颈（22 MB/chunk ≈ 1.1 ms @20 GB/s），SM 算力也不是。

**推论**：想再提速只能减少"两边都要吃的资源"——
少编码像素（降分辨率/降倍率）、或少让模型读显存（更小的模型/更少的分辨率）。
换编码器参数、换 C++、加 stream 都无效。

---

## 7. 上游对照：`vs-rife` 与 `vs-mlrt` 到底做了什么

> 结论全部来自**源码核对**（vs-rife `3488617` v5.7.0；vs-mlrt `44033da`，
> `vsmlrt.py` 3.23.2），不是博客转述。行号均指对应文件。

### 7.1 定位差异：帧服务器 ≠ 转码器

| 维度 | `vs-rife` | `vs-mlrt` (`vstrt`) | **本仓库** |
|------|-----------|---------------------|-----------|
| 形态 | VS 滤镜（Python + libtorch） | VS 滤镜（纯 C++ 插件 + TRT） | 独立 CLI 转码器 |
| 调度 | VS 帧图按需拉取，**一次一对帧** | 同左，`fmParallel` | 自建线程 + **K 对帧一批** |
| 后端 | Torch-TensorRT（dynamo→TorchScript） | ONNX→`trtexec`→engine（C++ 反序列化） | ONNX→TRT Python API→engine |
| 并发 | 5 条 stream + 每 stage host sync | `num_streams` 个实例（信号量限流） | 1 compute 流 + 1 copy 流 + 1 张图 |
| 批处理 | 无（B=1） | 无（B=1） | **B=K×n** |
| 主要用途 | 压制/播放（延迟优先） | mpv/SVP 实时播放 | 离线批量转码（吞吐优先） |

跨方案比倍数没有意义（基线不同、B 不同），比机制才有。

### 7.2 `vs-rife` 的真实做法（与流传版本有出入）

1. **后端是 Torch-TensorRT，不是 ONNXRuntime**：`torch.export` →
   `torch_tensorrt.dynamo.compile(use_explicit_typing=True)` → TorchScript →
   `torch.jit.load`（`vsrife/__init__.py:563-598`）。`torch_tensorrt` 是可选依赖。
2. **Head 与 IFNet 是两个独立 engine**（`*.ts` / `*.ts.encode`，`:479-598`），
   4.25 的 `IFNet.forward(..., f0, f1)` 直接把特征当入参（`IFNet_HDv3_v4_25.py:98`）。
   —— 这正是我们在 §3 做的事，区别是我们还把它捕进一张图，且 2x 时零拷贝。
3. **特征缓存是 Python dict，按源帧号索引**（`:627-665`），窗口约 10 帧
   （`cache_to_delete = real_n - 10`）。硬编码帧数、与分辨率无关，4K 下约 1.3 GB，
   seek 跳帧时还会缓慢泄漏。
4. **静态网格只是预计算，没折叠进 engine**：`tenFlow_div`、`backwarp_tenGrid`
   是 **forward 的入参** → engine 的输入 binding，每次前向多搬一份 2 通道
   全分辨率网格（fp32，768p 约 15.7 MB）。本仓库的网格在导出期就折叠成
   `Constant`（ONNX 里 351 个 Constant、无 `Range`）。
5. **warp 强制 FP32**：`warplayer.py:6-12` 把 input/flow 都 `.to(torch.float)`。
   这是修 `4cfcd18` 报的 4K 抖动/破碎，**不是**性能选择。
6. **不是"三流流水线并发"**：确有 5 条 stream，但**每个 stage 结束都 host
   `synchronize()`**（`:705/713/796`），单线程拉帧时串行；没用 CUDA Graph。
   多流只在 VS 多线程同时要不同帧时才有意义 —— 作者后来把 `num_streams` 删了
   （commit `4fe4a4b`；issue #37 里 12 流 115 fps、24 流 125~135 fps，
   25 流以上反而掉到 ~55 fps）。
7. **`t == 0` 原帧零拷贝直通**（`:669-670`），连 D2H 都不做。
8. **pad 到 `modulo=64`**（4.25），零填充、输出裁回（`:435-438`, `:717`）。

### 7.3 `vs-mlrt` 的真实做法

1. **`num_streams` = 实例池，不是线程池**：每实例
   `{cudaStream, IExecutionContext, pinned H2D/D2H, device in/out}`，
   用自实现 `TicketSemaphore` 限流（`vs_tensorrt.cpp:47-70, 506-548`）。
   并发来自 VS 的 `fmParallel` 帧线程。
2. **CUDA Graph 把 H2D + `enqueueV3` + D2H 一起捕获**（`trt_utils.h:193-228`），
   所以指针必须固定、实例内持久分配；**捕获失败不降级**，filter 直接建失败。
   TRT 后端默认 `use_cuda_graph=True`（`vsmlrt.py:143`）。
3. **RIFE v1 = 11 通道**（I0 3 + I1 3 + t 1 + 4 个坐标/缩放 plane），
   **v2 = 7 通道**（内部生成网格）。v1 要求 H/W 是 32 的倍数，1080p 必须补到 1088。
   我们的引擎是 **6+1 通道**，接口上等价于 v2。
4. **没有零拷贝**：始终 `vs_bitblt` → pinned → `cudaMemcpyAsync` → 推理 → 回拷
   （`inference_helper.h:74-124`）。H2D 用 `cudaHostAllocWriteCombined`，
   D2H 用普通 pinned（`trt_utils.h:380-406`）—— 这个细节可以抄，收益 <1%。
5. **fp16 不在插件里**：fp16 是 Python 拼 `trtexec --fp16`（TRT≥11 先转 ONNX fp16）；
   `use_cublas`/`use_cudnn`/`workspace`/`max_tactics` 也都是命令行参数。
6. **fp32 锁层**：`vsmlrt.py:1109-1131` 对 TRT<11 把
   `/Cast_2, /Reciprocal, /Sub_5, /Mul_*` 强制锁 fp32 —— 和 vs-rife 的 FP32 warp
   是同一个坑。
7. **RIFE 输入用 VS 原语拼**：`DuplicateFrames + Trim` 造 I1、
   `BlankClip(color=i/multi)` 造 t、最后 `Interleave`（`vsmlrt.py:1257-1295`）。

### 7.4 三个需要修正的流行说法

| 说法 | 实际情况 | 证据 |
|------|---------|------|
| "vs-rife 用 H2D/计算/D2H 三流实现三级流水" | 有 5 条流，但**每 stage 后 host sync**，单线程下串行 | `__init__.py:705,713,796` |
| "vs-mlrt 的杀手锏是 CUDA Graph 把几百次驱动交互压成一次" | Graph 把 H2D/D2H 也捕进去，主要收益是**省掉每帧 memcpy+launch**；失败不降级 | `trt_utils.h:193-228` |
| "vs-rife 分离 Head 砍掉 50% 以上特征计算" | 分离是真的，但缓存窗口仅 10 帧、**跨 t 不复用**（t=0.25/0.5/0.75 各跑一次 IFNet） | `__init__.py:479-665` |

### 7.5 技术点矩阵

| 技术点 | vs-rife | vs-mlrt | 本仓库 v2 | 判断 |
|--------|:-------:|:-------:|:---------:|------|
| 静态 shape + fp16 TRT 引擎 | ✅ | ✅ | ✅ | 已有 |
| `grid_sample` 原生 TRT 算子 | ✅ | ✅ | ✅（18 个 GridSample） | 已有 |
| CUDA Graph | ❌ | ✅ | ✅（**整块**：head+body） | 领先 |
| Head / IFNet 拆两个 engine | ✅ | ❌ | ✅ | 已有 |
| 零拷贝（head 输出→body 输入） | ❌ | ❌ | ✅（2x 指针别名） | **独有** |
| 批处理 B>1 | ❌ | ❌ | ✅ K×n | **独有** |
| 静态网格折叠进 engine | ❌ | ✅（v2） | ✅ | 已有 |
| 多 stream 并发 | ⚠️（串行） | ✅（实例池） | ❌ | 不抄（批处理无并发帧请求） |
| C++ 重写 | ❌ | ✅ | ❌ | 不抄（`experiments.md` §3 实测更慢） |
| 场景切换直通 | ✅ | ✅ | ❌ | 见 §8（收益取决于片源） |

---

## 8. 剩余空间（按 ROI）

### 8.1 端到端已经贴住编码器

- 模型侧 8.05 ms/chunk（504 对/s），编码器 11.1 ms/chunk（707 fps），
  流水线 11.3 ms/chunk（352 对/s）。
- 编码器空载 1500 fps、满载 520 fps。即使把模型优化到 0，也只能到 ~1500 fps
  （750 对/s），而现在 705 fps —— **理论剩余 +6%**，还要靠减模型访存去换。

### 8.2 还能做的

| 项 | 预期 | 代价 | 结论 |
|----|-----:|------|------|
| **4x/8x 的 head 跨 chunk 复用**（head 只算 K 帧） | 模型 −2%（2x）/ −1%（8x） | 特征环形缓冲 + 相位图 | 收益太小，不做 |
| **继续减模型访存**（grid 生成、`cat` 临时量） | 端到端 +1~3% | 改 engine 输出结构 | 可做，收益有限 |
| **场景切换直通** | 强依赖片源（动画/监控 5~15%，实拍≈0） | scene-change 检测 | 先量再做 |
| **双 NVENC / slice 并行** | 理论 1.4~1.6× | ffmpeg 不支持 slice mode，要手写 C++ + 码流拼接；两进程抢 context（`experiments.md` §5 实测 −40%） | ❌ |
| **零拷贝喂 NVENC**（device→NVENC） | +5%（`experiments.md` §2 旧数据） | 大改，且与 TRT 不叠加 | ❌ |
| **FP8** | 未知 | 算力不是瓶颈、带宽才是；精度风险 | 待验证 |

### 8.3 明确不做

| 方向 | 原因 |
|------|------|
| 多 CUDA stream 流水 | H2D 0.35 ms 已很小，CPU 全被 `bufwait` 掩盖；上限 +1% |
| 纯 C++ 重写 | `experiments.md` §3：C++ 零拷贝 643 fps 比 ffmpeg 管道 938 fps 还慢 |
| `--pad-mult 16`（720p 不 pad） | 端到端 +1.6%，画质 36.7 dB（v1 实测），**否决** |
| channels_last | K=4 时反慢 1% |
| 动态 shape 引擎 | 静态 shape 每层独立选 tactic 更快 |
| 编码器参数 | §6 实测全部在噪声内 |

---

## 9. 回归测试与修复的缺陷

全局回归时又抓出 **7 个问题**（4 个是 v1 就有的，3 个是 v2 引入/放大的），
全部已修：

| # | 位置 | 问题 | 后果 | 修法 |
|---|------|------|------|------|
| 1 | `cli.py` D2H → 下一轮 | 输出是常驻 scratch，D2H 与下一 chunk 的写入无同步 | 真实竞态（实测未暴露） | **双缓冲 `out0/out1`**，D2H 读 n%2、下一 chunk 写 (n+1)%2 |
| 2 | `engine.py:engine_name` | `--scale` / `--mode` 没进引擎文件名 | 换 `--scale 0.5` 会**误用 scale=1 的引擎**（v1 就有） | 非默认值加进名字：`..._s0.5_...` / `..._assemble_...` |
| 3 | `cli.py:ensure_engine` + `export` | `--pad-mult` 没传进 `export()`（内部硬编码 64） | 文件名与实际形状不符，随后校验报错（v1 就有） | `export(..., pad_mult=)` |
| 4 | `cli.py:video_encoder_args` | `--encoder x264` 被忽略（一直走 nvenc） | 文档写了但**功能不存在**（v1 就有） | `encoder == "x264"` 显式覆盖 codec |
| 5 | `cli.py:Runtime.get` | `--no-cudagraph` 对 TRT 无效（`use_graph` 没传） | 开关不起作用（v1 就有） | `TRTNet(*paths, use_graph=args.cudagraph)` |
| 6 | `cli.py:FrameReader` | 解码固定 `nv12`，而 4:2:0 表示不了奇数尺寸 | **奇数尺寸输入静默产出 0 帧**（v2 引入） | 检测奇数 → `pad` 到偶数并提示，输出为填充后的尺寸 |
| 7 | `cli.py` 码率打印 | `dur = max(done/fps, 1e-6)`，`done=0` 时炸 | 1 帧视频打印 `316080 Mbps` | 用实际输出帧数算 |

另外两处是 v2 开发中修掉的（记录在案）：

- `engine.py:_Ctx.run` 不看 `enqueueV3` 返回值 → 漏绑 tensor 时**静默出黑帧**
  （实测时"提速 60%"，其实是模型没跑）。现在抛异常。
- 输入 `copy_` 与 `execute_async_v3` 跨流 → 引擎读到半份输入，PSNR 从 66 dB
  随机掉到 24 dB。现在同流。

### 9.1 回归矩阵（正确性）

| 用例 | 结果 |
|------|------|
| `validate.py`（torch 路径 4 项判据） | ref 逐位 0，assemble 121 dB，pyramid 43 dB，fold 133 dB ✅ |
| TRT vs torch，2x（NV12 chunk 比对） | 源帧**逐位相同**；插值帧 0.034% 像素差 >2 级 ✅ |
| TRT vs torch，4x / 8x（图内 `index_select`） | 源帧逐位相同；插值帧 0.030~0.033% ✅ |
| 720p / 1080p / 2K / 1278×718（非 64 对齐）/ 1279×719（奇数） | 全部正确输出，帧数 = (N−1)×m + 1 ✅ |
| 1 帧 / 2 帧素材 | 1 帧→输出 1 帧；2 帧→3 帧 ✅ |
| 混合分辨率队列（720p→1080p，引擎/缓冲切换） | 两段各自满速 ✅ |
| `--mode assemble` / `--scale 0.5` | 各自建独立引擎并运行 ✅ |
| `--no-cudagraph` / `--encoder x264` / `--pix-fmt rgb24` | 均生效 ✅ |
| 编码器 `null` / `nvenc` / `x264` | 均正常，码率/速度符合预期 ✅ |
| 同一配置连跑两次（确定性） | 输出 **md5 完全相同** ✅ |
| 长跑 3 分钟 / 30000 帧（1080p30 实拍） | 186 对/s，输出 59999 帧正确；VRAM 4594 MiB、RSS 1290 MB **全程不变**（无泄漏）✅ |
| 无 Triton 时的 torch 回退 | nv12→rgb 48.93 dB（Triton 版 49.16 dB），源帧逐位同上 ✅ |

### 9.2 回归矩阵（性能，720p/1080p/2K，2x 除非注明）

| 配置 | TRT | torch |
|------|----:|------:|
| 720p 2x | **352 对/s（705 fps）** | 305（609 fps） |
| 720p 2x，`--encoder null` | **504 对/s** | — |
| 720p 4x | **145 对/s（580 fps）** | — |
| 720p 8x | **68 对/s（542 fps）** | — |
| 1080p 2x | **187 对/s（373 fps）** | 147（293 fps） |
| 2K 2x（2560×1472） | **84 对/s（168 fps）** | — |

### 9.3 仍然存在的（不影响正确性）

| # | 位置 | 问题 |
|---|------|------|
| 1 | `engine.py:engine_name` | 名字里仍没有 TRT 版本 / GPU 名 / 优化级别 → 升级 TRT 或换卡后可能读到不兼容引擎（反序列化失败会直接 `sys.exit`） |
| 2 | `net.py:_base_grid` | fp32 网格 cast 成 fp16（vs-rife `4cfcd18` 同类问题）；720p/1080p 未复现，4K 上线前处理 |
| 3 | `cli.py` | `--scale != 1` 时输出分辨率不变但内部金字塔变粗，画质会下降（文档已注明是 4K 降算力用，未做画质标定） |

## 10. 复现

```bash
# 素材
mkdir -p /tmp/rf && cd /tmp/rf
ffmpeg -y -stream_loop 4 -i demo720_long.mp4 -c copy -an demo720_x5.mp4    # 2400 帧
ffmpeg -y -stream_loop 19 -i demo1080p.mp4   -c copy -an demo1080_x20.mp4  # 1920 帧

# 端到端（默认就走 TRT；首次导出 + 构建 2 个引擎，约 2~4 分钟）
uv run python -m src -i /tmp/rf/demo720_x5.mp4   -m 2 -o /tmp/rf                    # 352 对/s
uv run python -m src -i /tmp/rf/demo720_x5.mp4   -m 2 -o /tmp/rf --no-trt           # 305（对照）
uv run python -m src -i /tmp/rf/demo720_x5.mp4   -m 4 -o /tmp/rf                    # 145
uv run python -m src -i /tmp/rf/demo1080_x20.mp4 -m 2 -o /tmp/rf                    # 183~187

# 模型侧（不带编码器）
uv run python -m src -i /tmp/rf/demo720_x5.mp4 -m 2 -o /tmp/rf --encoder null       # 504

# 编码器空载上限（务必用长样本，短样本低 35%）
cat demo720.nv12 | ffmpeg -v error -f rawvideo -pix_fmt nv12 -s 1280x720 -r 48 \
  -i - -an -c:v h264_nvenc -preset p1 -tune ull -rc constqp -qp 22 -bf 0 \
  -g 600 -no-scenecut 1 -multipass 0 -pix_fmt nv12 -f null -   # ≈1500 fps

# 数值自检
uv run python -m src.validate
```

---

## 11. 附录：v2 关键数据

### A. 模型侧（720p 2x，8 输出帧/chunk，1280×768 fp16）

| 段 | 耗时 |
|----|-----:|
| head 引擎（K+1=5 帧） | 0.79 ms |
| body 引擎（B=4） | 6.58 ms |
| 两者捕进一张图 + 指针别名 | **6.93 ms** |
| nv12→RGB（Triton） | 0.014 ms |
| RGB→NV12（Triton，含源帧拷贝） | 0.025 ms |
| **模型侧合计** | **8.05 ms**（`--encoder null` 496 对/s） |

### B. 与 v1 的逐位差异

| 对比 | PSNR | 说明 |
|------|-----:|------|
| v1 引擎 vs v2 引擎（`assets` 静帧） | 66.4 dB | fp16 重排 |
| v1 引擎 vs v2 引擎（视频帧） | 52~56 dB | 遮挡边缘 max 0.28 |
| 源帧：v1 rgb→nv12 vs v2 解码器 nv12 | 47.3 dB（mean 0.79 级） | **v2 逐位正确** |
| 插值帧 luma：v1（先量化再矩阵）vs v2（一次量化） | mean 0.89 级 | v2 误差更小 |

### C. 编码器（720p，h264_nvenc p1 ull qp22，1920 帧样本）

| 状态 | ms/chunk(8 帧) | fps |
|------|---------------:|----:|
| 空载 | 5.3 | 1500 |
| 模型满载 | 15.4 | 520 |
| v2 流水线内 | 11.1 | ~707 |

### D. 4x / 8x / 1080p（v2）

| 场景 | 对/s | 输出 fps | 说明 |
|------|-----:|---------:|------|
| 720p 4x（K=2） | 145 | 580 | Head 12 次 → 3 次 |
| 720p 8x（K=1） | 68 | 542 | Head 14 次 → 2 次 |
| 1080p 2x（K=4） | 187 | 373 | |
| 2K 2x（K=2，2560×1472） | 84 | 168 | 自动选 K=2 |
| 720p 2x（torch 路径） | 305 | 609 | TRT 比它快 16% |
| 1080p 2x（torch 路径） | 147 | 293 | TRT 比它快 27% |

显存峰值（含桌面 ~0.5 GiB）：720p 2x 约 2.3 GiB，720p 8x 约 3.0 GiB。

> 4x/8x 提升 28% 主要来自 Head：v1 每个 chunk 要跑 12/14 次单帧编码，
> v2 只要 3/2 次。
