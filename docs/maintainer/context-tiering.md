# 上下文两层寄存（host RAM + NVMe）

**状态**：设计中，未实现。这是 active work 的设计与实施计划，不是已交付能力。
**先例**：T1 是本条线上游的既有能力（被 `7028d96a` 删除），T2 在同源 3090 线上已实现；
见 §2。实施计划因此以"取回/移植"为第一选项，而非重写。

本文修订 [Paged KV Context Store](paged-kv-cache.md) §1.1 —— 那里把 `swap、KV offload`
列为 non-goal；本设计把它变成一条有明确契约的能力。范围只覆盖 **retained（空闲）上下文的
寄存与恢复**，不覆盖 active request 的抢占，也不改变单请求的上下文上限。

---

## 1. 问题

在本机（2 × Tesla V100-SXM2 16GB、TP2、`--kv-dtype int8`、NVFP4/Swift-1.5 profile）实测。

复用本身已经存在，而且几乎免费。池子装得下时，同一 prompt 再次到达：

| 请求 | `reuse=` | prefix 命中 | ttft |
|---|---|---:|---:|
| A（8,174 tok） | `full_reset` | 0 / 8174 | 7,123 ms |
| A 再来 | `restore_turn_checkpoint` | **8170 / 8174** | **43 ms** |

**166×**，没有 re-prefill。

问题出在装不下的时候。池 `--max-context 16384`，A=11,743、B=10,155：

| 请求 | `reuse=` | ttft |
|---|---|---:|
| A（11,743 tok） | `full_reset` | 10,327 ms |
| B（10,155 tok） | `full_reset` | 8,972 ms |
| A 再来（11,743 tok） | **`full_reset`** | **10,326 ms** |

A 被**丢弃**，回来时全额重算。A+B = 21,898 > 16,384，池子装不下，而今天的淘汰策略是
"丢"，不是"寄存"。

所以本设计的目标可以精确表述为：**把淘汰从"丢弃"改成"寄存"，让 `full_reset` 只发生在
寄存空间本身也耗尽的时候。**

---

## 2. 先例：这两层在别处已经存在

本设计**不是从零发明**。查证结果：T1（host RAM 层）是本条线上游的既有能力，被双卡移植提交
删掉了；T2（NVMe 层）在同源的 3090 线上已经实现。

### 2.1 T1 是上游能力，被我们删掉了

`upstream/master`（`geoffwatts/ninfer-v100`，merge-base = `b37d0dd3`，是我们 master 的祖先）
保留完整的 host RAM 层，并且它本身同步自根仓库 `Neroued/ninfer`：

| 能力 | 上游文件 |
|---|---|
| pinned host KV arena | `src/core/host_kv_arena.{h,cpp}` |
| 权限/容量/召回规划 | `src/runtime/engine/resource_manager.h`、`engine_core.h` |
| extent store / 压力规划 | `src/targets/qwen3_6/impl/runtime/host_kv_extent_store.h`、`pressure_planner.h` |
| 契约文档 | `docs/maintainer/resource-scheduling-and-context-cache.md` |
| 端到端测试 | `tests/test_resource_manager.cpp` |

上游 CLI 已暴露 `--device-state-slots N --host-state-slots N --host-kv-mib N`
（`src/serve/serve_options.cpp:76,219`），README 记载默认 *"one Device checkpoint slot,
eight pinned Host State slots, and 8 GiB of pinned Host KV"*。即 T1 的设计、实现、CLI、
测试、文档在上游是齐的。

删除发生在 `7028d96a` "feat: add dual V100 NVLink inference"（1011 files，
+107,985 / −196,967），同一提交还删掉了 `apps/perplexity/`、`bench/context_cost/`、
`bench/fixtures/ttft/`。

**fork 普查**：`geoffwatts/ninfer-v100` 的 31 个 fork 里 30 个仍原样保留
`src/core/host_kv_arena.{h,cpp}` 与 `resource-scheduling-and-context-cache.md`；
**只有 `plus1998/NInfer-V100-Duo`（我们）这一支删掉了**。V100 线上没有别人重做过，
因为不需要——上游就有。

V100 线上唯一在推进缓存准入的是 `mylordmonkeyman/ninfer-v100`（`host_kv_arena.cpp` 有改动），
方向是 active cache admission / batched decode cache hit，不是 tier 寄存。

### 2.2 T2 在同源 3090 线上已实现

根仓库 `Neroued/ninfer` 本身**没有**磁盘层，它由 `Don-Chad/ninfer-3090` 这条线下方的
`iamwavecut/ninfer-all` 加上：

| 提交 | 作者 | 内容 |
|---|---|---|
| `f39133e1` (2026-09-24) | Valeriy Selitskiy | `feat(core): add a disk page store and bridge for an L3 KV tier` |
| `546ed9a6` (2026-09-24) | Valeriy Selitskiy | `feat(disk-tier): restore pages through DirectStorage on Windows` |

文件：`src/core/disk_kv_store.{h,cpp}` + `src/core/disk_kv_bridge.{h,cpp}`，同一支还有
`src/runtime/engine/context_cache/{hybrid_resource_manager,materialization_planner}.h`。
`disk_kv_bridge.h` 注释原文：

> Engine-facing facade over the disk (L3) page stores: one store per KV family, a bounded queue
> of spill jobs drained by writer threads, and synchronous reads for restores. … the engine
> keeps a rolling 128-bit digest per token frontier … the same content in two sessions dedupes
> to one page. … A miss of any kind (never spilled, evicted, corrupt) makes the engine recompute.

`disk_kv_store.h` 是 4 KiB slot + 48 B 头（magic / identity / CRC-32 / LRU 戳）加原子替换
`.idx` 的 LRU 页存储，**自包含、不依赖 CUDA 和引擎类型**。同源的 Windows 分发
`BenGamliel/NInferEZ-Engine`、`ahnafnafee/ninfer-3080` 带同样四个文件。

**但它单独用不足以覆盖本问题。** 同族第三方评估 `1314521gjy/ninfer-fusion-kvmem`
（非 fork 的独立仓库，2026-09-30 决策评估）实测这条 `--disk-kv-*` 路线：`cache 0 (0.0%)`、
**全量重填**，且 `restore_chain` 要求 `page_count ≤ mapped_pages`——磁盘层单独用不打破
"池 ≥ 前缀"。该评估的结论是结构上只有"KV 权威副本放 tier、设备只保有界工作集、按差量搬运"
成立。

### 2.3 对本计划的影响

- **T1 是恢复，不是设计。** 优先评估从 `upstream/master` 取回那套子系统，而不是照下面的
  设计重写。规模在 4 万行量级（`engine_core.h` + `resource_manager.h` +
  `pressure_planner.h` + `host_kv_arena.*` + `host_kv_extent_store.h` + serve/CLI 接线），
  且 `src/targets/qwen3_6/impl/runtime/` 在两支间已经不同步。
- **T2 有参考实现可读**：`iamwavecut/ninfer-all` 的 `disk_kv_*`，尤其 slot 头格式、digest
  身份、miss-as-recompute 语义；但它的恢复路径依赖 Windows DirectStorage，Linux/NVMe 上
  要换成自己的后端。
- §4 的位置语义（搬迁而非预留、image 必须含 GDN state、TP2 barrier）与上述两份实现不冲突，
  可作为把两条线已有能力接到本机 TP2 双卡上的落点。

---

## 3. 为什么可行（承重事实）

这四点决定了实现的形状，逐条都在当前代码里可验证。

1. **复用单元已经存在。** `PrefixReusePath` 已是
   `{FullReset, AppendAtFrontier, RestoreTurnCheckpoint, RestoreResponseCheckpoint}`
   （`include/ninfer/types.h:423`）。而且契约明确：*"Reuse resumes a retained frontier or
   complete turn/response checkpoint; an arbitrary matching token prefix is not a reusable
   checkpoint."*（`src/serve/serve_options.cpp:150`）。也就是说"哪些前缀是可恢复的"这件事
   已经定义好了，缺的只是它的**字节离开 device 之后住在哪**。

2. **决策框架已经存在，而且按纳秒计价。** `SharedCapturePlanner`
   （`src/runtime/engine/shared_capture_planner.h`）配合 `ContextPortfolioValue`
   （`src/runtime/engine/context_portfolio_value.h`）已经在算 `rebuild_ns`、
   `baseline_recovery_ns`、`target_recovery_ns`，并用 `private_retention_weight`、
   `explicit_shared_credit`、`demand_mask` 折出净值。
   **今天只有两个出路：{留在显存} 或 {丢掉，付 `rebuild_ns`}。** 加一层存储就是**加第三个
   出路**，其恢复成本是 `bytes / 带宽` 而不是 `rebuild_ns`；`candidate_rebuild_ns` 本来
   就是入参。

3. **页搬迁不需要改 kernel，也不需要改 CUDA graph。** `PagedKVAllocation` 持有 host 侧的
   `page_ids_`，而 `publish_mapping()`/`publish_range()` **已经**在 stream 上把 device
   block table 写出来（`src/core/paged_kv_cache.h:168-180`）。因此"恢复时落到不同的物理页"
   = 改 host 侧数组 + 刷一次表。attention kernel 读的是表；`decode_graph.cpp` 约束的是节点
   **拓扑**和 **device residency**，不是页索引。这条把本来最硬的两个改动点整个消掉了。

4. **搬运通路已有先例。** `core/arena.cu` 有 `PinnedHostBuffer`；TP2 的 allreduce
   （`src/ops/common/allreduce.cu`）已经有 host-staged 传输回路和跨设备 event choreography。

---

## 4. 设计

### 4.1 层级

| 层 | 载体 | 粒度 | 用途 |
|---|---|---|---|
| T0 | device `PagedKVPool` | page（`P=64` token） | 现状，不改 |
| T1 | pinned host RAM | page slot | 一级寄存 |
| T2 | NVMe 文件 | page extent | 二级寄存，T1 压力下逐出 |

### 4.2 搬迁，不是预留

明确否掉"device page 继续为 owner 占着、只把字节搬到别处"：那样一字节显存都不释放。
采用 **释放 device page → 搬运整个 tier image → 恢复时重新分配页并重映射**。

### 4.3 tier image 的内容

一个被寄存的 checkpoint 必须是**完整可恢复状态**，不能只搬 KV：

- Main Text KV planes（每个 rank 自己那片）
- speculative backend（MTP）的 KV
- **GDN 线性注意力 state** —— 按 `paged-kv-cache.md`，Linear Attention state
  **不能从 KV 前缀单独重建**，只能来自保留了完整 continuation state 的
  target-declared checkpoint
- MTP target hidden state
- checkpoint 元数据：token ids、resume frontier、owner/demand mask、layout 描述

后三类不是 per-token 的，体量小；KV 是主体。

### 4.4 代价模型

每层：`restore_ns(tier) = bytes / 可达带宽(tier) + fixed_latency`。

按实测参数（每设备文本 KV = **16,896 B/token**；PCIe **Gen3 x4** ≈ 3 GB/s；NVMe ≈ 2 GB/s；
prefill ≈ 1,140 tok/s）：

| 出路 | 每 token 成本 | 相对 prefill |
|---|---:|---:|
| rebuild（re-prefill） | ~877 µs | 1× |
| T1 host RAM | ~5.6 µs | **~156× 更快** |
| T2 NVMe | ~8.4 µs | **~104× 更快** |

把这三个值一起喂进 `ContextPortfolioValue` 的 fold，让既有的 threshold 逻辑自己选。

### 4.5 分层准入与逐出

- T1 按 portfolio value 做 LRU；T1 有压力时把价值最低的 image **下沉**到 T2。
- T2 按 LRU 逐出；逐出 = 删除 extent，下次需要时退回 rebuild。
- **只寄存 retained/空闲 checkpoint，绝不寄存 active request 的页。**

### 4.6 拷贝效率

按**合并后的页区间**搬，不要逐页搬：把物理上相邻的页合并成单个 `cudaMemcpyAsync`。
整上下文搬迁应该收敛成少数几段大传输。单页 64 × 16,896 = **1.03 MiB**，已经到了逐页搬会
明显浪费带宽的尺度。

### 4.7 TP2

两个 rank 各搬自己那片（每设备 2 个 KV head）。**必须有一个跨设备 barrier**：两边都搬完，
才允许把释放出来的页交给别的 allocation。block table row 是 per-rank 的，各自本地重写。

### 4.8 异步位置

- **park**：在关键路径之外（请求出完最后一个 token 之后）。
- **restore**：在 claim 路径上，必须与 admission/排队**交叠**，不能串行地挡在 prefill 前面。

---

## 5. Non-goals

- **预取**。恢复就是按需触发。
- **在 live request 内部按页换入换出**。decode 每步扫过整个上下文
  （`gqa_attention_decode_i8.cuh` 的 `key_blocks` 循环 + split-K + online softmax），
  部分驻留的上下文根本不可用，image 必须整体回来才能继续。
- 压缩、去重、跨进程共享、跨 session 共享。
- 提高单请求的上下文上限（那是正交的另一件事）。

---

## 6. 实施计划

### P0 — 先做取回评估，再决定是否重写

- 对照 `upstream/master` 清点被 `7028d96a` 删除的 T1 子系统在本树的缺口：`host_kv_arena.*`、
  `resource_manager.h`、`engine_core.h`、`pressure_planner.h`、`host_kv_extent_store.h`，
  以及 serve/CLI 接线（`--device-state-slots`、`--host-state-slots`、`--host-kv-mib`）。
  产出：可取回 / 需改写 / 缺失三类清单，以及 `src/targets/qwen3_6/impl/runtime/` 的差异面。
- 若取回代价可控，P1 及以后改为适配取回实现；否则按下面从零路径执行。
- 从零路径：T1 store（pinned slot 分配器，slot 粒度 = page）。
- `park(image)`：KV 区间的合并 D2H + GDN/MTP state + 元数据。
- `restore(image)`：预留页 → H2D → 重写 `page_ids_` → `publish_mapping()`。
- **测试（这一阶段的门）**：park → 丢掉 → restore 之后
  (a) 恢复出来的 KV 字节与寄存前**逐字节相同**（精确 oracle）；
  (b) 后续生成与"从不 swap"的对照运行**逐 token 相同**（greedy、固定 seed）。
  单上下文、concurrency 1。

### P1 — 接入 planner 与 serve

- 在 `SharedCapturePlanner` 里加第三个出路，带 per-tier 的 `restore_ns`。
- park 策略：admission 失败时，寄存价值最低的 retained checkpoint，而不是丢弃。
- serve：`RestoreTurnCheckpoint` 命中一个已寄存条目时，在 resume 之前发起异步恢复。
- 可观测性：capacity 日志加 per-tier 驻留；`request_log` 的 reuse 记录带上 tier。

### P2 — NVMe 层

- 文件后端 store；T1→T2 下沉；T2 LRU。
- 失败处理：读失败或校验不符 = **miss**（退回 rebuild），永远不是正确性事件。
- 参考实现：`iamwavecut/ninfer-all` 的 `disk_kv_store.{h,cpp}`（slot 头 + CRC + 原子 `.idx`，
  自包含无 CUDA）与 `disk_kv_bridge.{h,cpp}`（spill 队列 + digest 身份）。其恢复走 Windows
  DirectStorage，Linux 侧需自建后端，但存储格式与 miss 语义可直接沿用。

### P3 — 准入与文档

- `docs/performance.md` 加 context-switch 表（park/restore 延迟与字节 vs re-prefill）。
- 修订 `paged-kv-cache.md` §1.1 与 `concurrent-inference-architecture.md`，把 swap 从
  non-goals 里拿掉，写上 tier 契约。
- CLI/serve 选项：`--context-tier-ram <size>`、`--context-tier-nvme <path:size>`，同步
  `--help`。

---

## 7. 验证

| 主张 | 证据 |
|---|---|
| 恢复后行为等价 | 恢复后的续写与不 swap 对照**逐 token 相同** |
| image 往返无损 | park→restore 的 KV 字节逐字节比较 |
| TP2 分片一致 | 两 rank 各自恢复出的分片与寄存前相等；跑 `tools/tp2` parity |
| 失败不污染正确性 | 注入损坏的 NVMe 页 → 必须表现为 miss |
| 收益真实 | 8k / 64k / 200k 下的 park/restore 延迟落在带宽模型内 |

---

## 8. 风险

- **瓶颈是 PCIe Gen3 x4（~3 GB/s），不是 NVMe。** 200k 上下文 ≈ 1.1 GB/设备 →
  单向 ~370 ms。比 re-prefill 快约 200×，但不再是零。
- restore 在 claim 路径上；如果不能与 admission 交叠，就会串行化，收益被吞掉。
- host slot 耗尽时仍然要丢——需要 T1 自己的 LRU。
- 把 swap 从 non-goal 清单里拿掉是**产品契约变更**，文档与测试必须同步移动。

---

## 9. 待定

- 共享 prefix 的 checkpoint 是否允许被**另一个** session claim（`explicit_shared_credit`），
  还是只允许 owner？
- restore 期间同时持有 device page 和 host slot，这段瞬时占用要不要计入 admission 容量？
- 只做 retained 路径，还是也允许抢占 value 低的 active request？后者是 preemption 味道的
  变体，本设计**明确推迟**。

---

## 10. 本机实测参考

- 每设备文本 KV：16,896 B/token（16 层 full attention × 2 KV head × 256 head_dim ×
  1 B × 2 平面 + FP16 group-64 scale）。
- 每设备 runtime 预留：`760,225,536 + 24,224 × N` 字节（N = token 数，含 MTP KV），
  在 179k–1M 区间与实测吻合到 ~2 KB。
- headless 基线（gdm3 已停、两块卡各 16,145 MiB 空闲）下：
  NVFP4 上限 179,200；NVFP4 + 桌面占用时 180,224 会失败；Swift-1.5 上限 208,896；
  GSQ-RCO 可开满原生 262,144。
