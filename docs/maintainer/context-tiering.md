# 上下文两层寄存（host RAM + NVMe）

**状态**：移植中，未实现。这是 active work 的**取回/移植计划与目标契约**，不是已交付能力，
也不是一份从零设计。
**先例**：T1（host RAM 层）是本条线上游的既有能力，被 `7028d96a` 删除；T2（NVMe 层）在同源
3090 线上已实现。取回面已在 §3 量化：18 个文件、14,748 行，其中计价/决策层在本树**已经存在
且与上游逐字节相同**，只是没有调用者。

本文修订 [Paged KV Context Store](paged-kv-cache.md) §1.1 —— 那里把 `swap、KV offload`
列为 non-goal；本计划把它变成一条有明确契约的能力。范围只覆盖 **retained（空闲）上下文的
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

所以本计划的目标可以精确表述为：**把淘汰从"丢弃"改成"寄存"，让 `full_reset` 只发生在
寄存空间本身也耗尽的时候。**

---

## 2. 先例：这两层在别处已经存在

本计划**不是从零发明**。查证结果：T1（host RAM 层）是本条线上游的既有能力，被双卡移植提交
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

- **T1 是恢复，不是设计。** 取回面已量化：18 个文件 / 14,748 行，见 §3。
  其中 688 行（计价与决策框架）本树已有且逐字节相同，只是没有调用者。
  需要重新接回的分歧文件另有约 9k 行 Program 逻辑，见 §3.3。
- **T2 有参考实现可读**：`iamwavecut/ninfer-all` 的 `disk_kv_*`，尤其 slot 头格式、digest
  身份、miss-as-recompute 语义；但它的恢复路径依赖 Windows DirectStorage，Linux/NVMe 上
  要换成自己的后端。
- §4 的位置语义（搬迁而非预留、image 必须含 GDN state、TP2 barrier）与上述两份实现不冲突，
  可作为把两条线已有能力接到本机 TP2 双卡上的落点。

---

## 3. 移植面（量化）

从 12 个种子头文件出发、按 `#include` 依赖做传递闭包，得到下面的取回集合。
数字都是 `upstream/master` 的行数。

### 3.1 必须取回：18 文件 / 14,748 行

| 行数 | 文件 | 角色 |
|---:|---|---|
| 3387 | `src/runtime/engine/resource_manager.h` | 权限/容量/召回/压力决策 |
| 2058 | `src/runtime/engine/engine_core.h` | 引擎内部状态与请求生命周期 |
| 1873 | `src/targets/qwen3_6/impl/runtime/logical_kv_store.h` | KV 逻辑页/物理页两层抽象 |
| 1485 | `src/targets/qwen3_6/impl/runtime/pressure_planner.h` | 寄存/下沉/召回规划 |
| 1199 | `src/runtime/engine/materialization_planner.h` | materialization 搜索 |
| 809 | `src/targets/qwen3_6/impl/runtime/state_image_store.h` | GDN/MTP state 镜像（device+host） |
| 711 | `src/targets/qwen3_6/impl/runtime/host_kv_extent_store.h` | host extent 物理槽 |
| 661 | `src/core/host_kv_arena.cpp` | pinned host KV arena |
| 655 | `src/runtime/engine/context_cost.cpp` | 复用/重建成本 |
| 460 | `src/targets/qwen3_6/impl/state/state_image.cpp` | state 镜像实现 |
| 369 | `src/runtime/engine/scheduler.h` | 调度器（本树已换掉） |
| 278 | `src/core/host_kv_arena.h` | |
| 206 | `src/runtime/engine/request_record.h` | |
| 173 | `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/state_image.h` | |
| 158 | `src/runtime/engine/causal_score_core.h` | |
| 139 | `src/runtime/engine/context_cost.h` | |
| 84 | `src/runtime/engine/context_cost_defaults.cpp` | |
| 43 | `src/targets/qwen3_6/impl/runtime/rebuild_work.h` | |

### 3.2 已经在本树、且与上游逐字节相同：688 行（零改动可用）

| 行数 | 文件 | 本树调用者 |
|---:|---|---|
| 428 | `src/runtime/engine/shared_capture_planner.h` | **0** |
| 138 | `src/runtime/engine/context_portfolio_value.h` | **0** |
| 45 | `src/runtime/engine/resource_search.h` | **0** |
| 77 | `src/targets/qwen3_6/impl/runtime/resource_projection.h` | **0** |

**这是本次移植最重要的杠杆。** 这四个文件（含 `rebuild_ns` /
`baseline_recovery_ns` / `target_recovery_ns` / `private_retention_weight` /
`explicit_shared_credit` 那套按纳秒计价的逻辑）整段活了下来，只是被摘掉了唯一调用者——
它们上游只被 `resource_manager.h` 与 `materialization_planner.h` 引用，而这两个正是被删掉的
文件。**计价逻辑不需要重写。**

### 3.3 被压缩、需要重新接回的分歧文件

| 上游 | 本树 | 文件 |
|---:|---:|---|
| 12526 | 3288 | `src/targets/qwen3_6/impl/runtime/program_impl.h` |
| 1420 | 509 | `src/targets/qwen3_6/impl/runtime/program.h` |
| 1310 | 305 | `src/targets/qwen3_6/impl/runtime/request_plan_impl.h` |
| 637 | 116 | `src/runtime/contract/types.h`（68 → 20 个顶层声明） |
| 600 | 273 | `src/targets/qwen3_6/impl/runtime/api_impl.h` |
| 549 | 484 | `src/runtime/engine/engine.cpp` |
| 377 | 413 | `src/serve/serve_options.cpp`（本树多出 TP2 选项） |
| 372 | 208 | `src/core/paged_kv_cache.h` |

`program_impl.h` 里 `CaptureGroup` / `SharedPrefixHandle` / `StateImageHandle` /
`LogicalKVPageHandle` / `HostKVArena` / `CaptureOffer` 的出现次数在本树**全部为 0**：
复用与寄存逻辑不是被简化，是被整段移除。

### 3.4 结构性硬冲突：pool 架构

上游 `paged_kv_cache.h`（372 行）是**逻辑/物理两层**：
`DeviceKVPagePool` + `DeviceKVPageHandle/Lease/Reservation`（带 generation 的句柄与页运行段）、
`KVExecutionTablePool`（device block table 作为一等资源，有 `publish(...)`）、
`HostKVAllocationView`/`HostKVAllocationConstView`，以及**直接做 D2H/H2D 的
`DeviceKVPagePool::copy_to_host()` / `copy_from_host()`**
（`upstream/master:src/core/paged_kv_cache.cpp:576,617`）。

本树是**单层** `PagedKVPool` + `PagedKVAllocation`，没有句柄 generation、没有执行表池、
也没有 `copy_to_host`/`copy_from_host`；`src/core/paged_kv_cache.cpp` 不再包含
`core/host_kv_arena.h`。

所以取回上游的 host tier 必然牵动所有 KV 调用方。这是本计划唯一需要先决策的点，见 §7 P2。

### 3.5 同时值得取回的验证工具

| 行数 | 文件 | 用途 |
|---:|---|---|
| 3320 | `tests/test_resource_manager.cpp` | Block B 的验收，**用 Fake Package，不需要真存储** |
| 652 | `tests/targets/qwen3_6/test_context_store.cpp` | extent store / 复用契约 |
| 369 | `tests/test_context_cost.cpp` | 成本模型 |
| 218 + 70 | `tests/targets/qwen3_6/test_state_image{,_layout}.cpp` | state 镜像 |
| 1759 | `bench/context_cost/*`（5 文件） | park/restore 与 re-prefill 的对照测量 |
| 1028 | `docs/maintainer/resource-scheduling-and-context-cache.md` | 本节设计的上级权威 |

---

## 4. 为什么可行（承重事实）

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
   `page_ids_`（`src/core/paged_kv_cache.h:185`），而 `publish_mapping()`
   （`src/core/paged_kv_cache.h:170`）经私有的 `publish_range()`
   （`src/core/paged_kv_cache.h:181`）**已经**在 stream 上把 device block table 写出来
   （`src/core/paged_kv_cache.cpp:386-397`，最终是 `cudaMemcpyAsync`）。因此"恢复时落到不同
   的物理页" = 改 host 侧数组 + 刷一次表。attention kernel 读的是表；`decode_graph.cpp`
   约束的是节点**拓扑**和 **device residency**，不是页索引。这条把本来最硬的两个改动点整个
   消掉了——**并且本树保留了这两处原语**，见 §7 P3。

4. **搬运通路已有先例。** `core/arena.cu` 有 `PinnedHostBuffer`；TP2 的 allreduce
   （`src/ops/common/allreduce.cu`）已经有 host-staged 传输回路和跨设备 event choreography。

---

## 5. 移植后的目标形态

这一节描述取回**之后**运行时的形态。责任归属很关键：下面只有最后两行是本树要新做的工作，
其余全部是取回上游已有能力，不重新发明。

| 能力 | 上游 | 取回来源 |
|---|---|---|
| pinned host KV 槽位分配与存活期 | 有 | `src/core/host_kv_arena.{h,cpp}` |
| KV 逻辑页/物理页两层、搬迁而非预留 | 有 | `logical_kv_store.h`；`DeviceKVPagePool::copy_to_host/from_host` |
| GDN/MTP state 的 device+host 镜像 | 有 | `state_image_store.h`、`impl/state/state_image.cpp` |
| 压力驱动的寄存/下沉/召回 | 有 | `pressure_planner.h` + `resource_manager.h` |
| 按纳秒计价的复用决策 | 有（本树已有同字节文件） | `shared_capture_planner.h` + `context_portfolio_value.h` |
| 续接纪律：哪些前缀可恢复 | 有（本树已有同字节文件） | 上游契约 + `PrefixReusePath` |
| **双 rank TP2 分片与 barrier** | **无**（上游单 GPU） | 本树新做，见 §5.7 |
| **T2 NVMe 层** | **无** | 同源 3090 线有参考实现，Linux 后端自建，见 §7 P6 |

### 5.1 层级

| 层 | 载体 | 粒度 | 用途 |
|---|---|---|---|
| T0 | device `PagedKVPool` | page（`P=64` token） | 现状，不改 |
| T1 | pinned host RAM | page slot | 一级寄存 |
| T2 | NVMe 文件 | page extent | 二级寄存，T1 压力下逐出 |

### 5.2 搬迁，不是预留

明确否掉"device page 继续为 owner 占着、只把字节搬到别处"：那样一字节显存都不释放。
采用 **释放 device page → 搬运整个 tier image → 恢复时重新分配页并重映射**。

### 5.3 tier image 的内容

一个被寄存的 checkpoint 必须是**完整可恢复状态**，不能只搬 KV：

- Main Text KV planes（每个 rank 自己那片）
- speculative backend（MTP）的 KV
- **GDN 线性注意力 state** —— 按 `paged-kv-cache.md`，Linear Attention state
  **不能从 KV 前缀单独重建**，只能来自保留了完整 continuation state 的
  target-declared checkpoint
- MTP target hidden state
- checkpoint 元数据：token ids、resume frontier、owner/demand mask、layout 描述

后三类不是 per-token 的，体量小；KV 是主体。

### 5.4 代价模型

每层：`restore_ns(tier) = bytes / 可达带宽(tier) + fixed_latency`。

按实测参数（每设备文本 KV = **16,896 B/token**；PCIe **Gen3 x4** ≈ 3 GB/s；NVMe ≈ 2 GB/s；
prefill ≈ 1,140 tok/s）：

| 出路 | 每 token 成本 | 相对 prefill |
|---|---:|---:|
| rebuild（re-prefill） | ~877 µs | 1× |
| T1 host RAM | ~5.6 µs | **~156× 更快** |
| T2 NVMe | ~8.4 µs | **~104× 更快** |

把这三个值一起喂进 `ContextPortfolioValue` 的 fold，让既有的 threshold 逻辑自己选。

### 5.5 分层准入与逐出

- T1 按 portfolio value 做 LRU；T1 有压力时把价值最低的 image **下沉**到 T2。
- T2 按 LRU 逐出；逐出 = 删除 extent，下次需要时退回 rebuild。
- **只寄存 retained/空闲 checkpoint，绝不寄存 active request 的页。**

### 5.6 拷贝效率

按**合并后的页区间**搬，不要逐页搬：把物理上相邻的页合并成单个 `cudaMemcpyAsync`。
整上下文搬迁应该收敛成少数几段大传输。单页 64 × 16,896 = **1.03 MiB**，已经到了逐页搬会
明显浪费带宽的尺度。

### 5.7 TP2

两个 rank 各搬自己那片（每设备 2 个 KV head）。**必须有一个跨设备 barrier**：两边都搬完，
才允许把释放出来的页交给别的 allocation。block table row 是 per-rank 的，各自本地重写。

### 5.8 异步位置

- **park**：在关键路径之外（请求出完最后一个 token 之后）。
- **restore**：在 claim 路径上，必须与 admission/排队**交叠**，不能串行地挡在 prefill 前面。

---

## 6. Non-goals

- **预取**。恢复就是按需触发。
- **在 live request 内部按页换入换出**。decode 每步扫过整个上下文
  （`gqa_attention_decode_i8.cuh` 的 `key_blocks` 循环 + split-K + online softmax），
  部分驻留的上下文根本不可用，image 必须整体回来才能继续。
- 压缩、去重、跨进程共享、跨 session 共享。
- 提高单请求的上下文上限（那是正交的另一件事）。

---

## 7. 移植计划

按依赖顺序分块取回。每块自带验收门，上一块的门不过不进下一块。

### P0 — 取回评估（本文，已完成）

见 §3。结论：**决策与计价层在本树已经存在且逐字节相同（688 行，零改动可用）**，
真正的取回量集中在物理存储层与 Program 的权能接口。

### P1 — 先让计价与决策层站起来（不碰存储）

只取回不依赖 pool / 真存储的部分，让它们在本树编译并跑通单元测试：

- `context_cost.{h,cpp}` + `context_cost_defaults.cpp`（878 行）：纯成本模型。
- `materialization_planner.h`（1199 行）：依赖本树已有的 `resource_search.h` 与
  `shared_capture_planner.h`。
- `request_record.h`（206 行）。
- `resource_manager.h`（3387 行）：**这一步的难点**。它是 `ResourceManager<Package>`，
  要求 `Package::Program` 提供 `resource_revision()`、`inspect()`、
  `reserve_materialization()`、`progress_context_transaction()`、
  `reserve_active_capture()`、`finish()`、`abort()`、`populate_runtime_stats()`，
  以及 `Package::{ResourcePlan, ActiveCaptureResult, ContextTransactionProgress,
  MaterializationResult}`。本树 `program.h`（509 行）没有这些。
- 依赖契约：`runtime/contract/types.h` 上游 637 行 / 68 个顶层声明，本树 116 行 / 20 个——
  取回时按需补类型，不要整文件覆盖。

**门**：`tests/test_context_cost.cpp`（369 行）与 `tests/test_resource_manager.cpp`
（3320 行）在本树编译并通过。后者用 `FakePreparedPrompt`/`FakeCheckpointSummary`/
`FakeContinuationSummary`/`FakeSharedPrefixSummary` 一套 Fake Package，
**因此不需要真实存储层**——它若通过，说明 Block B 的权能接口定义已经足够精确。

### P2 — 决策：Program 权能接口怎么落（门，必须先选）

上游 `program.h`（1420）/`program_impl.h`（12526）里的 `CaptureGroup`、
`SharedPrefixHandle`、`StateImageHandle`、`LogicalKVPageHandle`、`HostKVArena`、
`CaptureOffer` 在本树 `program_impl.h`（3288）中出现次数**全部为 0**。P1 的权能接口有两种落法：

- **(a) 忠于上游**：连同 `logical_kv_store.h`、`state_image_store.h` 一起取回上游 Program
  结构，把 `program.h`/`program_impl.h` 恢复到上游形态。语义与上游
  `resource-scheduling-and-context-cache.md` 一致，压力/召回逻辑可直接搬；
  代价是要把本树为 TP2 加的 peer 锁步与 `concurrent_executor.h` 重新表达，改动面最大。
- **(b) 保留本树 Program，只补权能**：让本树 Program 实现 P1 需要的 8 个方法，存储层按
  本树单层 `PagedKVPool` 语义实现。改动面小、TP2 结构不动；代价是 `pressure_planner.h`
  的对接要重写，并与上游文档产生语义偏离（文档需同步改）。

**建议先做 (b)**：把 P3 的第一项（`host_kv_arena` 对接）当作探测。选 (b) 的前提是
`pressure_planner.h` 与 `logical_kv_store.h` 的耦合能在不引入上游两层 pool 的前提下表达清楚；
做不到就转 (a)。

### P3 — 物理存储层（Block D）

- `host_kv_arena.{h,cpp}`（939 行）：pinned host KV 槽位。上游版本只依赖
  `paged_kv_cache.h`；本树同一头文件从 372 行缩到 208 行，对接需按单层语义替换
  `HostKVAllocationView`。
- `state_image_store.h`（809）+ `impl/state/state_image.cpp`（460）+
  `export/.../state_image.h`（173）：GDN/MTP state 镜像。**与 pool 架构无关，可独立取回。**
- `host_kv_extent_store.h`（711）、`logical_kv_store.h`（1873）、`rebuild_work.h`（43）、
  `pressure_planner.h`（1485）：与 pool 强耦合，是 P2 选项的实际工作量所在。
- `park`/`restore` 落到本树已有的两处原语：`PagedKVAllocation::page_ids()`
  （host 侧数组，`src/core/paged_kv_cache.h:185`）与 `publish_mapping()`
  （`src/core/paged_kv_cache.h:170` → `src/core/paged_kv_cache.cpp:386-397`）。

**门**：`tests/targets/qwen3_6/test_context_store.cpp`（652）、`test_state_image.cpp`（218）、
`test_state_image_layout.cpp`（70）通过；park → 丢掉 → restore 后
(a) KV 字节与寄存前**逐字节相同**（精确 oracle），(b) 续写与"从不 swap"的对照运行
**逐 token 相同**（greedy、固定 seed）。单上下文、concurrency 1。

### P4 — 引擎壳与调度（Block C）

`engine_core.h`（2058）依赖上游 `scheduler.h`（369），本树用 `concurrent_executor.h`
（1300）+ `request_memory.h`（42）取代。二选一：把 `EngineCore` 接到本树
`concurrent_executor`，或取回上游调度器。`engine.cpp` 549 → 484 的分歧也在这一步收口。

**门**：现有 serve 端到端测试不回退；concurrency 1..8 与 MTP/DFlash2 路径的既有测试全绿。

### P5 — TP2 适配（本树独有工作，无上游可搬）

- 两个 rank 各持一份 arena，各搬自己那 2 个 KV head 的分片。
- **跨设备 barrier**：两边都搬完，才允许把释放出来的页交给别的 allocation。
- block table row 是 per-rank 的，各自本地重写。
- `SequenceKVBundle` 的 `text_peer`/`backend_peer` 锁步必须在 park/restore 时保持。

**门**：两 rank 各自恢复出的分片与寄存前相等；`tools/tp2` parity 通过。

### P6 — T2 NVMe 层

- 参考 `iamwavecut/ninfer-all` 的 `disk_kv_store.{h,cpp}`（4 KiB slot + 48 B 头 + CRC-32
  + 原子替换 `.idx`，自包含无 CUDA）与 `disk_kv_bridge.{h,cpp}`（spill 队列 + digest 身份
  + miss-as-recompute）。
- 恢复后端在 Linux/NVMe 上自建（上游用 Windows DirectStorage）。
- 读失败或校验不符 = **miss**（退回 rebuild），永远不是正确性事件。
- 注意 §2.2 的第三方实测：磁盘层单独用**不打破"池 ≥ 前缀"**，它只能作为 T1 的下沉层。

### P7 — 文档与 CLI

- 取回上游 `docs/maintainer/resource-scheduling-and-context-cache.md`（1028 行）作为本计划的
  上级权威，把本文中与之重复的部分删掉——**一个当前权威，不留平行文档**。
- 修订 `paged-kv-cache.md` §1.1 与 `concurrent-inference-architecture.md`，把 swap 从
  non-goals 里拿掉，写上 tier 契约。
- `docs/performance.md` 加 context-switch 表（park/restore 延迟与字节 vs re-prefill）。
- CLI/serve：先对齐上游 `--device-state-slots`、`--host-state-slots`、`--host-kv-mib`；
  再加 `--context-tier-nvme <path:size>`。同步 `--help` 与 `serve_usage_text` 测试。

### 7.1 取回验证工具

`tests/test_resource_manager.cpp`（3320）、`tests/test_context_cost.cpp`（369）、
`tests/targets/qwen3_6/test_context_store.cpp`（652）、
`tests/targets/qwen3_6/test_state_image{,_layout}.cpp`（288）、
`bench/context_cost/*`（1759）。这些是取回过程中逐块可用的门，不要等到最后才跑。

---

## 8. 验证

| 主张 | 证据 |
|---|---|
| 恢复后行为等价 | 恢复后的续写与不 swap 对照**逐 token 相同** |
| image 往返无损 | park→restore 的 KV 字节逐字节比较 |
| TP2 分片一致 | 两 rank 各自恢复出的分片与寄存前相等；跑 `tools/tp2` parity |
| 失败不污染正确性 | 注入损坏的 NVMe 页 → 必须表现为 miss |
| 收益真实 | 8k / 64k / 200k 下的 park/restore 延迟落在带宽模型内 |

---

## 9. 风险

- **瓶颈是 PCIe Gen3 x4（~3 GB/s），不是 NVMe。** 200k 上下文 ≈ 1.1 GB/设备 →
  单向 ~370 ms。比 re-prefill 快约 200×，但不再是零。
- restore 在 claim 路径上；如果不能与 admission 交叠，就会串行化，收益被吞掉。
- host slot 耗尽时仍然要丢——需要 T1 自己的 LRU。
- 把 swap 从 non-goal 清单里拿掉是**产品契约变更**，文档与测试必须同步移动。
- **最大不确定点是 P2**：若 (b) 失败转 (a)，改动面会从"补 8 个 Program 方法"跳到
  "重接约 9k 行 Program 逻辑"，并连带重做 TP2 peer 锁步。这是本计划唯一需要中途重新决策的门。
- `runtime/contract/types.h` 从 637 行 / 68 个顶层声明缩到 116 行 / 20 个：Block A/B 依赖的
  契约类型在本树被大幅削减，取回时可能比 §3.1 的行数清单放大。
- 取回上游文件时不要"整文件覆盖"分歧文件（`paged_kv_cache.h`、`engine.cpp`、
  `contract/types.h`、`serve_options.cpp`）。本树在这些文件里带着 TP2 的改动，
  覆盖会静默丢掉双卡语义。

---

## 10. 待定

- 共享 prefix 的 checkpoint 是否允许被**另一个** session claim（`explicit_shared_credit`），
  还是只允许 owner？
- restore 期间同时持有 device page 和 host slot，这段瞬时占用要不要计入 admission 容量？
- 只做 retained 路径，还是也允许抢占 value 低的 active request？后者是 preemption 味道的
  变体，本计划**明确推迟**。
- **P2 的 (a)/(b) 选择**：忠于上游 Program 结构，还是保留本树 Program 只补权能接口？
  这是阻塞 P3 的决策，判定依据见 §7 P2。

---

## 11. 本机实测参考

- 每设备文本 KV：16,896 B/token（16 层 full attention × 2 KV head × 256 head_dim ×
  1 B × 2 平面 + FP16 group-64 scale）。
- 每设备 runtime 预留：`760,225,536 + 24,224 × N` 字节（N = token 数，含 MTP KV），
  在 179k–1M 区间与实测吻合到 ~2 KB。
- headless 基线（gdm3 已停、两块卡各 16,145 MiB 空闲）下：
  NVFP4 上限 179,200；NVFP4 + 桌面占用时 180,224 会失败；Swift-1.5 上限 208,896；
  GSQ-RCO 可开满原生 262,144。
