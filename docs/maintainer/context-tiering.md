# 上下文两层寄存（host RAM + NVMe）

**状态**：**已落地并验证**（retained 上下文的寄存与恢复）。本文是这项能力的当前权威：契约、
落点、实测证据、验证门，以及明确未做的部分。§3–§7 保留最初的取回/移植推演，用来解释为什么
落点是现在这样；其中**没有执行**的清单已在 §7.8/§7.9 与开头注明，不要当成计划执行。

**范围**：只覆盖 **retained（空闲）上下文的寄存与恢复**，不改变单请求的上下文上限，
也不覆盖 active request 的抢占。位置语义是**搬迁**而不是预留：寄存后 device 页被释放，
恢复时写到新的物理页，所以 image 从不依赖原页号。

本文修订 [Paged KV Context Store](paged-kv-cache.md) §1.1 与
[小规模并发推理架构](concurrent-inference-architecture.md) §1.2 —— 那里把 `swap、KV offload`
列为 non-goal，现在它是一条有明确契约的能力。

## 0. 现在的能力（已落地）

### 0.1 形状

淘汰不再等于丢弃。一条 retained lane 被淘汰时按三层寄存器处理：

| 层 | 载体 | 进入 | 回到 device |
|---|---|---|---|
| L1 device | paged KV pool | — | — |
| L2 host RAM（pinned） | `HostKVArena` + `HostLinearStateArena` | `evict_retained_lane` → `park_lane` | `restore_lane` |
| L3 NVMe | `DiskKVBridge`：内容寻址 64-token 块 + state image | `spill_device_prefix` / `spill_parked_prefix` | `restore_lane_from_disk` |

一次寄存搬运的是**整个可继续状态**：两个 rank 的 Main Text KV、MTP KV、两个 rank 的 GDN 线性
注意力 state、hidden 行，以及 ledger/prefix 元数据。GDN state 不能从 KV 前缀重建，所以 image
必须包含它，这也是 image 远大于 KV 本身的原因。

**逐出完整性（本轮落地）**：任何一条 retained lane 在 device 页被回收前，字节一定先落到下一层。

- 一条 retained lane 在 **request 完成时**就把前缀 `spill_device_prefix` 到下层
  （`publish_retained_prefix`）。否则"盘上有记录"就取决于逐出顺序：进程退出时手上正握着的那条
  会话会一条记录都没有，重开后**恰好是最后在用的那条**要重付 prefill。发布是增量的：
  `SequenceState::disk_published_frontier` 记着已发布 frontier，每个 tap 只从它的上一块开始走，
  所以一条会话的第一轮搬整条前缀，之后每轮只搬新增的块 + 新 frontier 的 state image；
  `spill_pages` 对已在库里的块还有一层 `contains` 跳过。
- admission pass 只寄存**别的** retained lane；请求自己要用的那条 lane，由
  `start_prefill_lane` 在被覆写之前 `spill_device_prefix` 出去。
- `park_lane` 成功后就地 `spill_parked_prefix`，所以 L2 里的每条 lane 在 L3 上都已有记录。
- `evict_retained_lane` 在 L2 装不下时，先 `spill_device_prefix` 再 `clear_lane`。
- 磁盘命中失败（读失败、校验失败、链不完整）一律退化为 recompute，不是错误结果。
- `forward-only`：`disk_kv_path` 为空时，以上路径全部是空操作，行为与加这一层之前逐字节相同。

### 0.2 磁盘身份

- **键**：`DiskKVIdentity{lo,hi,tag,frontier}`。`{lo,hi}` 是 `PrefixDigests` 在 `frontier` 处的
  前缀摘要——**每 token 一条滚动摘要**，是前 `frontier` 个 token 及其 token_type、position 轴、
  以及在该 frontier 折入的 rewrite checkpoint / Vision item 的纯函数。
- **块**：KV 按 64-token 块存，块的键取**块末**的摘要；链的前缀纯度因此可以直接比对。
- **tag**：`identity_tag(backend, proposal_head, storage)`；TP2 下 rank r 用 `tag | (r << 24)`，
  因为每个 rank 持有同一批 token 的不同分片。一条 frontier 只有**所有 rank 的分片都在**才可恢复。
- **state image**：按 `frontier` 寻址（不分 rank），是唯一让恢复合法的 commit——链可以先写，
  只有 image 落地才发布这个 frontier。

### 0.3 哪两个 frontier 可恢复

一条 lane 换出时写**两个 tap**：

1. **prompt frontier**（就是 rewrite checkpoint 的 frontier）：整条 prompt 结束时捕获的 state。
   同一个 prompt 再来、或后续 turn 把它当前缀，都落在这一点上。
2. **channel end**（`execution_frontier`）：本条 lane 最后提交 token 之后的状态。下一个 turn 的
   prompt 原样包含上一次生成的响应（`preserve_thinking=true` 的 ResponseReplay），所以它也落在
   这一点上。

两个 tap 覆盖的正是"同一段内容的下一次请求"。其余 frontier 没有 state image，也就不可恢复——
对 GDN 模型这是硬约束：没有 state 的 KV 前缀单独无法使用。

### 0.4 实测（2 × Tesla V100-SXM2 16GB、TP2、27B NVFP4、MTP3、INT8-G64 KV）

| 量 | 结果 |
|---|---|
| host KV park（132 MiB） | 42.0 ms（3.14 GB/s，PCIe Gen3 上限附近） |
| host KV restore（132 MiB ×2） | 82.8 ms（3.19 GB/s） |
| GDN linear state park（146.8 MiB） | 46.7 ms（3.14 GB/s） |
| host tier 端到端（1943-token 前缀、2 lane 池、`host_context_bytes=256 MiB`） | 单 lane image 379,142,144 B；三种 arena 全部计入 RSS（+891 MiB）；恢复后的续写与"从不 swap"对照**逐 token 相同**，`computed_prefill_tokens` 增量 **0** |
| NVMe tier 端到端（同场景 + `disk_kv_path`） | 两次逐出落盘 763,723,776 B；**销毁 engine**（host arena、device 页、parked image、resident 元数据全部消失）后重开同一目录，两条前缀各自从 NVMe 恢复：**0 个 prefill token**、`reused_prompt_tokens` = prompt frontier、续写与冷跑**逐 token 相同**，TTFT 2.92 s / 5.91 s 冷 → **0.124 s / 0.193 s** 从盘上回来（冷跑的对照值在并发 prefill 下取得，所以这个倍数偏保守） |
| **多会话池端到端**（8 × 150,022 token 会话、池 153,600 token、`disk_kv_bytes = 96 GiB`、引擎销毁后重开） | 每条被逐出的会话在盘上占 5,462 MiB，8 条共 43,697 MiB（42.7 GiB）；销毁 engine 并重开后盘上占用不变；**8 条全部 0 个 prefill token**、`reused_prompt_tokens` = 150,022 = prompt frontier、续写与各自的冷跑**逐 token 相同**、`disk_tier_restores` = 8；TTFT 207.5–227.7 s 冷 → **6.0–8.0 s** 从盘上恢复（30–37×）。全程 `host_tier_parked_bytes` = 0（见 §0.5） |
| **串行服务 profile 端到端**（真 HTTP endpoint：`ninfer-serve --max-context 200000 --kv-capacity 200000 --max-concurrency 1 --host-kv-mib 2048 --disk-kv-path <dir> --disk-kv-mib 98304`，tp2 / MTP3 / INT8-G64） | 8 条 194,950 token 会话各冷 prefill 一次：TTFT 298.7–327.8 s（595–653 tok/s）。**进程完整重启两次**后重发同样 8 条：`cache=194,948/194,950`（frontier 差 2 token 的 append）、`reuse=append_frontier`、**0 prefill token**、TTFT **5.2–6.2 s**（≈50× 冷启动）；重启后启动行 `live=55.34 GiB` 即 8 条记录全部由新进程重新载入。全程 `host-tier parked=0` |

### 0.5 目标形态：10 × 150K 的算术

本机 150K lane ≈ 2.51 GiB INT8 KV/device（16.5 KiB/token/device），加 state image 后约
2.6 GiB/device。

| 层 | 预算（本机实测） | 装得下的 150K lane |
|---|---|---|
| L1 device | 池开 153,600 token（= 一条 150K lane + 余量）时 device 已用 15.1 / 16.4 GB，只剩 ~1.2 GB | **1**（第二条 150K lane 还要 +2.6 GB/device） |
| L2 host RAM | `host_context_bytes`（每条 parked image ≈ KV 5.4 GiB + state + 隐藏行） | ≤ `max_concurrency`（= 2）；本机实测**未用上**，见下 |
| L3 NVMe | `disk_kv_bytes`，Main family 占 65% | 96 GiB → 62.4 GiB / 每条 4.72 GiB ≈ **13**；默认 64 GiB → **8** |

每条 150K 会话在 L3 上的实际成本（实测推算）：Main KV 4.72 GiB（双 rank × 2344 页 ×
1,081,344 B）+ MTP KV 0.29 GiB + state image 2 × 154 MB ≈ **5.33 GiB**（实测 5,462 MiB，
多出的部分是两个 tap 各写一张 state image）。

所以"10 个 150K 对话不 re-prefill"在容量上可达：device 装 1 条，其余在 host/NVMe 上轮转，
回到前台时从下层搬回；只有 L2 与 L3 都装不下的才会真的 recompute。代价是延迟不是正确性——
PCIe Gen3 x4（~3 GB/s）单向搬 2.6 GiB 约 0.9 s，而这条 prompt 的冷 prefill 在 V100 上是分钟级。

为什么实测里 L2 一直是空的：本机池只装得下 1 条 150K lane，新会话因此被**直接**准入到同一条
lane（`can_admit_lane` 把该 lane 自己的常驻前缀算作可回收），`start_prefill_lane` 在替换它之前
把前缀直接 spill 到 L3 —— 从来不需要 park，也就没有 host image。L2 起作用的前提是池能同时装下
两条，需要去 evict **别的** lane（`evict_retained_lane`）。

串行服务 profile（`--max-concurrency 1`）把这个原因变成结构性事实：被替换的 lane 永远就是
被准入的那条 lane，所以前缀只会经 `start_prefill_lane` → `spill_device_prefix` 直接落到 L3，
L2 一次都用不上。`--host-kv-mib` 在这种部署里只承担"打开 NVMe 层"的作用（`enable_disk_tier`
要求 host 层已开），容量给多小都不影响 0-re-prefill，见 `docs/serving.md`。

两条注意：

- **`max_concurrency` 仍是 8**（产品语义未变）：同时 active 的请求还是 8 条，10 条 150K 是
  "10 段会话轮流活动"，不是"10 条同时 decode"。同时驻留的 150K lane 数受 device 显存限制，
  本机是 1（实测：池 153,600 token 时 device 只剩 ~1.2 GB，而一条 150K lane 要 2.6 GB）。
- **L2 的实际占用不止 `host_context_bytes`**：它只约束 KV arena。线性 state arena 按
  `lane 数 × rank 数 × 每个 slot 的布局` 定容（27B、TP2、8 lane、两个 slot ≈ 2.4 GB），
  staging 也是常驻的。给 `host_context_bytes` 定预算时要把这两块加进去。

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

> **本节的定位**：这是最初写下的取回清单，用来量化工作量、判断可行性。实际落地**没有**按它
> 逐文件取回（见 §0 与 §7.8）：`engine_core.h`、`logical_kv_store.h`、`host_kv_extent_store.h`、
> `pressure_planner.h`、`state_image_store.h` 都没有进本树，对应能力改由本树已有的
> `paged_kv_cache` / `host_kv_*` / `disk_kv_*` 原语提供。这一节保留下来只是为了解释当时的判断。

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

所以取回上游的 host tier 必然牵动所有 KV 调用方。这是本计划唯一需要先决策的点，见 §7.4。

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
   消掉了——**并且本树保留了这两处原语**，见 §7.5。

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
| **T2 NVMe 层** | **无** | 同源 3090 线有参考实现，Linux 后端自建，见 §7.8 |

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

按依赖顺序分块取回。每块自带验收门；上一块的门不过，不进下一块。

### 7.1 依赖图与总览

```text
Block A  计价与决策（零引擎依赖，纯 host）
  src/core/transfer_work.h (17)                     新增
  include/ninfer/types.h                           补类型（不覆盖）
  src/runtime/contract/types.h                     补类型（不覆盖）
  context_cost.{h,cpp} + context_cost_defaults.cpp  (878)
  materialization_planner.h                         (1199)
  ── 已在本树：shared_capture_planner.h / context_portfolio_value.h /
     resource_search.h / resource_projection.h      (688，零改动)
                        ↓
Block B  资源管理器（模板，只要 Package 接口）
  resource_manager.h (3387) + request_record.h (206)
                        ↓
Block D  物理存储（需要 §7.4 的 P2 决策）
  host_kv_arena.{h,cpp} (939)                       独立
  state_image_store.h (809) + impl/state/state_image.cpp (460)
    + export/.../state_image.h (173)                独立
  host_kv_extent_store.h (711)                      硬依赖上游两层 pool
  logical_kv_store.h (1873)                         硬依赖上游两层 pool
  pressure_planner.h (1485) + rebuild_work.h (43)   依赖上面两者
                        ↓
Block C  引擎壳（engine_core.h 2058 + engine.cpp）← Block B + 调度器决策
```

| 阶段 | 内容 | 取回行数 | 门 | 估计 |
|---|---|---:|---|---|
| P1 | Block A + Block B | 2077 + 17 | `ninfer_resource_manager_test` | 3–5 人日 |
| P2 | Program 权能接口决策 | — | 决策记录 + 小规模探测 | 1–2 人日 |
| P3 | Block D | 6493 | `test_context_store` / `test_state_image*` / `test_context_cost` + park→restore 精确性 | 8–15 人日 |
| P4 | Block C | 2585 | 现有 serve 端到端不回退 | 4–8 人日 |
| P5 | TP2 分片与 barrier | 新代码 | `tools/tp2` parity + 两 rank 分片相等 | 3–6 人日 |
| P6 | T2 NVMe | 参考实现 | 注入损坏 → 表现为 miss | 5–10 人日 |
| P7 | 文档与 CLI | 1028 | `--help` / `serve_usage_text` 测试 | 2–3 人日 |

**序贯关键路径是 P1 → P2 → P3**。P1/P2 不碰 CUDA、不碰存储、不需要 GPU，
是本计划里唯一能"先完整拿到确定性再往下走"的部分。

P2 已落定走 (b)（见 §7.4）：Block D 不取回 `logical_kv_store.h`/`host_kv_extent_store.h` 的上游
类型；`pressure_planner.h` 与存储无关（判定 2），其纯数据决策类型随 Program 权能一并加入，
执行侧换成本树原语。`host_kv_arena.{h,cpp}` 与 `core/kv_page_geometry.h` 已就位，P3 的取回
行数相应低于上表。

### 7.2 接口规格（P1/P2 的验收依据）

#### 7.2.1 `Package` 必须提供的 25 个 typedef

取自上游 `tests/test_resource_manager.cpp:1731` 的 `FakePackage`——那是这套接口的
最小可编译定义，可直接当作本树 Program 的对齐清单：

```text
Program                 PreparedPrompt          RequestBasePlan
AdmissionCandidate      ResourcePlan            PersistentBackfillProof
SequenceHandle          ContinuationHandle      SharedPrefixHandle
CaptureOffer            ContinuationSummary     SharedPrefixSummary
CaptureAssessment       CapturePressurePlan     ActiveCaptureResult
ContextTransactionProgress                      MaterializationResult
StartResult             FinishResult            AbortResult
PressureTargetHandle    AssessedPressureTarget  CommitResult
DiscardResult           CacheSessionKey
```

#### 7.2.2 `Program` 必须提供的 22 个方法

按 `resource_manager.h` / `materialization_planner.h` / `shared_capture_planner.h` /
`engine_core.h` 里对 `program.<m>()` 的调用取并集：

```text
abort                              begin_capture_pressure_planning
begin_pressure_planning            checkpoint_recovery_work   (2 重载：Continuation/SharedPrefix)
finalize_context_transaction       finish
has_context_transaction            inspect_admission
inspect_capture                    isolated_request_feasible
physical_usage                     progress_context_transaction
prove_persistent_backfill          release_continuation
reserve_active_capture             reserve_active_capture_with_pressure
resource_revision                  seal_identity
shared_capture_matches             shared_capture_split_prefill_work
skip_capture                       start_resource_transaction
```

签名以 `upstream/master:src/targets/qwen3_6/impl/runtime/program.h:520-620` 为准
（`plan_request` / `causal_score` / `commit` / `decode` 等执行侧方法已在两树共有，
不在本清单内）。

#### 7.2.3 `ResourceManager` 对外暴露的 15 个方法

`engine_core.h` 对 `resources_.<m>()` 的调用，即 Block B→C 的接口面：

```text
abort            adopt                  apply_commit        apply_discard
clear_after_program_cleanup           context_transaction_kind
finish           inspect                lane_state          populate_runtime_stats
progress_context_transaction          prove_persistent_backfill
release_failed_commit                 reserve_active_capture  reserve_materialization
```

### 7.3 P1 — 计价与决策层（Block A + Block B）

#### 取回清单（全部来自 `upstream/master`）

| 行数 | 文件 | 动作 |
|---:|---|---|
| 17 | `src/core/transfer_work.h` | 新增，无依赖 |
| 139 | `src/runtime/engine/context_cost.h` | 新增 |
| 655 | `src/runtime/engine/context_cost.cpp` | 新增 |
| 84 | `src/runtime/engine/context_cost_defaults.cpp` | 新增 |
| 1199 | `src/runtime/engine/materialization_planner.h` | 新增 |
| 3387 | `src/runtime/engine/resource_manager.h` | 新增 |
| 206 | `src/runtime/engine/request_record.h` | 新增 |
| 369 | `tests/test_resource_manager.cpp` | 新增 |
| — | `include/ninfer/types.h` | **补类型，不覆盖** |
| — | `src/runtime/contract/types.h` | **补类型，不覆盖** |
| — | `tests/CMakeLists.txt` | 加 1 条 `ninfer_add_test`（上游第 70 行） |

#### 本树适配点

1. **`include/ninfer/types.h`（上游 965 → 本树 550）。** 需要补的公开类型包括
   `ContextCacheOptions`、`ContextCacheHints`、`ContextCostOptions`、`ContextCostSummary`、
   `ContextCostPresetSource`、`CacheRetentionHint`、`PromptCacheMarker` /
   `PromptCacheMarkerKind` / `PromptCacheMarkerLocation`、`PromptContinuationMode`、
   `MaterializationDiagnostics`、`MaterializationStopReason`、`SharedCandidateEvidence`、
   `RuntimeHostWorkStats`。**逐个搬，不要整文件覆盖**——本树在该头文件里带着 TP2/DFlash2 的改动。
2. **`src/runtime/contract/types.h`（上游 637 行 / 68 个顶层声明 → 本树 116 行 / 20 个）。**
   同上，按 `materialization_planner.h` 与 `resource_manager.h` 的编译错误逐个补。
3. **不要在 P1 取回 `context_cost.cpp` 之外的任何存储代码。** `context_cost.h` 只依赖
   `core/transfer_work.h`、`ninfer/types.h`、`runtime/contract/types.h`；`materialization_planner.h`
   只依赖 `context_cost.h`、`context_portfolio_value.h`、`resource_search.h`——后者三份本树已有。
4. **`test_context_cost.cpp` 不在本阶段。** 它 `#include "core/host_kv_arena.h"`，
   属于 Block D，归 §7.5 的门。

#### 门

`test_resource_manager.cpp` 只 `#include "runtime/engine/resource_manager.h"` 加标准库，
自带 `FakePackage`，**不碰 CUDA、不碰 GPU、不碰真存储**：这是本计划最便宜也最硬的一块验证。
它若能在这棵树里编译通过并全绿，说明 §7.2 的权能接口定义已经足够精确。

```bash
cmake -S . -B build-v100-test -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DCMAKE_CUDA_FLAGS="-isystem /usr/local/cuda-12.8/targets/x86_64-linux/include" \
      -DBUILD_TESTING=ON -DNINFER_BUILD_APPS=OFF
cmake --build build-v100-test --target ninfer_resource_manager_test -j
LD_LIBRARY_PATH=build/_deps/install/lib ./build-v100-test/tests/ninfer_resource_manager_test
```

注意 `BUILD_TESTING` 默认 `OFF`，现有 `build-v100-duo/` 与 `tools/v100/build.sh`
都显式关掉它，需要单独一个测试构建目录；`LD_LIBRARY_PATH` 是 FFmpeg 的传递依赖
（`libavformat` 拉 `libswresample`，rpath 不传递）。`-DCMAKE_CUDA_FLAGS` 里的
`-isystem` 是 glibc ≥ 2.41 主机的**必要条件**：CMake 的 CUDA 编译器探测在项目
`-isystem` 生效之前运行，会撞上 glibc C23 `cospi`/`sinpi` 与 CUDA 12.8
`crt/math_functions.h` 的声明冲突；补上同一个 `-isystem` 后探测与真实 TU 一致，详见
[V100 移植说明](../v100.md)。

#### 已落地（本轮实测）

- `ninfer_resource_manager_test` 全绿，`tests/test_resource_manager.cpp` 未改一行。
- `ninfer` / `ninfer-serve` 在 `build-v100-duo` 全量重建（161 个 CUDA TU）并链接成功，
  `ninja` 报 no work to do；`test_request_log`、`test_concurrent_executor` 一并复验通过。
- 8 个取回文件与 `upstream/master` **逐字节相同**（`diff` 输出为空）。
- 类型补齐：`nvtx::Name` 增 `ProgramSubmit`/`DeviceWait`/`ProgramPost`；`PrefixReusePath`
  同时容纳本树的 realized 词表（`FullReset`…）与上游 planner 词表（`Root`…`SharedStablePrefix`）。
  两套词表的统一是 §7.4 P2 的决策，本阶段不预判映射。
- 类型补齐的实际规模：`include/ninfer/types.h` +490 行、`src/runtime/contract/types.h` +529 行。
  其中有 24 个顶层声明只被 `src/product/logging/*`、`src/serve/operational_log.cpp` 这类
  **未参与任何构建目标**的同树文件引用，或完全无引用；保留它们是为了与上游 `types.h` 同步，
  不是 P1 的必需项，清理与否留待 P2 处理。
- 回归修复：`runtime/contract/types.h` 现在传递包含 `core/nvtx.h`，而 `ninfer_v100_corpus`
  只链 `ninfer_engine`、没有 CUDA include 目录，因此编译失败；已按 `tests/CMakeLists.txt`
  的做法补 `${CUDAToolkit_INCLUDE_DIRS}`（`bench/CMakeLists.txt`）。

#### 完成判据与风险

- 判据：`ninfer_resource_manager_test` 全绿，且 `ninfer` / `ninfer-serve` 仍能构建。
- 风险：`types.h` 与 `contract/types.h` 的类型补齐可能连带牵出更多引用，
  实际取回行数会超过 2077。这是本阶段唯一的不确定点。

### 7.4 P2 — 决策门：Program 权能接口怎么落

上游 `program.h`（1420）/`program_impl.h`（12526）里的 `CaptureGroup`、`SharedPrefixHandle`、
`StateImageHandle`、`LogicalKVPageHandle`、`HostKVArena`、`CaptureOffer` 在本树
`program_impl.h`（3288）中出现次数**全部为 0**。所以 §7.2.2 的 22 个方法与 25 个 typedef
有两种落法：

- **(a) 忠于上游**：连同 `logical_kv_store.h`、`state_image_store.h` 一起取回上游 Program
  结构，把 `program.h` 恢复到上游形态（1420 行）、`program_impl.h` 补回约 9k 行。
  语义与上游 `resource-scheduling-and-context-cache.md` 一致，压力/召回逻辑可直接搬；
  代价是要把本树为 TP2 加的 peer 锁步、`concurrent_executor.h`、`tp2_*` 重新表达。
- **(b) 保留本树 Program，只补权能**：让本树 Program 实现那 22 个方法，存储层按本树单层
  `PagedKVPool` 语义实现。改动面小、TP2 结构不动；代价是 `pressure_planner.h` 与
  `logical_kv_store.h` 的对接要重写，并与上游文档产生语义偏离。

#### 判定依据（用 P3 的第一项当探测）

先做 `host_kv_arena` 的对接探测，观察三件事：

1. `HostKVExtentStore` 的接口能否在不引入上游两层 pool 的前提下表达——
   它现在吃 `const LogicalKVPageStore&`（`page_layout()`）并返回
   `std::vector<DeviceKVPageHandle>`（`host_kv_extent_store.h:83,144`），**这是硬依赖**。
2. `pressure_planner.h` 的 `PressureDecision` 是否只依赖 extent 抽象，还是深入到
   `DeviceKVPageHandle` 的 generation 语义。
3. 本树 `PagedKVAllocation::page_ids()` + `publish_mapping()` 能否承担
   `DeviceKVPageHandle` 在搬迁中的角色（本树无 generation，页身份是 `int32` 索引）。

三条都成立 → 走 (b)；任一条不成立 → 转 (a)。

#### 决策（P2 已落定）：走 (b)

**结论：保留本树 Program，只补权能（(b)）。** 三条判定依据全部成立；第 1 条只在"必须自己
写适配层"的意义上成立，不构成"必须取回上游两层 pool"的结构性依赖。

| 判定 | 结论 | 证据 |
|---|---|---|
| 1. `HostKVExtentStore` 能否脱离两层 pool 表达 | 成立（需新适配层） | 它对 store 的依赖只有 16 个方法/数据面：`physical_pool().geometry()`、`valid`、`device_resident`、`host_resident`、`host_replica`、`content_epoch`、`committed_columns`、`address_references`、`source_pins`、`can_pin_source`、`pin_source`/`unpin_source`、`can_attach_host_replica`/`attach_host_replica`/`detach_host_replica`、`physical`。全部是可放在 `PagedKVAllocation` 之上的元数据，`physical(page)` 的句柄退化成 `page_ids()` 的 `int32` 物理页号。**物理层已实证**：`core/host_kv_arena.{h,cpp}`（939 行，除 `.h` 的 `#include` 一行外与上游逐字节相同）落在本树 |
| 2. `PressureDecision` 是否只依赖 extent 抽象 | 成立 | `pressure_planner.h` 只 `#include "program.h"` + 标准库，全文不引用 `LogicalKVPageStore`/`DeviceKVPageHandle`；`PressureDecision` 是 `PressureKVDecision{begin_page,page_count,kind}` 的页区间、`PressureStateDecision` 与转移需求，按页数/字节计价。其中的 generation 是 planner 自己的 session/slot generation，与页句柄无关 |
| 3. `page_ids()`+`publish_mapping()` 能否承担搬迁角色 | 成立（用所有权替代 generation） | `ninfer_context_tier_probe_test` 实证：保留中的 allocation 的页不会被别的 allocation 复用（所有权即 pin）；释放后的物理页号会被回收，所以 image 必须按内容而非页号寻址、restore 必须重新 materialize；按物理相邻页合并的 D2H park + H2D restore 逐字节相等；`publish_mapping()` 把恢复后的页号写回执行表 |

**探测（本轮实测）**

- `ninfer_host_kv_arena_test` 全绿：几何桥、`plan_host_kv_page_layout`、
  `plan_host_kv_transfer_work`/`plan_device_kv_copy_work`（page-major/head-major 合并计数）、
  arena allocate/split/`plan_after_releases`/`apply_recipe`。**不碰 CUDA kernel、不碰 device
  pool**——上游 T1 的物理层确实是"恢复"，不是"设计"。
- `ninfer_context_tier_probe_test` 全绿（真实 `PagedKVPool`）：保留页不复用、park/restore
  逐字节、执行表回写、释放页号回收。

**附带修正**

- `src/core/paged_kv_storage.h` 原先与 `upstream/master` 逐字节相同，但**在本树不可编译**：
  它 `switch` 到 `KvCacheStorage::Fp8E4M3Row256/Nvfp4Group16/Fp8KeyNvfp4Value`，而本树
  `KvCacheStorage` 只有 `BFloat16/Int8Group64`（V100 只有两种 KV 存储，见 `include/ninfer/types.h`）。
  已删掉三个不可达分支；它现在可编译，并被探测用于从 `PagedKVStorageLayout` 推导平面表。
- 几何桥落在 `src/core/kv_page_geometry.h`：`KVPlaneGeometry = PagedKVPlaneSpec`（复用池的
  平面类型，不再定义第二份），`KVPageGeometry{page_tokens, device_plane_order, planes}`，以及
  `kv_page_geometry(PagedKVPoolSpec)` 投影。这是 §7.5.1 "`KVPageGeometry` ← `PagedKVStorageLayout`"
  的精确形式——`PagedKVStorageLayout` 是**单层**的 K/V 向量 schema，`KVPageGeometry` 是**多层
  平面表**，两者是推导关系而非同一类型。

**对 (b) 的范围含义**

P3 不再取回 `logical_kv_store.h`（1873）/`host_kv_extent_store.h`（711）的**上游类型**；
`pressure_planner.h`（1485）本身不引用任何存储类型（判定 2），但它的决策类型
（`PressureDecision`/`PressureStateDecision`/`PressureKVDecision`/`ResourceCandidateState`）定义在
上游 `program.h`、本树尚无，需作为**纯数据**随 Program 权能一并加入——它们不牵动存储。需要
新写的原语（P3 的落点）：

- 逻辑页身份：`(PagedKVAllocation*, page index)` + `epoch` + `committed_columns`；
- 设备副本生命周期：`drop_device_replica` 不能把页还给池（否则丢身份），需要"保留但非
  device-resident"的所有权记录；
- host extent：`HostKVArena` 已就位，extent 描述符表照 `HostKVExtentStore` 的
  `prepare`/`device_sources`/`publish`/`abort`/`view` 形状写，`DeviceKVPageHandle` 换成 `int32` 页号；
- 合并 D2H/H2D：按物理相邻页合并成少数 `cudaMemcpyAsync`（`publish_mapping` 已有 H2D 路径，
  D2H 与区间合并是新增）；
- **TP2 不动**：本树 `program_impl.h` 的 peer 锁步、`text_peer` 保持原样。这是 (b) 相对 (a)
  的主要收益，也是选 (b) 的直接理由。

**次决策（P1 遗留）**

- **`PrefixReusePath` 两套词表**：保留单一枚举，含义对应如下，planning 侧由
  `materialization_planner` 写、realized 侧由本树写，`request_log` 同时渲染两族；不新增适配
  函数，因为两条路径各自只写自己那族（`RequestPlanSummary` 用 planner 侧，`BeginSummary`/
  `GenerationMetrics` 用 realized 侧）。

  | planner（planning） | engine（realized） |
  |---|---|
  | `Root` | `FullReset` |
  | `PrivateEndpoint` | `AppendAtFrontier` |
  | `PrivateTurnClosure` | `RestoreTurnCheckpoint` |
  | `PrivateResponseReplay` | `RestoreResponseCheckpoint` |
  | `PrivateLongAnchor` | （无对应；long-anchor checkpoint 本树尚未捕获） |
  | `SharedStablePrefix` | （无对应；shared prefix 属 P4+） |

- **P1 补齐的 24 个无引用顶层声明**：保留。它们来自上游 `types.h`/`contract/types.h`，是
  Block C/D 的契约面；当前只被本树未参与任何构建目标的 `src/product/logging/*`、
  `src/serve/operational_log.cpp` 引用或完全无引用。消费者集合要到 P4 才定型，现在删除会制造
  往返；P4 收口时仍未引用者一并删除。

### 7.5 P3 — 物理存储层（Block D）

#### 已落地（P3 首块，本轮实测）

`src/core/host_kv_offload.{h,cpp}`：`HostKVOffload` 把一个 `PagedKVAllocation` 的页搬到
`HostKVArena`（pinned host RAM）再搬回新页。page-major，按"逻辑与物理都相邻"的段做
`cudaMemcpy2D`；恢复写进目标 allocation 自己的物理页、由调用方 `publish_mapping`。它不释放
源页、也不依赖源页号——image 按内容寻址。

`tests/test_host_kv_offload.cpp`（`ninfer_host_kv_offload_test`）按 27B/tp2/int8-group64 真实
文本 KV 几何（16 层 × 4 平面 = 64 平面，1.03 MiB/页/设备）实测 128 页（132 MiB）：

| 量 | 结果 |
|---|---|
| park D2H 132 MiB | 42.0 ms（3.14 GB/s） |
| restore H2D 132 MiB ×2 | 82.8 ms（3.19 GB/s） |
| host RSS | 112.8 → 245.1 MiB（+132 MiB，正好一份 KV） |
| 目标指针 | `cudaMemoryTypeHost`（pinned，不是普通堆） |
| 设备页回收 | 释放后 `free_pages()==512` |
| 正确性 | 同一 image 恢复进**两个不同**物理页集合，逐字节相同 |

带宽落在 §9 的 PCIe Gen3 模型内。测到的是 KV 层 primitives；把 engine 的 retained eviction
接上它（还要镜像 GDN state / hidden / ledger）见下一块。

#### 已落地（P3 第二块：GDN state + engine 逐出路径，本轮实测）

上一块只搬 KV；而 §5.3 明确要求 tier image 必须包含**不能从 KV 前缀重建**的 GDN
线性注意力 state。这一块把它补齐并接到了真实的 eviction 路径上。

**新原语**

- `src/core/host_linear_state.{h,cpp}`：`plan_host_linear_state_layout(spec, slots)` 把卷积态与
  递归态排成"region → layer → slot"的紧凑镜像；`HostLinearStateArena` 在构造时一次性分配
  全部 pinned 字节（逐出路径上零 CUDA 分配），`park_linear_state` 每个 (slot, region) 发一条
  `cudaMemcpy2DAsync`（host pitch = 每层紧凑字节数，device pitch 取 slot 记录跨度）并在返回前
  同步；`restore_linear_state{,_to}` 按**内容**寻址，允许 image 落到与寄存时不同的 slot。
- `src/targets/qwen3_6/impl/runtime/host_lane_state_store.{h,cpp}`：一个 `HostLaneStateStore`
  持有一个 KV arena、每个布局一个 `HostKVOffload`、一个按 lane 数定容的
  `HostLinearStateArena`，以及 rank 0 两条 hidden 行（tail / rewrite checkpoint）的 pinned 镜像。

**engine 接线**

- `EngineOptions::host_context_bytes`（默认 0 = 关闭）；`MemorySummary` 增
  `host_tier_capacity_bytes` / `host_tier_parked_bytes`。
- `ProgramImplCore::evict_retained_lane` 先尝试 `park_lane`（整个 tier image：两 rank 的
  Text KV + MTP KV、两 rank 的 GDN state、hidden、ledger/prefix 元数据），失败才退回原来的
  `clear_lane`；`start_prefill_lane` 的复用分支按 `FullReset`/其余 走
  `drop_parked_lane`/`restore_lane`；`clear_lane` 丢弃寄存镜像。
- 寄存后 `sequence.retained = true` 且 prefix 元数据保留，planner 仍把它计价为可复用前缀；
  `concurrent_executor` 在 tier 开启时让"严格更长复用"的逐出候选胜过直接准入（等长仍优先
  直接准入，所以未开 tier 的路径逐字节不变）。

**实测**

`tests/test_host_linear_state.cpp`（`ninfer_host_linear_state_test`，48 层，卷积 [5120×3]
BF16 + 递归 [128×128×24] FP32，2 slot = 146.8 MiB/rank）：

| 量 | 结果 |
|---|---|
| park D2H 146.8 MiB | 46.7 ms（3.14 GB/s，PCIe 上限） |
| 目标指针 | `cudaMemoryTypeHost` |
| 正确性 | 同一 image 逐字节恢复到 slot {0,2}，并再次迁移恢复到 {1,3} |
| arena | 寄存即计费、`release` 即归还 |

`tests/targets/qwen3_6_27b/test_engine_host_tier_real.cpp`
（`ninfer_qwen3_6_27b_host_tier_real_test`，27B/NVFP4，tp2 + MTP，`max_concurrency=2`，
64 页池，`host_context_bytes=256 MiB`）：两个互不相关、约 1943 token 的长前缀并发占满两条
lane，再提交一条约 2746 token 的第三前缀把它挤出池子。

| 量 | 结果 |
|---|---|
| 单 lane 镜像 | 379,142,144 B（361.6 MiB）= 两 rank KV（~64 MiB）+ 两 rank GDN state（~294 MiB）|
| RSS 增量 | +891 MiB（pinned 三块 arena 全部计入 RSS） |
| 恢复 oracle | 恢复后的续写与"从不 swap"的对照运行**逐 token 相同**，复用路径与 frontier 相同，`computed_prefill_tokens` 增量为 **0** |
| GDN 被搬运 | 镜像 >200 MiB，KV 单独只有 ~64 MiB |
| 归还 | 第二轮 park/restore 后 parked 字节**精确回到**第一轮的值（镜像被 arena 归还并复用） |

发现并修掉的接线 bug：`restore` 路径先是自己 `bind_row`、随后 `start_prefill_lane` 又
`bind_sequence_kv`，触发"already bound"。按 `HostKVOffload::restore` 的契约（"调用方负责重
发布映射，本方法既不 bind 也不同步"）删掉前者即可。

#### 已落地（P3 第三块：逐出完整性 + NVMe 端到端，本轮实测）

上一块把 lane 镜像落到 host RAM，但"淘汰"本身还有两个洞：admission pass 只寄存**别的**
retained lane（请求自己要用的那条被直接丢掉），而 host arena 满时整条 lane 也只是被
`clear_lane`。这一块把两处都补成"先落到下一层再释放"，并把 NVMe 层接到同一条路径上。

要点、落点与实测见 §0（尤其 §0.1 的逐出完整性、§0.4 的 NVMe 端到端表、§7.8 的接线说明）。

**仍未做**：把 tier 成本喂进 admission/`ContextPortfolioValue` 的取舍（§7.1），
host arena 自身的回收策略（§9），以及 CLI/serve 的开关（§7.9）。

#### 取回清单

| 行数 | 文件 | 备注 |
|---:|---|---|
| 939 | `src/core/host_kv_arena.{h,cpp}` | 接口见 §7.5.1 |
| 809 | `src/targets/qwen3_6/impl/runtime/state_image_store.h` | |
| 460 | `src/targets/qwen3_6/impl/state/state_image.cpp` | |
| 173 | `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/state_image.h` | 本树 export 目录唯一缺的文件 |
| 711 | `src/targets/qwen3_6/impl/runtime/host_kv_extent_store.h` | 依赖上游两层 pool |
| 1873 | `src/targets/qwen3_6/impl/runtime/logical_kv_store.h` | 依赖上游两层 pool |
| 1485 | `src/targets/qwen3_6/impl/runtime/pressure_planner.h` | |
| 43 | `src/targets/qwen3_6/impl/runtime/rebuild_work.h` | |
| 1759 | `bench/context_cost/*`（5 文件） | 对照测量 |
| 1309 | `tests/test_context_cost.cpp` + `test_context_store.cpp` + `test_state_image{,_layout}.cpp` | |

#### 7.5.1 独立可取回的子块

**`host_kv_arena`（939 行）** 只依赖 `core/paged_kv_cache.h` 与 `core/transfer_work.h`。
它自己的接口是自洽的：`plan_host_kv_page_layout(KVPageGeometry)` →
`HostKVPageLayout`、`HostKVArena(capacity, supported_layouts)`、
`allocate(layout, pages)` → `optional<HostKVAllocation>`、
`recipe(...)` + `apply_recipe(...)` 的原子多目标分配，以及
`HostKVAllocationHandle`/`View`/`ConstView` 的 generation 句柄。
**唯一的适配点是 `KVPageGeometry`**——它来自上游 `paged_kv_cache.h`（本树该头文件已缩到 208 行）。
本树按 §7.4 的 P2 决策在 `src/core/kv_page_geometry.h` 提供它：`KVPlaneGeometry` 复用池的
`PagedKVPlaneSpec`，`KVPageGeometry` 是多层平面表，`kv_page_geometry(PagedKVPoolSpec)` 做投影，
而平面表本身由 `paged_kv_storage_layout`（`src/core/paged_kv_storage.h`）推导。该头文件原先
与上游逐字节相同却在本树不可编译（引用了本树没有的三个 `KvCacheStorage` 值），P2 已修掉；
`core/host_kv_arena.{h,cpp}` 也已按原样落在本树。

**`state_image`（809 + 460 + 173）** 与 pool 架构无关，可独立取回。
`StateImageStore(StateImageDevicePool&, HostStatePool*, logical_capacity)`，
提供 `reserve_destination()` / `reserve_logical_destination()` / `reserve_reset()` /
`activate_reset()` 与 device/host 占用统计。这一块直接对应 §5.3 里
"GDN state 不能从 KV 前缀重建"的那条约束。

#### 7.5.2 `park` / `restore` 的落点

落到本树已有的两处原语上，不需要改 kernel 或 CUDA graph（见 §4 第 3 条）：

- `PagedKVAllocation::page_ids()` — host 侧页数组，`src/core/paged_kv_cache.h:185`
- `PagedKVAllocation::publish_mapping()` — `src/core/paged_kv_cache.h:170`，
  经 `publish_range()`（:181）到 `cudaMemcpyAsync`（`src/core/paged_kv_cache.cpp:386-397`）

`park` = 合并页区间的 D2H（按物理相邻页合并成少数几段大传输，见 §5.6）+ GDN/MTP state
+ 元数据；`restore` = 预留页 → H2D → 重写 `page_ids_` → `publish_mapping()`。

#### 门

```bash
cmake --build build-v100-test --target \
  ninfer_context_cost_test ninfer_qwen3_6_context_store_test \
  ninfer_qwen3_6_state_image_test ninfer_qwen3_6_state_image_layout_test -j
ctest --test-dir build-v100-test -R 'context_cost|context_store|state_image' --output-on-failure
```

外加两条本阶段独有的行为门（新写测试）：

1. park → 丢掉 → restore 之后，恢复出的 KV 字节与寄存前**逐字节相同**（精确 oracle）。
2. 后续续写与"从不 swap"的对照运行**逐 token 相同**（greedy、固定 seed）。
   单上下文、concurrency 1。

后两条是本计划真正的正确性门；前面的单测只证明各子块自洽。

### 7.6 P4 — 引擎壳与调度（Block C）

| 上游 | 本树 | 文件 |
|---:|---:|---|
| 2058 | — | `src/runtime/engine/engine_core.h` 需取回 |
| 158 | — | `src/runtime/engine/causal_score_core.h` 需取回（`engine.cpp` 直接 include） |
| 369 | — | `src/runtime/engine/scheduler.h` 上游版 |
| — | 1300 | `src/runtime/engine/concurrent_executor.h` 本树替代 |
| — | 42 | `src/runtime/engine/request_memory.h` 本树新增 |
| 549 | 484 | `src/runtime/engine/engine.cpp` 需收口 |
| 377 | 413 | `src/serve/serve_options.cpp` 需收口 |

`engine_core.h` 的依赖是 `request_record.h` + `resource_manager.h`（都来自 P1）+ `scheduler.h`。
二选一：把 `EngineCore` 接到本树 `concurrent_executor`，或取回上游调度器。
本树 `concurrent_executor.h` 的注释写明是 "for every backend" 的固定容量调度与批量 decode，
它不是 TP2 专属件——**优先接到它**，否则会丢掉本树的并发改动。

门：现有 serve 端到端测试不回退；concurrency 1..8、MTP、DFlash2 的既有测试全绿。

### 7.7 P5 — TP2 适配（本树独有，无上游可搬）

- 两个 rank 各持一份 arena，各搬自己那 2 个 KV head 的分片。
- **跨设备 barrier**：两边都搬完，才允许把释放出来的页交给别的 allocation。
- block table row 是 per-rank 的，各自本地重写。
- `SequenceKVBundle` 的 `text_peer`/`backend_peer` 锁步（`program.h` 里有注释说明两池
  页几何相同、操作同序）必须在 park/restore 时保持。

门：两 rank 各自恢复出的分片与寄存前相等；`tools/tp2` parity 通过。

### 7.8 P6 — T2 NVMe 层

- 参考实现：`iamwavecut/ninfer-all` 的 `disk_kv_store.{h,cpp}`（4 KiB slot + 48 B 头
  = magic/identity/CRC-32/LRU 戳，原子替换 `.idx`，自包含无 CUDA）与
  `disk_kv_bridge.{h,cpp}`（有界 spill 队列 + writer 线程 + digest 身份 + miss-as-recompute）。
- 读失败或校验不符 = **miss**（退回 rebuild），永远不是正确性事件。
- 注意 §2.2 的第三方实测：磁盘层单独用**不打破"池 ≥ 前缀"**，它只能作为 T1 的下沉层。

门：注入损坏的 NVMe 页 → 必须表现为 miss 而不是错误结果。

#### 已落地（本轮实测）：L3 页存储整层照抄

`src/core/disk_kv_{store,bridge}.{h,cpp}` + `direct_storage_reader.{h,cpp}` 与
`tests/test_disk_kv_{store,bridge}.cpp` 取自 `iamwavecut/ninfer-all` 的 `a9155e0b`，
**逐字节未改**（含测试），已在 `build-v100-test` 编过、两条测试全绿。

**修正本文原先的判断**：不需要"在 Linux 上自建恢复后端"。`disk_kv_store.cpp` 的 POSIX 分支
本来就在（`open` / `ftruncate` 稀疏文件 / `mmap` / `pwrite` / `pread`），DirectStorage 只在
`_WIN32 && NINFER_DIRECTSTORAGE` 下编译，非 Windows 的 `open()` 直接抛异常且
`available_in_build()` 为 false，bridge 只在 `options.direct_storage` 为真时调用它 —— 所以
Linux 上留 `false` 即可，代码零改动。

**还缺什么**（P6 只剩引擎接线这一块）：

**身份层已落地（本轮）**。上游 key 是 `DiskKVIdentity{lo,hi,tag,frontier}`，其中
`{lo,hi} = prefix_digests.at(frontier)`：**每个 token frontier 一条滚动的 128-bit 内容摘要**
（上游 `ProgramImpl::disk_identity`，`tag = capture_identity_tag()`）。**修正本计划原先的
误判**：上游另有一条 64-token 分块链 `runtime::prefix_cache::block_hash`
（`h(b) = xxh3_64(h(b-1) || tokens || extra)`，命中还要精确比 tokens），那是 Hybrid 常驻
索引的判据，**不是**磁盘 key。本树已落：

- `src/targets/qwen3_6/impl/runtime/prefix_digests.{h,cpp}`：`PrefixDigests`，与上游
  `PrefixShortlistDigests` 同算术、同域分隔、同顺序（rewrite checkpoint 在它关闭的
  frontier 折入，Vision item 在最后一个 span 末尾折入），只吃 `PreparedPromptData`。
- `.../impl/runtime/identity_tag.h`：`identity_tag(backend, head, storage)`。
- `.../impl/runtime/identity.h`：`make_identity(digests, tag, frontier) -> DiskKVIdentity`，
  复用 core 已抄的 `DiskKVIdentity`，不做平行类型。
- 测试 `test_runtime_mechanisms.cpp::test_prefix_digests`（前缀纯度、Vision/rewrite 折入点、
  `append_generated` ↔ 重建、`truncate`、`restore`、越界拒绝，外加独立 Python 转写出的
  wire-format golden）。它仍是**短名单**：真正复用判据是逐 token 的 `ResidentPrefixIdentity`，
  摘要碰撞只退化为一次比较，绝不产生假命中。

**块级索引层已落地（本轮 · 按 B 走）**。T2 粒度选 B：按 64-token 块索引，命中比整条 lane
镜像更细。已落：

- `src/runtime/prefix_cache/block_hash.{h,cpp}`：`kBlockTokens` / `kRootLookupHash` /
  `block_lookup_hash` / `block_lookup_hashes`，自上游**逐字节照抄**（只依赖 `ninfer/types.h`）。
- `src/runtime/prefix_cache/block_index.{h,cpp}`：`BlockIndex`，取自上游 `PrefixCacheIndex`
  节点树的可分核——`child_key` 把父节点混进 key、扁平 multimap 选桶、命中永远精确比
  `parent/hash/extra/tokens`、句柄带 generation、`remove_subtree` 让整棵失效；另加
  `BlockCopy{Device,Host,Disk}` 记录块字节在哪一层。**索引只持有身份与拓扑**，字节的搬运与
  驻留由 tier 负责。
- `src/targets/qwen3_6/impl/runtime/block_keys.{h,cpp}`：`vision_ranges` /
  `accumulate_vision` / `prompt_block_keys`，自上游 `block_keys` 适配到本树
  `PreparedPromptData`（同 key 算术与顺序）。
- 测试：`ninfer_block_index_test`（链式哈希的前缀纯度、extras 覆盖、强制同哈希不假命中、
  同 tokens 异 extra 必 miss、句柄 generation、子树删除）；
  `test_runtime_mechanisms.cpp` 的 `test_block_keys`（Vision key 在它开启的块起效、异内容异
  key、文本 prompt 无 extras）。

**没抄的部分（明确不做）**：上游 `prefix_index.cpp`（1637 行）里的 snapshot / tap /
GDSF / host slab / device slot / 持久化，是 Hybrid 模式（HPC）专门的概念；本树没有这些对象，
也超出"前缀指引的 chunk 存储管理"的范围。

**引擎接线已落地（本轮）**。上游 `models/qwen3_5/program/storage/disk_tier.cpp`（~600 行）是
它们 `ProgramImpl` 的成员，接的是它们 Hybrid 的 storage（`physical_pool()`、
`LogicalKVPageHandle`、snapshot 模型），与本树的两层 pool 结构性不兼容，所以**没有整文件取回**，
而是把它的语义落到本树已有的原语上：

- `EngineOptions::disk_kv_path` / `disk_kv_bytes` → `SequencePlanningInputs` →
  `SequencePlanImpl` → `ProgramImplCore` 构造尾部 `enable_disk_tier`。`disk_kv_path` 非空时要求
  host tier 已开启（恢复要读 hidden 行），且后端只支持 MTP（DFlash 的 lane 恢复走它自己的
  context cache，磁盘层不携带）。
- `qwen3_6::detail::LaneDiskTier`（`impl/runtime/lane_disk_tier.{h,cpp}`）：按 64-token 块写入
  每个 rank 的链，最后写 state image 作为 commit（`spill_pages` / `spill_state`）。同一份
  batch 接口既服务 device 侧（`spill_device_prefix`）也服务 host 侧（`spill_parked_prefix`，
  直接从 parked image 流式写出）。`plan_resume` 在**本 prompt 自己的摘要**下找最深的、
  每个 rank 链完整且 image 存在的 frontier。`allow_eviction=false`：磁盘族满了就让这次 spill
  失败，绝不挤掉另一条对话仍在恢复的记录。
- 恢复：`plan_request_base` 每个请求探测一次 `disk_restorable_frontier(prompt)`，把
  `FullReset` 改写成该 frontier 上的 `AppendAtFrontier`；`start_prefill_lane` 先把 lane 清空
  （同时把它自己那条 retained 前缀 spill 出去），再 `restore_lane_from_disk`：逐 batch 读进
  staging image → H2D → 重发布 block table，最后读 state image 决定 frontier。任一步失败就
  退化成冷 prefill。磁盘方案按**它可能退化的那次冷 prefill**计价（`service_base = 0`、
  `reusable_prompt_tokens = 0`），因为只有真去读字节的那一次才知道是命中还是 miss。
- `MemorySummary` 新增 `disk_tier_capacity_bytes` / `disk_tier_used_bytes` /
  `disk_tier_restores`，让"这次是从盘上回来的"可以直接读出来。

**没有**把 `BlockIndex` 接成引擎的第二套权威。磁盘层的 `DiskKVStore` 索引已经按内容
（`{lo,hi,tag,frontier}`）回答"有没有、在哪"；device 驻留由 lane 自己的
`PagedKVAllocation` 回答，host 驻留由 `parked_images_` 回答。再加一层 `BlockIndex` 会是同一
事实的平行副本，而 per-block 的 device/host 驻留目前**没有消费者**。`BlockIndex` 仍是纯身份/
拓扑层（`copies` 位留给未来真的要用块级驻留的那个调用方）。

### 7.9 P7 — 文档与 CLI

**文档**：

- 本文是当前权威：§0 记已交付能力与实测，§7.8 记 NVMe 接线，§9 记风险，本节记 CLI 与文档。
- `paged-kv-cache.md` §1.1、`concurrent-inference-architecture.md` §1.2 把 swap/KV offload 从
  non-goal 里拿掉，改为指向本文。
- `docs/performance.md` 增 context-tier 小节（park/restore 带宽与 0-re-prefill 端到端）。
- `docs/README.md` 把本文从"待实现"挪进 maintainer references。
- **没有**取回上游 `resource-scheduling-and-context-cache.md`（1028 行）：它描述的是上游的
  `resource_manager` 形态，与本文 §0 的落点不是一回事，取回会造出第二套平级权威。

**CLI/服务（已做）**：三个选项在 `apps/cli` 与 `ninfer-serve` 上同名并列在 `--help` 里：

| 选项 | 映射 |
|---|---|
| `--host-kv-mib N` | `EngineOptions::host_context_bytes`（`0` = 关，NVMe 层随之关闭） |
| `--disk-kv-path DIR` | `EngineOptions::disk_kv_path`（要求 host 层已开） |
| `--disk-kv-mib N` | `EngineOptions::disk_kv_bytes`（`0` = 引擎默认 64 GiB） |

`ninfer-serve` 启动记录在 host/disk 层任一开启时追加 `host-tier=/parked=/disk-tier=/live=/
restores=`。用户文档见 `docs/serving.md` 的 "Retained context" 与 `docs/cli.md`。上游把 NVMe
层的容量写进路径（`--context-tier-nvme <path:size>`），本树拆成 `--disk-kv-path` +
`--disk-kv-mib`，因为容量在 `EngineOptions` 里本来就是独立字段；上游的
`--device-state-slots`/`--host-state-slots` 对应本树没有的对象，不取回。

### 7.10 工时估计与里程碑（事后回看）

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M1 | P1 计价/决策层：`ninfer_resource_manager_test` 全绿 | ✅ |
| M2 | P2 权能接口决策；`host_kv_arena` 对接 | ✅ |
| M3 | P3 park→restore 逐字节 / 逐 token 正确（含逐出完整性） | ✅ |
| M4 | P4 + P5：见 §7.6/§7.7 | 本树**没有**取回 `engine_core.h`，也没有重做 TP2 peer 锁步；host tier 直接接在现有引擎壳（`evict_retained_lane` + admission policy + `start_prefill_lane`）上，双卡由 `ranks = 2` 的逐 rank 分片覆盖 |
| M5 | P6 + P7 | NVMe 下沉 ✅；CLI/serve 选项与文档 ✅（见 §7.9） |

最初的区间主要被 P2 的 (a)/(b) 选择支配；实际按 (b) 走通，且 P4/P5 用更小的改动达成，
没有出现"重接约 9k 行 Program 逻辑"的那条分支。

**建议的第一步就是 P1**：它不碰 CUDA、不碰存储、不需要 GPU，用一份只含一个 `#include`
的测试文件就能验证整套权能接口定义是否正确。这是整条移植路径上最便宜的一次"方向对不对"检验。

---

## 8. 验证

| 主张 | 证据 |
|---|---|
| 恢复后行为等价 | 恢复后的续写与不 swap 对照**逐 token 相同** |
| image 往返无损 | park→restore 的 KV 字节逐字节比较 |
| TP2 分片一致 | 两 rank 各自恢复出的分片与寄存前相等；跑 `tools/tp2` parity |
| 失败不污染正确性 | 注入损坏的 NVMe 页 → 必须表现为 miss |
| 收益真实 | 8k / 64k / 200k 下的 park/restore 延迟落在带宽模型内 |
| 目标形态规模 | `ninfer_qwen3_6_27b_disk_pool_real_test`（默认 8 × 150K）：池 153,600 token、引擎销毁重开后 8 条全部 **0 prefill token**、各自续写与冷跑逐 token 相同、盘上 42.7 GiB 不变。门里**不发 flush 请求**：最后那条常驻会话的记录只能来自它自己的完成发布，`NINFER_DISK_POOL_CONCURRENCY=1` 覆盖单 lane 的串行形态，`NINFER_DISK_POOL_SPEC=none` 覆盖非投机（默认）解析路径 |
| 服务形态规模 | 真 HTTP endpoint 的 8 × 194,950 token 会话池（见 §0.4）：两次进程重启后 8/8 零 prefill 恢复，TTFT 5.2–6.2 s |

---

## 9. 风险

- **瓶颈是 PCIe Gen3 x4（~3 GB/s），不是 NVMe。** 200k 上下文 ≈ 1.1 GB/设备 →
  单向 ~370 ms。比 re-prefill 快约 200×，但不再是零。
- restore 在 claim 路径上；如果不能与 admission 交叠，就会串行化，收益被吞掉。
- **host arena 自己不回收**。`HostKVOffload` / `HostLinearStateArena` 只分配与释放，不会为了
  新 lane 去挤掉旧 lane。L2 满时这一次 park 失败，该 lane 降级到 L3（它已经有记录），恢复变成
  一次磁盘读而不是 host 读——**开了 NVMe 层就不丢前缀，代价是延迟；只开 host 层才会真的丢。**
  真正的 demotion 策略（按 parked 时间挤掉最老的一条）需要一个 executor 侧的 policy hook：
  Program 无法在 `evict_retained_lane` 里作废另一条 lane 的缓存 plan，做错会把一个请求放上
  没有 KV 的 lane。
- **完成时的发布是同步的，且第一轮要搬整条前缀。** `publish_retained_prefix` 在 terminal round
  里调用，与逐出路径共用同一套 `spill_device_tap`：一条会话的第一轮会把整条 chain 从 device
  搬到 host 再落盘（195K 会话 ≈ 7 GiB，秒级），之后每轮只有新增块 + 一张 state image
  （195K 下 ≈ 0.6 GiB）。串行 profile（`--max-concurrency 1`）下这段延迟不影响别人；并发 profile
  下 `max_concurrency` 条 lane 同时 terminal 会叠加，是这套设计当前最重的临界区。把它挪到后台
  线程或 idle tick 需要 executor 侧的调度钩子（与 demotion 策略同一处），本轮没做。
- **一个 tap 可能只落一半**：`spill_device_tap` 先写 KV chain（大）再写 state image（154 MB）。
  state family 装不下时 tap 失败，已经写进去的 chain 就成了**孤儿**——占盘、但因为没有 state
  image 而永远无法 resume（实测触发过一次：2 GiB 预算下 state family 只有 205 MiB，第二条会话
  落盘 55 MiB 无用数据）。state family 是每条会话 308 MB 的固定成本（2 个 tap），所以短会话
  下它比 Main family 先爆（4K 会话：Main 138 MB vs state 308 MB）。修法是把 state image 写在
  chain 之前——失败时只浪费 154 MB 而不是 5 GiB。
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

---

## 11. 本机实测参考

- 每设备文本 KV：16,896 B/token（16 层 full attention × 2 KV head × 256 head_dim ×
  1 B × 2 平面 + FP16 group-64 scale）。
- 每设备 runtime 预留：`760,225,536 + 24,224 × N` 字节（N = token 数，含 MTP KV），
  在 179k–1M 区间与实测吻合到 ~2 KB。
- headless 基线（gdm3 已停、两块卡各 16,145 MiB 空闲）下：
  NVFP4 上限 179,200；NVFP4 + 桌面占用时 180,224 会失败；Swift-1.5 上限 208,896；
  GSQ-RCO 可开满原生 262,144。
