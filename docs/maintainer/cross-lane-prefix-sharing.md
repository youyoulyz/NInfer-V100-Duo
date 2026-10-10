# 跨 lane 前缀共享（block 级 refcount）

**状态**：**进行中**。P0（本文 + 契约修订）与 P1 的 store 半边（页组 read-only borrow +
refcount）已落地；P1 的 Program/admission 接线与 §9 其余阶段待做。本文是这项能力的当前权威：
目标、物理约束、数据结构、refcount 语义、COW 边界、分阶段验收。实现完成后 §1–§8 仍是契约，
§9 的阶段表按实际完成情况改写。

**范围**：让**同一条 prefix** 被 N 条 active lane 同时复用（`max_concurrency` 条共用一份
device 页与一份 state 快照），而不改变单请求上下文上限、不改变 `max_concurrency ≤ 8`、
不引入 active request 抢占。共享是**内部存储语义**，不改 `.ninfer` 契约与外部协议。

本文修订 [Paged KV Context Store](paged-kv-cache.md) §1.1 与 §10.3 —— 那里把
`active requests 之间共享可写 prefix / page reference counting / copy-on-write branching`
与"同一 retained bundle 被多个 active requests 分叉"列为 non-goal，现在它是一条有明确契约的能力。

---

## 1. 目标与非目标

目标：

- 同一份前缀被 N 条 lane 同时 claim：N 条并发请求全部 `prefix_reuse_path` 命中，**0 个
  prefill token**；
- 块级去重：共享的 device 页只占一份显存，页组引用计数归零才回收；
- 复用点落在 target 证明拥有完整 continuation state 的 checkpoint 上（见 §2）；
- 淘汰按"引用计数为零的叶节点 + 价值/LRU"选牺牲者，不再按 lane 序号。

非目标：

- **不降低互不相同会话的 per-conversation 成本**。radix 只对重复前缀有效；`10 × 150K`
  这种互不相同的会话仍靠 [上下文两层寄存](context-tiering.md) 的 L3 轮转，这一点不变；
- 不做任意位置的 state 分支（state 捕获仍是显式策略，见 §5）；
- 不做 request 抢占、不做跨 GPU 存储、不改 pool layout 或 page size。

## 2. 决定上限的物理事实

Qwen3.6/3.8 是混合模型，两类层的可共享性完全不同：

| 层 | 每 token payload | 能否块级共享 |
|---|---|---|
| full attention | 每 token 一份 K/V，内容只取决于该 64-token 块 | **能**（Main 4.72 GiB + MTP 0.29 GiB / 150K lane） |
| linear attention（GDN） | **没有**每 token KV，只有一份定长 recurrent state | **不能**；只能快照，claim 时整份拷贝（约 146.8 MiB/rank，与长度无关） |

推论：**一个可恢复点必须同时命中 pages 和 state**。树可以按 64-token 去重 pages，但
"0 prefill 续跑"只在**持有 state image 的节点**成立。state 不可能每块一份
（146.8 MiB / 64 token 不成立），所以复用点仍受 checkpoint 稀疏性约束 —— 这一点与
`paged-kv-cache.md` §2"reusable-state granularity 由完整 sequence state checkpoint 决定"
一致，本文不改变它。

## 3. 现状：三块已 staged 但未接线

设计不是从零开始。树里有三块现成构件，**全部零调用方**（`6276fee3`，2026-10-09 引入）：

| 构件 | 位置 | 语义 | 现状 |
|---|---|---|---|
| radix 块树 | `src/runtime/prefix_cache/block_index.{h,cpp}`、`block_hash.{h,cpp}` | 节点 = 一个 64-token 块；键 =（parent, chained hash, 64 tokens, extra）；`pins`、`copies = Device\|Host\|Disk` 位图、generation-checked handle、`remove_subtree` | 仅被 `tests/test_block_index.cpp` 引用 |
| 链式块键 | `src/targets/qwen3_6/impl/runtime/block_keys.{h,cpp}` | `prompt_block_keys()`：prompt → 每 64-token 块的链式 lookup hash + 累积 Vision key | 仅被自身 `block_keys.cpp` 引用 |
| 共享发布决策层 | `src/runtime/engine/{resource_manager,materialization_planner,shared_capture_planner}.h` | `SharedPrefixHandle`、`shared_owners`、`explicit_shared_credit`、`FinalScheduleIntent::shared_capture_frontiers` | 仅被 `tests/test_resource_manager.cpp` 的 `FakePackage` 实例化 |

`kBlockTokens = 64` 与 `kPagedKVPageSize` 相等，正是"一个树节点拥有一组页"的几何前提。
`BlockIndex` 的自我定位是 **identity and topology only**："字节在哪是 tier 的事，caller 通过
`copies` 记录自己持有的副本并自己搬字节"。这条边界要保持：页 id 不进树。

已经可共享的下游是 L3：磁盘层的键 `DiskKVIdentity{lo, hi, tag, frontier}`
（`identity.h` 的 `make_identity`）**已经是内容寻址**，同一条前缀天然映射同一条目。
缺的是 device/host 层。

## 4. 数据结构

- `BlockIndex` 保持只管 identity/topology，不塞页 id；
- 新增 `SharedBlockStore`（Program 侧，与 pool 同生命周期）：
  `BlockRef → { pages_main[rank], pages_mtp[rank], optional StateImage, refcount, last_use }`；
- **页的键**用 `prompt_block_keys` 的链式 hash；**state 的键**用 `make_identity`。一个 frontier
  只有两个键都匹配才算 hit，且两者必须同源导出 —— `identity.h` 的注释记录过一次"重复推导
  identity 导致每个 checkpoint 都带一个 lookup 永远产生不出来的 tag"的事故，不要重犯；
- Vision、token_type、position 轴已经折进 `extras` 与 `PrefixDigests`，不会误命中；
- TP2：树是**每 rank 一份同构元数据**（同 key、同拓扑、同 depth），页 id 分 rank 存；分配必须
  rank 对称，任一 rank 失败则整体放弃共享并退回独占路径。

## 5. 引用计数、COW 与 state

**页组生命周期**：`FREE | OWNED(lane) | SHARED(refcount > 0)`。
`refcount` = 树内部引用（有子节点或挂有 state image → 1）+ 当前映射它的 lane 数。
`BlockIndex::pins` 只承担树内的 pin 语义，device 页的 refcount 记在 `SharedBlockStore`。

- `clear_lane` / `evict_retained_lane` 从"释放 bundle"改成 **release ref**；
- `can_admit_lane_after_retained_eviction` 的容量账必须把 `refcount > 0` 的页算作不可回收；
- 只有 `remove_subtree` 且 refcount 归零才回 free set。

**state**：节点持一份 state image。claim 时 state 是 **device→device 拷进该 lane 的私有
slot**（`linear_state_slots`），页是**映射共享**。拷贝量约 146.8 MiB/rank，µs–ms 量级，相对
分钟级 prefill 可忽略；但 N 条 lane 同时 claim 同一节点会产生 N × 146.8 MiB 的瞬时流量。

**COW 边界**：prompt 长度不是 64 的倍数时，最后一个不满块**共享整页 + 写前私有化该页**
（≤64 token 的 payload），不把 frontier 向下取整 —— 与 `paged-kv-cache.md` §10.2 的
non-page-aligned frontier 语义一致。

**decode 追加**：跨页前先查树；append 后的块若已在树中，把新块也 refcount++ 而不是复制。
这自然支持"两条 lane 在同一对话前缀上分叉"。

## 6. 准入与淘汰

现状：`find_admission_lane` 对每条 free lane 各算一份 plan，按 `reusable_prompt_tokens` 排序。
这个"按 lane 找前缀"的模型有两个已实测的缺陷：

- 同一份前缀只可能有一条 lane 持有，其余 lane 只能全量 prefill（本文要修的主问题）；
- 所有候选 lane 都必须丢弃各自 retained 前缀时，平分复用分会退化成**死磕最低序号 lane**。

改造后：

1. `lookup(prompt)` 沿树下潜，得到最深**可恢复**节点（pages 与 state 同时命中）；
2. 命中 → claim（页 refcount++、state 拷私有 slot）；未命中 → 冷 prefill，完成后按
   `paged-kv-cache.md` §9.5 的 retain 语义 `intern_chain` 入树并挂 state image；
3. 淘汰只在 **refcount 为零的叶节点**上按价值/LRU 选牺牲者，`evict_retained_lane` 的
   "按 lane 序号"逻辑退化成摘叶 —— 同时修掉上面第二个缺陷；
4. "要不要为这条 request 额外抓一个 checkpoint（分裂点 capture）"由已存在的价值决策层
   （`SharedCapturePlanner` / `FinalScheduleIntent::shared_capture_frontiers`）决定，而不是
   无条件捕获；该层目前只差真实 Package 的实现。

## 7. 与两层寄存（tier）整合

现状是**按 lane** 的 image：`LaneStateImage`、`parked_images_[lane]`、`spill_device_prefix` /
`spill_parked_prefix` / `publish_retained_prefix` 都以 lane 为索引。

改成**按节点**：一个节点在 device / host / disk 三处各有副本，`BlockIndex::copies`
（`Device|Host|Disk` 位图）已经是这个语义。L3 已经内容寻址、天然去重；真正要新做的是 L2
host 层按节点而不是按 lane 存 image。收益：同一 checkpoint 被 N 条 lane 命中时，tier 只存一份。

## 8. 代价与边界

- 每次 claim 拷 state：约 146.8 MiB/rank；相对冷 prefill（分钟级）可忽略；
- state image 占位：每节点一份，而本机 device 启动后只剩 ~545 MiB（见 `context-tiering.md`
  §0.5），所以可恢复节点的主体必须放 L2/L3 —— **这是可恢复节点数量的主要容量限制**；
- 显存收益只对重复前缀成立；互不相同的前缀零收益；
- 共享需要 N 条 lane 的 block table 指向同一页组，因此"一个 GPU execution unit 期间
  page mappings 稳定"这条不变量必须按 lane 分别维持（共享页在 unit 内只读）。

## 9. 分阶段与验收

| 阶段 | 内容 | 验收证据 |
|---|---|---|
| **P0** | 本文 + `paged-kv-cache.md` §1.1/§10.3 契约修订；`docs/README.md` 路由 | `git diff --check`；与本表一致 |
| **P1** | 页组加 `SHARED` + refcount；**单个 retained entry 允许 N 条 lane 同时 claim**（state 各拷一份），lookup 仍走现有 lane 扫描 | 八条 byte-identical 的 10000-token 流：**8/8 `restore_turn_checkpoint`、0 re-prefill**（当前是 3/8 重 prefill、窗口 78.43 s → 2.28 s 量级） |
| **P2** | 接 `BlockIndex`：admission 走树 lookup，retain 时 `intern_chain`，淘汰改摘叶 | 同前缀不同后缀的第二条请求在 checkpoint 节点命中，suffix 只 prefill 增量 |
| **P3** | state 共享：节点挂 state image、claim 拷私有 slot、分裂点 capture 接 `SharedCapturePlanner` | 两条 lane 同时 decode 同一前缀，输出与"从不共享"对照**逐 token 相同**（沿用 `context-tiering.md` §0.4 的验收法） |
| **P4** | tier 按节点化 + 观测（`--log-stats` 出 shared pages / hits / refcount） | 端到端 + `docs/serving.md` / README 更新 |

P1 单独即可消除 §6 的第一个缺陷，且不引入新算法；P2 起才真正使用 radix 树。

P1 分两半，进度如下：

- **store 半边（已落地）**：`PagedKVPool::adopt_shared()` 让一个 allocation 把 owner 的物理页发布
  进自己的 block-table row；borrow count 保证这些页在最后一个 borrower 释放前不回 free set，
  即使 owner 先释放自己的前缀；borrowed allocation 没有自己的 entitlement，不能 grow / trim /
  resize。`adopt_prefix()` 补上真正的形状：**借用的前缀 + 自有的后缀**在同一 allocation 里，
  `owned_page_count()` 区分两者，entitlement / materialize 只算自有部分，trim 不能切进借用部分。
  验证：`tests/test_kv_cache.cpp`。
- **Program/admission 半边（待做）**：`adopt_retained_prefix(dst_lane, src_lane, frontier)` ——
  借页 + 用 `LinearAttentionStatePool::copy_slot()` 把 src 的 GDN state 拷进 dst 的私有 slot +
  搬 ledger/identity/frontier 元数据；以及让 `plan_request_for_lane` 的"在 lane O 命中"能够落到
  lane L 执行。**在这半边落地前，跨 lane 共享对产品行为没有任何影响。**

## 10. 不改变的东西

`max_concurrency ≤ 8`、单 GPU、单 resident model、`.ninfer` 唯一 C++ 产品产物、OpenAI 与
Anthropic 外部协议行为。本文不引入 active request 抢占，也不改变单请求上下文上限。

## 11. 参考

- [Paged KV Context Store](paged-kv-cache.md) —— pool layout、page ownership、reservation、
  reusable checkpoint 语义（本文修订其 §1.1 与 §10.3）
- [上下文两层寄存](context-tiering.md) —— retained 上下文的 host/NVMe 寄存与恢复
- [小规模并发推理架构](concurrent-inference-architecture.md) —— admission 与 execution unit
- `src/runtime/prefix_cache/` —— 已 staged 的块 hash 与块索引
- `src/targets/qwen3_6/impl/runtime/block_keys.h` —— prompt → 链式块键
