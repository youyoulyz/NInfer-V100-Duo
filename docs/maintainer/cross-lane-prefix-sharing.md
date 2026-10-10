# 跨 lane 前缀共享（block 级 refcount）

**状态**：**P2 + P3 + P4 已落地**。P0（本文 + 契约修订）、P1（store 半边的页组 read-only
borrow + refcount、Program/admission 半边的跨 lane claim）、P2（内容寻址的 `SharedBlockStore`：
admission 走树 lookup、commit 时 `intern` 入树、淘汰按"只有它服务的 frontier"排序 + 摘叶）、
P3（节点挂 state image：lane 被顶掉前把 checkpoint 的 GDN state 与 hidden 行capture 成镜像，
claim 拷进私有 slot，不再依赖任何 lane 存活）与 P4 的节点化半边（节点自己持有页组 hold +
state image；tier 的"按节点去重"落在索引与镜像上）都已完成并通过端到端验收（§9）；
`--log-stats` 出共享索引的 nodes/owners/ownerless/payloads/hits/pruned。本文是这项能力的当前
权威：目标、物理约束、数据结构、refcount 语义、COW 边界、分阶段验收。§1–§8 是契约。

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
- `SharedBlockStore`（`src/runtime/prefix_cache/shared_block_store.{h,cpp}`，已落地）：
  `BlockRef → { owners[≤8] = {lane, frontier, role, kind, last_use}, pages[4], payload }`。
  - `owners` 是"哪条 lane 还能把这段链交出去"的承诺：lane 存活时最便宜，claim 前用 lane 的实时
    state 重新验证（P2）；
  - `pages[4]`（Main Text rank0/rank1、backend rank0/rank1）是这一段 64-token 块的页组，由
    **节点自己持有**：capture 时 `hold_page`，节点被摘掉时归还（P4 半边）；
  - `payload` 是节点自带的续跑镜像 `{tail page, image slot, frontier, kind, role, copies}`（P3 +
    P4）：一个 frontier 的 GDN state、hidden 行与 metadata 存在 host 镜像槽里，claim 不再需要任何
    lane；`copies` 说明这份 payload 的 KV 在哪——`Device` 表示 store 持着链上的页组，`Host` 表示
    Program 已经把每个 pool、每个 rank 的 KV 各存了一份 pinned host 镜像，device 页组已经还回池子
    （`--shared-kv-mib` 给出这个 host 预算；装不下就退回 `Device`）。
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
P2 阶段页组仍由产生它的 lane allocation 持有（P1 的 borrow count 保证 claimer 存活期间不被回收），
所以节点只需要 **owner 承诺**（`lane` + 它当时能续到的 `frontier`）；P4 把节点变成页组 owner
之后，这里的 refcount 才同时成为页的 refcount。

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

`find_admission_lane` 仍对每条 free lane 各算一份 plan，按 `reusable_prompt_tokens` 排序，
但"每条 lane 只能复用自己的前缀"这个限制已经去掉：**plan 现在按前缀而不是按 lane 计算**。
实测过的两个缺陷里，第一个（同一份前缀只有一条 lane 能用）由 P1 修掉，第二个（所有候选都要
丢弃 retained 前缀时死磕最低序号 lane）留给 P2。

P1 的规则：

1. 先按 lane 自己的 state 求最深可续 frontier（`deepest_resident_reuse`：commit 的
   `execution_frontier`，或 prefill 捕获的 rewrite checkpoint）；
2. 自己的 state 续不上时，**扫描全部兄弟 lane**：idle（`retained`）的按同一规则借它的驻留前缀，
   仍在跑请求的只能借它**恢复所依据的 rewrite checkpoint**（`checkpoint_claim`）—— 它自己
   的活 frontier、frontier 以上的页、`current` GDN slot 每轮都在动，只有那个 checkpoint 的
   frontier 以下（页已写完、不再被写）和它自己那份 state slot（下一次 capture 前无人覆盖，
   且 capture 会先更新这份元数据，所以失效的 claim 会在 admission 时被验出）可以借；
3. plan 只为自己拥有的那一段计价：`*_page_entitlement` 减去借用页数，借用的页仍记在 donor 的
   entitlement 上（`summary.admission` 保持冷起点的保守账，见 §8）；
4. 命中就 claim（页 refcount++、state 拷私有 slot）；未命中 → 冷 prefill，完成后按
   `paged-kv-cache.md` §9.5 的 retain 语义保留前缀 —— 这条路径与本文之前完全一致；
5. 平分复用分（reuse 相同）时按"不丢弃前缀 > 丢弃前缀，其次**借别人的 > 吃自己的**"取舍：
   吃掉 owner 自己的 lane 会让这份前缀变成一条在跑请求的、后面的兄弟只能去 claim 一条正在写的
   lane，所以**让 owner 留在 idle 上**、由别人借，claim 在整个 burst 里都稳定；
6. 借来的前缀落在哪条 lane 上与该 lane 原来持有什么无关：adoption 会像冷启动一样替换掉它自己
   的前缀（开 tier 时先 spill 到 NVMe），因为 plan 已经判定它续不上这条 prompt；
7. "要不要为这条 request 额外抓一个 checkpoint（分裂点 capture）"仍由已存在的价值决策层
   （`SharedCapturePlanner` / `FinalScheduleIntent::shared_capture_frontiers`）决定，而不是
   无条件捕获；该层目前只差真实 Package 的实现。

P2 已把这条 lane 扫描换成树的 `lookup(prompt)`，并让淘汰按"只有它服务的 frontier"排序 + 摘叶：

1. **publish 时机**：prefill 结束、rewrite checkpoint 建立时（owner = 该 checkpoint 的
   frontier，供仍在跑的 lane 被兄弟 claim），以及请求终态 retain 时（owner = `execution_frontier`，
   供 idle lane 被 append 复用）。两条路径都只在该 request 允许复用、无 media、非 DFlash 时发布；
2. **lookup 语义**：`block_lookup_hashes(prompt)` 沿树下沉，返回**最深**的、至少有一条 lane 承诺
   的节点 + 该节点的 owner 集合；`serves(owner, tokens)` 由 Program 用 lane 的实时 state
   （idle → `deepest_resident_reuse`，busy → `checkpoint_claim`）逐 lane memoize 求值。稳态下
   候选集合与旧扫描一致（差别只在"只有内容匹配的 lane 才被求值"）；唯一残留差异是 adoption lane
   在它自己的 prefill 结束、publish 之前那一小段：它克隆来的 checkpoint 在旧扫描里已经可借，
   在新实现里要等它 publish，而同一内容的 idle owner 通常仍在，因此复用深度不变；
3. **淘汰顺序**：`retained_lane_eviction_order()` 按 `exclusive_frontiers()`（只有这条 lane 服务
   的最深 frontier）升序、再按最近 retain 时间升序给出受害者；executor 的 evict 循环按这个顺序
   驱逐，且永不动当前 plan 依赖的 donor。旧实现按 lane 序号从小到大吃，正是"死磕最低序号 lane"
   那个缺陷；
4. **摘叶**：lane 的链被替换或拆除时（cold/adopt/clear）先 `forget_lane` 再 `prune()`：沿树把
   `owner_count == 0` 的叶节点逐个摘掉，摘掉末端会暴露父节点，于是整条没人服务的分支会消失；
   索引容量不足时先摘这种叶节点，仍然不够才按 LRU 牺牲还有承诺的叶节点。

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
| **P1** | 页组加 `SHARED` + refcount；**单个 retained entry 允许 N 条 lane 同时 claim**（state 各拷一份），lookup 仍走现有 lane 扫描 | 八条 byte-identical 的 10000-token 流（`run-prefix-compare.sh 10000 8`）：**8/8 `restore_turn_checkpoint`、0 re-prefill、窗口 2.51 s**（改造前 3/8 重 prefill、78.43 s）；`adopt_probe.py` 里被 adoption 的两条流输出与冷启动逐字节相同 |
| **P2** | 接 `BlockIndex`：admission 走树 lookup，commit 时 intern 入树，淘汰按 exclusive frontier + 摘叶（已落地，见下） | 八条 byte-identical 的 10000-token 流（`run-prefix-compare.sh 10000 8`）：**8/8 `restore_turn_checkpoint`、`computed_prefill_tokens` 最大 2、窗口 2.51 s、aggregate 200.5 tok/s**（同 profile distinct 基线 2.29 s / 220.5 tok/s）；`adopt_probe.py` 两条 adoption 输出与冷启动逐字节相同；`--log-stats` 报 `shared_nodes=16 shared_owners=64 shared_hits=18/34`。"同前缀不同后缀"（checkpoint 之后才分歧的 replay）走同一条 lookup→claim 路径、由 checkpoint frontier 决定复用深度，本轮没有单独测量；host tier / NVMe tier 两条真实权重端到端 gate（`ninfer_qwen3_6_27b_host_tier_real_test`、`ninfer_qwen3_6_27b_disk_tier_real_test`）在改动后的淘汰顺序下仍通过 |
| **P3** | state 共享：节点挂 state image、claim 拷私有 slot（已落地，见下） | `ninfer_qwen3_6_27b_shared_payload_real_test`（2 lane、3 个不同对话，第三个顶掉第一个 lane）：被顶掉的前缀 `payloads=1`，再问它 `reuse != full_reset`、`reused=1943`、`prefill_tokens=0`、生成 token 与冷跑逐 token 相同；同一场景把镜像关掉（`shared_state_images=0`）则 `reuse=full_reset`、`prefill_tokens=1943`。分裂点 capture 仍由 `SharedCapturePlanner` 的价值层决定（本轮只挂 checkpoint 与 retain 两个点） |
| **P4** | 节点持有页组 + state image + **KV 每节点一份副本**（`--shared-kv-mib`，已落地）+ 观测（`--log-stats` 出 shared nodes/owners/ownerless/payloads/held_pages/hits/pruned） | `ninfer_qwen3_6_27b_shared_payload_real_test` 三个变体：仅 state image → `payloads=1` 且 `held_pages>0`；state image + `--shared-kv-mib 256` → `payloads=1 **held_pages=0**`，同一条被顶掉的前缀仍 `reuse=3 reused=1943 prefill_tokens=0` 且生成 token 与冷跑逐 token 相同；全关 → `reuse=full_reset prefill_tokens=1943`。节点页组 hold/归还、镜像槽 LRU、host payload 的 lookup 不依赖页组均由 `ninfer_shared_block_store_test` 覆盖 |

P1 单独即可消除 §6 的第一个缺陷，且不引入新算法；P2 起才真正使用 radix 树。

**P2 落地形态**（`SharedBlockStore` + Program 接线）：

- `intern`：`prompt_block_keys`/`block_lookup_hashes` 给出的链式 hash 沿 `BlockIndex` 插树，
  每个覆盖到的节点打上 owner 承诺 `{lane, frontier}`；同一内容再次 intern 只是追加/刷新 owner
  （每节点最多 8 个，超出按 LRU 顶掉），所以 N 条 lane 的同一前缀共用同一批节点；
- `lookup`：只读树，返回最深命中节点 + 仍被 `serves` 接受的 owner 集合。**承诺不是答案**：plan
  照旧用 `deepest_resident_reuse` / `checkpoint_claim` 对 lane 的实时 state 重新求值，失败就退回
  冷 prefill（P1 的 fallback 计价不变）；
- 遗忘与摘叶：`clear_lane`、cold 重建、adoption 替换链之前都 `forget_lane` + `prune`，所以承诺
  只活在"lane 还持有这条链"的期间；索引满时先摘无人服务的叶节点，再按 LRU 牺牲；
- 驱逐顺序：`retained_lane_eviction_order()` 按 `exclusive_frontiers()`（只有该 lane 服务的
  最深 frontier）升序、再按 retain 时间升序，executor 不再按 lane 序号吃前缀；
- 观测：`--log-stats` 的吞吐行追加
  `shared_nodes/shared_owners/shared_ownerless/shared_hits=<hits>/<lookups>/shared_pruned`。

**P3 + P4 节点化落地形态**：

- 节点自己持有页组：capture 时对链上每个块的 4 个页组 `hold_page`（`PagedKVPool::hold_page`
  复用 borrow 计数，不需要 row），节点被摘掉时归还。页组只在该节点有 payload 时被 hold，
  lane 存活期间池子的账目与 P1 完全一致；
- 节点自带 state image：lane 被顶掉（`evict_retained_lane` / cold 重建 / adoption 替换）之前，
  `capture_node_payload` 把该 lane 的 rewrite checkpoint 连续状态——两个 rank 的两组 GDN slot、
  rank 0 的两条 hidden 行、ledger/identity/digests 前缀、frontier 与 kind——存进镜像槽，并把
  链上的页组 hold 住；镜像槽数量由 `--shared-state-images N` 定（每张约 `2 × rank × 294 MiB`
  的 host RAM）；
- claim：`adopt_node_prefix` 用链上的页组 reserve 出一个"借用前缀 + 自有 suffix"，frontier 所在
  的那一页照旧 copy 成自有页（COW 边界不变），再把镜像 `restore_linear_state_to` 到本 lane 的
  两组 slot、拷 hidden 行、重建 sequence metadata（frontier/checkpoint/identity/digests），
  `reuse` 走 `restore_turn_checkpoint` / `restore_response_checkpoint`，之后按原路径只 prefill
  suffix；
- 容量：页组被 hold 意味着池子少了几页。`reclaim_payload_pages` 在 cold 重建（lane 的页已经
  交回）之后按 LRU 丢镜像，刚 capture 的那张 last_use 最新、最后被动到；真正没有空间时，
  executor 也会先 `drop_shared_payloads(1)` 再考虑吃掉一条 retained 对话——镜像始终是"机会
  拷贝"，不会阻止准入。

**KV 每节点一份副本（P4 收尾）**：`--shared-kv-mib N`（默认 0）给出一个 pinned host 预算，
capture 时先试着把这道链每个 pool、每个 rank 的 KV 用 `HostKVOffload::park_range` 各存一份
per-node 镜像；四份都成功就把 `payload.copies` 记为 `Host`、**不**再 hold device 页组（页随 lane
的交还回池子），claim 时用 `HostKVOffload::restore` 把镜像落进该 lane 自己的一整套新页里
（entitlement 不拆分，等价于冷起点）；任何一份装不下就整份放弃，退回 `Device` 持有页组的模式。
这样被顶掉的对话不再占用 device 容量，代价是 host RAM（每 1943 token 约 35 MB，随前缀线性增长）。

仍未做：host/NVMe 的"每节点一份"目前是**状态镜像 + KV 镜像各一份**（两份都在 host 预算里，KV 由
`--shared-kv-mib` 限制、state 由 `--shared-state-images` 限制）；把两者合成一个统一的 per-node
image、或让 L3 也按节点（而不是按 64-token 块）组织，都只影响存储成本与实现形状，lookup /
claim / 淘汰的语义不再变。

P1 分两半，两半都已落地：

- **store 半边**：`PagedKVPool::adopt_shared()` 让一个 allocation 把 owner 的物理页发布
  进自己的 block-table row；borrow count 保证这些页在最后一个 borrower 释放前不回 free set，
  即使 owner 先释放自己的前缀；borrowed allocation 没有自己的 entitlement，不能 grow / trim /
  resize。`adopt_prefix()` 补上真正的形状：**借用的前缀 + 自有的后缀**在同一 allocation 里，
  `owned_page_count()` 区分两者，entitlement / materialize 只算自有部分，trim 不能切进借用部分。
  验证：`tests/test_kv_cache.cpp`。
- **Program/admission 半边**：`adopt_lane_prefix()` 在 admission 时把 donor 的续跑状态整体克隆到
  执行 lane —— 前缀页 read-only 借用、frontier 所在的那一页（两条 lane 都要往里写）拷成自有页、
  两个 rank 的 GDN state 两个 slot 与两条 hidden row 一起拷、ledger/identity/digests/
  `rewrite_checkpoint`/`execution_frontier` 全部继承。读代码时确认的三处互锁都已处理：

  1. 克隆后 `start_prefill_lane` 的复用判定与 MTP/DFlash readiness 门对 **donor 的 sequence**
     求值（`resume`），并且 `text_kv_page_entitlement` / `backend_kv_page_entitlement` 减去
     donor 已映射的页数，`can_admit_lane` 的容量账不会把同一批页算两遍；
  2. adopted 前缀的页数正好等于 `trim_sequence_kv(sequence, base, ...)` 要的
     `ceil(base/64)` —— 借用部分是 `base/64` 页、frontier 那一页自有，所以 trim 的
     "不能切进借用部分" 守卫不会拒绝；
  3. 复用路径不可用时回退到 plan 已经计价的冷 prefill（adoption 是纯增益路径）：回退会连同
     `summary.service_work_quanta` 一起换回冷起点那份投影，否则一条 9998-token 的冷 prefill
     只会被计价 2 token，service-work 账当场溢出。

  实测（2 x V100-SXM2-16GB、`--tp 2 --spec mtp --draft-tokens 3`、INT8 KV、8 lane、
  `--prefill-chunk 128`、`--max-context 90112`）：八条 byte-identical 的 9998-token 流
  8/8 `restore_turn_checkpoint`、`computed_prefill_tokens` 最大 2、窗口 2.51 s、
  aggregate 200.4 tok/s；同 profile 的 distinct-8x10000 基线是 2.27 s / 222.5 tok/s，
  改造前同一条 shared 用例是 78.43 s、3/8 重 prefill、6.4 tok/s。

  这条路径的数值正确性由 `tools/v100/context-lab/adopt_probe.py` 保护：同一 prompt 先冷跑一次
  取参考文本，再并发两条（其中必然有一条只能借兄弟 lane 的前缀）——两条的输出必须与参考逐字节
  相同，且都是 `restore_turn_checkpoint` + 2 token prefill。

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
