#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "runtime/prefix_cache/block_hash.h"
#include "runtime/prefix_cache/shared_block_store.h"
#include "core/gdn_replay_records.h"
#include "core/host_kv_offload.h"
#include "core/host_linear_state.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/impl/runtime/host_lane_state_store.h"
#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/identity_tag.h"
#include "targets/qwen3_6/impl/runtime/lane_disk_tier.h"
#include "targets/qwen3_6/impl/runtime/dflash_context.h"
#include "targets/qwen3_6/impl/runtime/linear_state_slots.h"
#include "targets/qwen3_6/impl/runtime/prefix_digests.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"

#include <cstdint>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>
#include <string>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using PreparedPromptData    = qwen3_6::PreparedPromptData;
using RewriteCheckpointKind = qwen3_6::RewriteCheckpointKind;
using RewriteCheckpointSpec = qwen3_6::RewriteCheckpointSpec;

using ReusePath = ninfer::PrefixReusePath;

[[nodiscard]] constexpr bool is_rewrite_checkpoint_restore(ReusePath path) noexcept {
    return path == ReusePath::RestoreTurnCheckpoint || path == ReusePath::RestoreResponseCheckpoint;
}

[[nodiscard]] constexpr ReusePath restore_path(RewriteCheckpointKind kind) noexcept {
    return kind == RewriteCheckpointKind::TurnClosure ? ReusePath::RestoreTurnCheckpoint
                                                      : ReusePath::RestoreResponseCheckpoint;
}

enum class RewriteCheckpointAction : std::uint8_t {
    Drop,
    KeepExisting,
    ReclassifyExisting,
    CaptureNew,
    DeferCapture,
};

enum class MtpBridgeMode : std::uint8_t {
    None,
    BeforeSuffix,
    AfterExactHit,
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
struct RequestBasePlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::shared_ptr<const qwen3_6::VisionControl> vision_control;
    std::size_t vision_transient_bytes = 0;
    std::optional<qwen3_6::RewriteCheckpointSpec> rewrite_checkpoint;
    bool allow_prefix_reuse = false;
    // Deepest frontier of this prompt the disk tier can resume (0 when it cannot or the tier is
    // off). Probed once per request, because the answer depends on the prompt alone: any lane may
    // use it, and the restore re-checks it before trusting a byte.
    std::uint32_t disk_restore_frontier = 0;
};

template <>
struct RequestPlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    NINFER_QWEN36_RUNTIME_NS::ReusePath reuse = NINFER_QWEN36_RUNTIME_NS::ReusePath::FullReset;
    std::uint32_t reuse_base                  = 0;
    NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode mtp_bridge =
        NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode::None;
    bool prepare_mtp = false;
    std::optional<NINFER_QWEN36_RUNTIME_NS::VisionPrefillPlan> vision;
    NINFER_QWEN36_RUNTIME_NS::RewriteCheckpointAction rewrite_checkpoint_action =
        NINFER_QWEN36_RUNTIME_NS::RewriteCheckpointAction::Drop;
    std::optional<qwen3_6::RewriteCheckpointSpec> rewrite_checkpoint_capture;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    // Set when this lane's prefix comes from the NVMe tier instead of from the lane itself: the
    // lane starts empty and the restore fills it, falling back to a cold prefill on a miss.
    std::uint32_t disk_restore_frontier = 0;
    // Set when this lane continues a SIBLING's retained prefix rather than one of its own: the lane
    // whose device pages this plan maps read-only, instead of recomputing the same prompt. Those
    // pages stay charged to the donor's entitlement, so `*_page_entitlement` above covers only the
    // region this lane owns, and the cold footprint the plan can still fall back to is the one in
    // `summary.admission`.
    std::optional<std::uint32_t> adopt_lane;
    // Set when this lane continues a chain the shared index owns on its own: the node whose page
    // groups it maps read-only and whose state image it restores. Unlike `adopt_lane`, no lane has
    // to still hold the prefix, so the claim survives the conversation that produced it.
    std::optional<runtime::prefix_cache::BlockRef> adopt_node;
    // The service quanta the same request would need to prefill the whole prompt, which is what the
    // base plan priced and what an adoption that loses its lender falls back to. Adopting is a pure
    // gain over that plan, so the fallback has to charge it again rather than the suffix it priced.
    std::uint64_t cold_service_work_quanta = 0;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using RequestPlanImpl     = qwen3_6::detail::RequestPlanImpl<Variant>;
using RequestBasePlanImpl = qwen3_6::detail::RequestBasePlanImpl<Variant>;

enum class PendingKind : std::uint8_t {
    None,
    Begin,
    Ordinary,
    Speculative,
};

struct PendingCandidate {
    PendingKind kind            = PendingKind::None;
    std::uint32_t base_E        = 0;
    std::uint32_t base_S        = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t produced      = 0;
};

enum class Lifecycle : std::uint8_t {
    Empty,
    Prefilling,
    Active,
    Pending,
    Complete,
};

struct RewriteCheckpoint {
    bool valid                 = false;
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::uint32_t frontier     = 0;
};

struct SequenceKVBundle {
    PagedKVAllocation text;
    std::optional<PagedKVAllocation> backend;
    // Rank 1's allocation in ITS OWN text KV pool at tp == 2. The two pools have identical page
    // geometry (only the per-page byte count differs, because rank 1 holds 2 of the 4 KV heads),
    // and every pool operation below is issued on both in the same order, so the two allocations
    // hold the same page ids and publish identical block tables.
    std::optional<PagedKVAllocation> text_peer;
    // Rank 1's allocation in ITS OWN MTP (backend) KV pool at tp == 2. Same lockstep argument as
    // `text_peer`: identical page geometry, every pool operation issued on both in the same order.
    std::optional<PagedKVAllocation> backend_peer;
};

// Everything a parked lane must remember that already lives on the host.
struct LaneStateMetadata {
    std::vector<TokenId> ledger;
    qwen3_6::detail::ResidentPrefixIdentity prefix_identity;
    // The frozen digest image, so a spill derives its keys from the snapshot rather than from a
    // live sequence that a later admission may already have rewound.
    qwen3_6::detail::PrefixDigests prefix_digests;
    std::uint32_t execution_frontier      = 0;
    std::uint32_t ledger_frontier         = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::uint32_t mtp_draft_count         = 0;
    std::int32_t rope_delta               = 0;
    bool tail_hidden_valid                = false;
    RewriteCheckpoint rewrite_checkpoint;
};

// One lane's parked bytes, both ranks. Move-only: the images return their extents to the Program's
// HostLaneStateStore when this is destroyed or reset.
struct LaneStateImage {
    std::optional<HostKVImage> text;
    std::optional<HostKVImage> backend;
    std::optional<HostKVImage> text_peer;
    std::optional<HostKVImage> backend_peer;
    std::optional<HostLinearStateImage> linear;
    std::optional<HostLinearStateImage> linear_peer;
    LaneStateMetadata metadata;
};

// One node's self-contained continuation: the GDN state of every rank, rank 0's two hidden rows and
// the prefix metadata a claim re-checks. The KV pages are NOT here -- the store owns those in device
// memory -- so this is a state image, not a lane image, and it is what lets a prefix outlive the
// lane that produced it.
struct SharedPayloadImage {
    LaneStateMetadata metadata;
    std::optional<HostLinearStateImage> linear;      // rank 0, both slot roles
    std::optional<HostLinearStateImage> linear_peer; // rank 1
    // Rank 0's hidden rows: the tail row then the rewrite-checkpoint row, `hidden_bytes` each.
    std::vector<std::byte> hidden;
    // The chain's KV, one image per pool and rank, when the payload has a host budget to park into.
    // Empty means the store holds the device page groups instead; either way a claim restores the
    // same bytes.
    std::optional<HostKVImage> text;
    std::optional<HostKVImage> text_peer;
    std::optional<HostKVImage> backend;
    std::optional<HostKVImage> backend_peer;
    std::uint64_t last_use = 0;
};

// Page-group slots of one shared block: Main Text on rank 0 and rank 1, then the speculative backend
// on both. The store moves ids; these names say which pool has to hold them.
inline constexpr std::uint32_t kSharedSlotText        = 0;
inline constexpr std::uint32_t kSharedSlotTextPeer    = 1;
inline constexpr std::uint32_t kSharedSlotBackend     = 2;
inline constexpr std::uint32_t kSharedSlotBackendPeer = 3;

struct DecodeGraphProfile {
    std::uint32_t batch_size             = 1;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class         = 0;
    DecodeGraphDefinition definition;
};

struct DecodeGraphTopology {
    std::uint32_t topology_class = 0;
    DecodeGraphExecutable executable;
    std::optional<std::size_t> installed_profile;
};

struct DecodeGraphFamily {
    std::vector<DecodeGraphProfile> profiles;
    std::vector<DecodeGraphTopology> topologies;
};

// Target model continuation for one logical sequence. This state remains meaningful after the
// request which produced it has finished, so it is deliberately separate from request lifecycle,
// output, sampling, and round-control state.
struct SequenceState {
    std::optional<SequenceKVBundle> kv;
    Tensor tail_hidden;
    Tensor rewrite_checkpoint_hidden;
    std::uint32_t lane = 0;

    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::vector<TokenId> ledger;
    qwen3_6::detail::ResidentPrefixIdentity prefix_identity;
    // Rolling content digest of every token frontier this sequence has committed, in lockstep with
    // `ledger` and `prefix_identity`. It is what the disk tier keys a block by, so the spill and a
    // later lookup derive the same identity for the same content.
    qwen3_6::detail::PrefixDigests prefix_digests;
    std::int32_t rope_delta               = 0;
    // Largest frontier whose whole chain AND state image the NVMe tier already holds. Every tap
    // that publishes a prefix starts one block below it, which is why a completed turn costs the
    // blocks it just produced instead of the whole conversation again.
    std::uint32_t disk_published_frontier = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, qwen3_6::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    bool tail_hidden_valid        = false;
    bool retained                 = false;
    // The request that produced this chain licensed prefix reuse and carried no media, so the
    // chain may be published to the shared block index. A chain that may not stay resident for a
    // sibling must not appear there either: the index is what a later request looks its prompt up
    // in, and a promise from a chain nobody may continue is a false hit waiting to be verified.
    bool shareable = false;
    RewriteCheckpoint rewrite_checkpoint;
};

// Request/round control is not retained with a reusable SequenceState. A later concurrent Engine
// gives every occupied request slot its own instance of this state.
struct RequestControl {
    Lifecycle lifecycle = Lifecycle::Empty;
    PendingCandidate pending;
    ops::SamplingConfig sampling_host;
    GenerationTimings timings;
    SpeculativeStats speculative_stats;

    struct Prefill {
        PreparedPromptData prompt;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<schedule::VisionPrefillSession> vision;
        runtime::TransientRegion transient;
        std::optional<RewriteCheckpointSpec> rewrite_checkpoint_capture;
        std::uint32_t base               = 0;
        std::uint32_t cursor             = 0;
        std::uint32_t prompt_tokens      = 0;
        std::uint32_t initial_mtp_extent = 0;
        double elapsed_seconds           = 0.0;
        bool prepare_mtp                 = false;
        ReusePath reuse                  = ReusePath::FullReset;
        MtpBridgeMode mtp_bridge         = MtpBridgeMode::None;
    };

    std::optional<Prefill> prefill;
};

// Rank 1's complete runtime mirror: its own arenas, its own shard of the weights, its own halved
// decoder state, and its own RoundState. It owns no bookkeeping -- lanes, page accounting,
// sampling and the pinned host round buffers all live once, on rank 0.
struct PeerRuntime {
    PeerRuntime(DeviceContext& peer_device, const LoadedModelData& peer_model,
                const SequencePlanImpl& plan);

    PeerRuntime(const PeerRuntime&)            = delete;
    PeerRuntime& operator=(const PeerRuntime&) = delete;

    DeviceContext& device;
    const LoadedModelData& model;
    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3_6::DecoderState> decoder;
    // Rank 1's own GDN replay records: the speculative verify round records this device's own
    // head/channel shard and folds it here, so the two devices commit the same accepted prefix
    // from records neither ever exchanges.
    std::optional<GdnReplayRecords> replay_records;
    qwen3_6::RoundState io;
    std::optional<typename qwen3_6::detail::TP2DFlashExtension<Variant>::State> tp2_dflash;
    Tensor prefill_hidden;
    // Rank 1's OWN penalty counters. `ops::SamplingConfig::token_counts` is a raw device pointer,
    // and the MTP round's acceptance runs on both devices (see the replicated-accept note in
    // mtp_impl.h), so rank 1 must never be handed rank 0's. It is not a cache: rank 0 remains the
    // source of truth (ProgramImplCore::install_sampling zeroes both, and the one increment that
    // happens on rank 0 alone -- prefill's bonus token -- is copied across before the first decode
    // round), and thereafter both are advanced by the same Op over bit-identical inputs.
    Tensor token_counts;
};

namespace ninfer::targets::qwen3_6::detail {
class HostLaneStateStore;
} // namespace ninfer::targets::qwen3_6::detail

class ProgramImplCore {
public:
    ProgramImplCore(const LoadedModelData& model, const LoadedModelData* peer_model,
                    const SequencePlanImpl& plan, ExecutionContext& execution);
    ~ProgramImplCore() noexcept;

    [[nodiscard]] RequestBasePlan
    plan_request_base(const PreparedPromptData& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan plan_request_for_lane(std::uint32_t lane,
                                                    const PreparedPromptData& prompt,
                                                    const RequestBasePlan& base);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const RequestPlan& plan) const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] std::optional<std::uint32_t>
    plan_adopted_lane(const RequestPlan& plan) const noexcept;
    // The order in which retained lanes may give way to make room for an admission: the shortest
    // resumable prefix first, least recently retained on a tie. A retained lane is a conversation
    // the process is holding, and dropping the longest one costs the most to rebuild, so capacity
    // pressure must not walk lane numbers and take whatever sits lowest.
    [[nodiscard]] std::vector<std::uint32_t> retained_lane_eviction_order() const;
    [[nodiscard]] SharedPrefixStats shared_prefix_stats() const;
    // Drops the least recently used lane-free payloads, giving their page groups and image slots
    // back. Capacity pressure prefers this to eating a retained conversation: a payload is an
    // opportunistic copy of a prefix, while a retained lane is a conversation the process holds.
    std::uint32_t drop_shared_payloads(std::uint32_t count) noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                PreparedPromptData&& prompt,
                                                                RequestPlan&& plan,
                                                                runtime::TransientRegion transient);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled);
    void abort_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    void evict_retained_lane(std::uint32_t lane) noexcept;

    // Host tier. `enable_lane_tier` sizes and allocates the pinned arenas once; until it runs the
    // Program behaves exactly as before, and eviction discards the lane instead of parking it.
    void enable_lane_tier(std::size_t host_bytes);
    [[nodiscard]] bool lane_tier_enabled() const noexcept { return lane_store_ != nullptr; }
    [[nodiscard]] bool has_parked_lane(std::uint32_t lane) const noexcept;
    // Parks `lane`'s complete continuable state into pinned host RAM and releases its device KV
    // pages. Returns false when the lane is not retained, already parked, or the tier cannot fit it.
    [[nodiscard]] bool park_retained_lane(std::uint32_t lane);
    // Rebuilds `lane`'s device state from its parked image. Returns false when nothing is parked.
    [[nodiscard]] bool restore_parked_lane(std::uint32_t lane);
    [[nodiscard]] std::size_t parked_host_bytes() const noexcept;
    [[nodiscard]] std::size_t tier_device_kv_bytes() const noexcept;

    // NVMe tier. `enable_disk_tier` opens the content-addressed block store once at construction;
    // it requires the host tier, whose arenas hold the staging and the hidden rows a restore needs.
    // The tier is layout-neutral: it stores bytes the other tiers shed, and every lookup is decided
    // by content, so it never changes what a plan computes.
    void enable_disk_tier(std::string path, std::size_t bytes);
    [[nodiscard]] bool disk_tier_enabled() const noexcept { return disk_tier_ != nullptr; }
    // Deepest frontier of `prompt`'s own prefix the disk tier can resume, or 0. Consulted once per
    // request, so a prompt can address a prefix no lane ever produced.
    [[nodiscard]] std::uint32_t disk_restorable_frontier(const PreparedPromptData& prompt) const;
    [[nodiscard]] std::size_t disk_tier_capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t disk_tier_used_bytes() const noexcept;
    // Resumes that landed from the NVMe tier, cumulative since the Program opened it.
    [[nodiscard]] std::uint64_t disk_tier_restores() const noexcept { return disk_restores_; }
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;

    void reset_memory_peaks() noexcept;

    const LoadedModelData& model;
    ExecutionContext& execution;
    DeviceContext& device;
    const int tp;
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    const SpeculativeBackend speculative_backend;
    const DType kv_dtype;
    const std::int32_t kv_quant_group;
    const ProposalHead proposal_head;
    const bool vision_enabled;
    const bool use_cuda_graph;
    const std::size_t kv_payload_bytes;
    const std::size_t gdn_state_bytes;
    const std::size_t graph_allowance_bytes;
    // Measured graph residency per device (index = rank). At tp1 only [0] is populated. At tp2 the
    // single cross-device graph materializes driver state on BOTH devices, and each is checked
    // against the SAME per-device allowance -- graph_allowance_bytes is a per-device budget, like
    // every other field in device_reservation_bytes.
    std::array<std::size_t, 2> graph_observed_bytes{0, 0};
    // Node count of ONE captured decode graph (the first profile of the captured family). At tp2
    // one graph holds both devices' nodes, so this is the direct measurement of whether the peer's
    // half of the schedule was captured rather than left out.
    std::size_t graph_node_count = 0;
    const WorkspacePlan workspace_plan;

    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    // YaRN rotary state, resolved once at construction.
    //
    // `rope_frequency_storage[rank]` is a dedicated 32-float device allocation on rank `rank`'s OWN
    // device, made once here and never touched again: CUDA Graph capture bakes the pointer into the
    // replayed rope launch node, so it must outlive every replay, and at tp 2 each rank ropes its
    // own head-local q/k on its own device, where a pointer into the other rank's allocation is not
    // addressable. It is deliberately NOT part of the planned persistent arena: under
    // `RopeMode::Native` nothing is allocated at all, which is what keeps a native plan and its
    // memory summary byte-identical to the pre-YaRN engine.
    //
    // `rope_frequency[rank]` is the descriptor every text rope call site reads (through
    // `ExecutionCore::rope_frequency` -> `TextContext`); a null `inv_frequency` IS the native path.
    std::array<DeviceBuffer, 2> rope_frequency_storage;
    std::array<ops::RopeFrequencyOverride, 2> rope_frequency{};
    const RopeMode rope_mode;
    const std::uint32_t effective_max_context;
    const double yarn_mscale;
    std::optional<PeerRuntime> peer;
    std::optional<ops::PeerEvents> peer_events;
    // Created once at tp2 when graphs are on; forks rank 1's stream into rank 0's capture.
    std::optional<DecodeGraphPeerBridge> graph_bridge;
    std::optional<schedule::TpPeerCore> peer_core;
    std::unique_ptr<qwen3_6::DecoderState> decoder;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<DFlashPersistentState> dflash;
    std::optional<typename qwen3_6::detail::TP2DFlashExtension<Variant>::State> tp2_dflash;
    qwen3_6::RoundState io;
    Tensor prefill_hidden;
    Tensor sampling_config;
    Tensor token_counts;
    Tensor tail_hidden_store;
    Tensor rewrite_checkpoint_hidden_store;

    std::array<SequenceState, kMaximumConcurrency> sequences;
    std::array<RequestControl, kMaximumConcurrency> requests;
    std::unique_ptr<qwen3_6::detail::HostLaneStateStore> lane_store_;
    std::array<std::optional<LaneStateImage>, kMaximumConcurrency> parked_images_;
    // The NVMe tier under the host tier, and the frontier of the last restore that came off it.
    std::unique_ptr<qwen3_6::detail::LaneDiskTier> disk_tier_;
    std::uint32_t disk_restores_ = 0;

    DecodeGraphFamily ordinary_graphs;
    DecodeGraphFamily mtp_graphs;
    DecodeGraphFamily dflash_graphs;
    DecodeGraphFamily dflash2_graphs;

    PinnedHostBuffer round_host;
    TokenId* host_tokens = nullptr;
    // Debug-only capture of the round's full-vocabulary logits, raw BF16 bits. OFF by default and
    // completely inert until enable_logits_capture(true) allocates the host buffer: with it off,
    // copy_round_logits() is a predictable branch and nothing is allocated or copied, so no
    // production path (tp1 or tp2) is changed. With it on, the buffer is overwritten every time a
    // token is sampled from io.logits at prefill finalization, for both tp1 and tp2 (io is rank
    // 0's window; logits_tp2 gathers into it the same way the tp1 leaf writes it), so no
    // tp-specific code is needed. Prefill only -- the decode path is CUDA-graph-capturable and an
    // unconditional memcpy inside a captured region is rejected at capture time; a probe that
    // requests exactly one output token completes on prefill's own sampled token and never runs a
    // decode round, which is what the parity harness (tools/tp2/parity.cpp) uses. Valid only
    // immediately after Engine::generate()/wait() returns to the calling thread for a single-lane
    // engine -- concurrent lanes would overwrite it out of order.
    std::vector<std::uint16_t> logits_capture;
    bool logits_capture_enabled = false;
    std::optional<PinnedHostBuffer> ordinary_host;
    qwen3_6::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    qwen3_6::OrdinaryDecodeEgress* ordinary_host_egress   = nullptr;
    // Rank 1's own pinned ordinary ingress: rank 0's record with every row's
    // `sampling[row].token_counts` nulled. Rank 1 mirrors the ordinary round for its half of the
    // weights and never samples (the output head is vocabulary-split and sampling belongs to rank
    // 0), so the sampling configs in its frame are inert -- but they must not carry a rank-0
    // DEVICE address into rank 1's frame: dereferencing one from rank 1 is a silent cross-device
    // fault. A separate pinned buffer rather than a patched copy at issue time, for the same
    // reason as `mtp_peer_host_ingress`: the upload is inside the captured decode graph, which
    // re-reads this exact host address at every replay.
    std::optional<PinnedHostBuffer> ordinary_peer_host;
    qwen3_6::OrdinaryDecodeIngress* ordinary_peer_host_ingress = nullptr;
    bool peer_egress_check_enabled     = false;
    std::uint64_t peer_egress_rounds     = 0;
    std::uint64_t peer_egress_mismatches = 0;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3_6::MtpDecodeIngress* mtp_host_ingress = nullptr;
    qwen3_6::MtpDecodeEgress* mtp_host_egress   = nullptr;
    // Rank 1's own pinned MTP ingress: byte-for-byte rank 0's record except that each row's
    // `sampling[row].token_counts` names rank 1's counter lane. It has to be a separate pinned
    // buffer rather than a patched copy made at issue time, because the ingress upload is inside
    // the captured decode graph and the graph re-reads this exact host address at every replay.
    std::optional<PinnedHostBuffer> mtp_peer_host;
    qwen3_6::MtpDecodeIngress* mtp_peer_host_ingress = nullptr;
    std::optional<PinnedHostBuffer> dflash_host;
    qwen3_6::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    qwen3_6::DFlashDecodeEgress* dflash_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> dflash2_host;
    qwen3_6::DFlash2DecodeIngress* dflash2_host_ingress = nullptr;
    qwen3_6::DFlash2DecodeEgress* dflash2_host_egress = nullptr;
    std::optional<PinnedHostBuffer> dflash2_peer_host;
    qwen3_6::DFlash2DecodeIngress* dflash2_peer_host_ingress = nullptr;

    std::size_t workspace_logical_peak_bytes = 0;

    // See logits_capture's comment. Read-only; the caller owns thread-safety (single-lane, read
    // after wait() returns). Empty while capture is disabled.
    [[nodiscard]] std::span<const std::uint16_t> last_round_logits_bf16() const noexcept {
        return logits_capture;
    }

    // Turns the debug capture above on or off. Enabling sizes the host buffer to the round's
    // logits row count; disabling releases it. Must not be called while a round is in flight (the
    // executor holds its execution mutex across this call).
    void enable_logits_capture(bool enabled);

    // Debug-only, OFF by default: after each MTP decode round at tp == 2, read rank 1's MTP egress
    // back and compare it field for field with rank 0's. The two ranks run the acceptance Op over
    // bit-identical inputs, so their egress records are argued to agree; enabling this turns that
    // induction into a measurement. Costs one ~1 KiB device-to-host copy plus a host compare per
    // round while enabled, and nothing at all while off. Counters are cumulative over the
    // Program's lifetime; a mismatch is counted (per row, per field) rather than thrown, so a test
    // can read the totals after a clean run.
    void enable_peer_egress_check(bool enabled) noexcept;
    [[nodiscard]] std::uint64_t peer_egress_check_rounds() const noexcept {
        return peer_egress_rounds;
    }
    [[nodiscard]] std::uint64_t peer_egress_check_mismatches() const noexcept {
        return peer_egress_mismatches;
    }

private:
    void clear_lane(SequenceState& sequence, RequestControl& request) noexcept;
    void ordered_reset(SequenceState& sequence);
    void prepare_graphs();
    // Copies rank 0's counter lane onto rank 1. Called once, after prefill's bonus token is
    // sampled -- the only counter increment that happens on rank 0 and not on rank 1.
    [[nodiscard]] static Tensor token_counts_lane(const Tensor& storage, std::uint32_t lane);
    void publish_peer_token_counts(const SequenceState& sequence);
    // Mirrors `mtp_host_ingress` into `mtp_peer_host_ingress`, swapping every row's counter
    // pointer for rank 1's. No-op at tp1 or without MTP.
    void publish_peer_mtp_ingress(std::span<const std::uint32_t> lanes);
    void publish_peer_dflash2_ingress(std::span<const std::uint32_t> lanes);
    // Mirrors `ordinary_host_ingress` into `ordinary_peer_host_ingress` with every row's counter
    // pointer nulled. No-op at tp1 or without an ordinary frame.
    void publish_peer_ordinary_ingress();
    // Debug-only: reads rank 1's MTP egress back and compares it, field for field, with rank 0's.
    // No-op unless the check is enabled and a peer exists.
    void check_peer_mtp_egress(std::size_t rows);
    void install_sampling(SequenceState& sequence, RequestControl& request,
                          const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    void set_peer_i32(Tensor& tensor, std::int32_t value);
    void copy_tail(SequenceState& sequence, const Tensor& source);
    void copy_round_token();
    void copy_round_logits();
    void resolve_non_speculative_pending(SequenceState& sequence, RequestControl& request,
                                         std::uint32_t accepted_tokens, bool terminal);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill(SequenceState& sequence,
                                                             RequestControl& request);
    void enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                       std::span<const std::uint32_t> starts,
                                       std::span<const std::uint32_t> counts);
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
    void mark_workspace_usage(std::size_t phase_bytes) noexcept;
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                          std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_mtp_batch(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash_batch(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash2_batch(std::span<const std::uint32_t> lanes,
                         std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] bool park_lane(std::uint32_t lane);
    void restore_lane(std::uint32_t lane);
    // Loads a content-addressed prefix the disk tier holds into `lane`, returning the frontier that
    // landed (0 when the record is no longer complete, which is a cold prefill).
    [[nodiscard]] std::uint32_t restore_lane_from_disk(std::uint32_t lane, std::uint32_t frontier,
                                                       const PreparedPromptData& prompt);
    // Writes one lane's whole prefix to the disk tier. `parked` sources it from the lane's host
    // image, `device` from the live device allocations. False leaves whatever the tier already
    // held, so a failed spill only costs a recompute later.
    [[nodiscard]] bool spill_parked_prefix(std::uint32_t lane) noexcept;
    [[nodiscard]] bool spill_device_prefix(std::uint32_t lane) noexcept;
    // Writes a single (frontier, state-role) tap from the lane's live device allocations, walking
    // its chain from `first_page` (0 for a prefix nothing has published yet).
    [[nodiscard]] bool spill_device_tap(std::uint32_t lane, std::uint32_t frontier, bool checkpoint,
                                        std::uint32_t first_page) noexcept;
    // Publishes the prefix of a lane that has just completed a request. A retained lane keeps its
    // device KV and is normally published when the next request displaces it, but that leaves the
    // conversation the process is holding when it exits with no record at all: the next process
    // would re-prefill exactly the session that was active last. Best effort, like every spill.
    void publish_retained_prefix(std::uint32_t lane) noexcept;
    // Packs one lane's resumable state image into the disk tier's staging buffer.
    [[nodiscard]] bool stage_tier_state(std::uint32_t lane, bool checkpoint) noexcept;
    [[nodiscard]] std::uint32_t program_identity_tag() const noexcept;

    // Deepest frontier one resident sequence can continue `prompt` from, by the rule a lane's own
    // plan has always used: the committed frontier when the target state and its append readiness
    // agree, otherwise the rewrite checkpoint its prefill captured. `resume` is the sequence the
    // plan would continue, which is a sibling lane's when this plan adopts a shared prefix.
    struct ResidentReuse {
        ReusePath path = ReusePath::FullReset;
        std::uint32_t base = 0;
    };
    [[nodiscard]] ResidentReuse deepest_resident_reuse(const SequenceState& resume,
                                                       const PreparedPromptData& prompt) const;
    // The frontier a lane that is STILL RUNNING its own request can hand to a sibling. Its live
    // frontier and its live GDN state move every round, so neither can be lent; what it can hand
    // over is the rewrite checkpoint its request resumed from. The pages below that frontier are
    // already written and a lane that only appends never touches them again, and its state image
    // sits in a slot the request keeps until a later capture -- which updates this metadata first,
    // so a claim that a re-capture invalidated fails the check instead of being trusted.
    [[nodiscard]] ResidentReuse checkpoint_claim(const SequenceState& sibling,
                                                 const PreparedPromptData& prompt) const;
    // Clones the donor's continuable state into `lane` up to `plan.reuse_base`: its KV pages are
    // mapped read-only below the frontier and the page the frontier sits inside is copied, and both
    // ranks' GDN state, hidden rows, ledger and identity come across. False leaves `lane` untouched,
    // so the plan falls back to the cold prefill it also priced.
    [[nodiscard]] bool adopt_lane_prefix(std::uint32_t lane, const PreparedPromptData& prompt,
                                         const RequestPlanImpl& plan);
    // Maps one plane of a donor's prefix into a freshly reserved allocation up to `frontier`: every
    // page the frontier fully covers is borrowed read-only, and the page the frontier sits inside is
    // copied into a page this allocation owns, because both lanes continue by writing inside it.
    void adopt_kv_plane(PagedKVPool& pool, PagedKVAllocation& destination,
                        const PagedKVAllocation& source, std::uint32_t frontier,
                        std::uint32_t owned_pages, cudaStream_t stream);
    // The same mapping from raw page ids instead of a source allocation, which is what a node
    // payload offers: `prefix` is the chain's page group per block and `tail` the page the frontier
    // sits inside (absent when the frontier ends on a block boundary).
    void adopt_kv_plane_ids(PagedKVPool& pool, PagedKVAllocation& destination,
                            std::span<const std::int32_t> prefix, std::int32_t tail,
                            std::uint32_t frontier, std::uint32_t owned_pages, cudaStream_t stream);

    // The offload that parks one pool slot's KV, or nullptr when the node KV budget is off.
    [[nodiscard]] HostKVOffload* shared_kv_offload_for(std::uint32_t slot) noexcept;
    // Parks the chain's KV into pinned host RAM, one image per pool and rank. False leaves the
    // images empty and the caller keeps the device page groups instead.
    [[nodiscard]] bool park_payload_kv(std::uint32_t lane, std::uint32_t frontier,
                                       SharedPayloadImage& image);

    // The pool behind a shared page slot, or nullptr when this profile has no such pool.
    [[nodiscard]] PagedKVPool* shared_page_pool(std::uint32_t slot) noexcept;
    [[nodiscard]] const PagedKVPool* shared_page_pool(std::uint32_t slot) const noexcept;
    // Records the page groups a lane's bundle maps for the chain ending at `tip`.
    void attach_chain_pages(runtime::prefix_cache::BlockRef tip, const SequenceKVBundle& bundle);

    // --- Shared block index (P2) -------------------------------------------------------------
    //
    // Every lane that holds a shareable prefix publishes its chain here, keyed by content. A lane
    // that cannot continue its own state then looks its prompt up in the tree and adopts the
    // deepest chain a sibling still serves, instead of scanning every lane and comparing every
    // resident prefix by hand. The promise a node carries is only a candidate: `lane_serves`
    // re-derives the continuation from the named lane's live state before a plan may use it.

    // Publishes `frontier` tokens of a lane's committed chain. A no-op unless the request that
    // produced the chain licensed reuse and this lane still maps its pages.
    void publish_lane_chain(std::uint32_t lane, std::span<const TokenId> tokens,
                            std::uint32_t frontier,
                            runtime::prefix_cache::SharedRole role, std::uint32_t kind);

    // Captures the lane's checkpoint state into a node payload before the lane is displaced, and
    // takes the holds that keep the chain's page groups alive. False leaves the index as it was, so
    // the prefix simply stops being shared and falls back to the cold or disk path it always had.
    [[nodiscard]] bool capture_node_payload(std::uint32_t lane);

    // Continues `lane` from a node payload: maps the chain's page groups read-only (copying the page
    // the frontier sits inside), restores the captured state into the lane's private slots and
    // rebuilds the sequence metadata. False leaves `lane` untouched for the cold fallback.
    [[nodiscard]] bool adopt_node_prefix(std::uint32_t lane, const PreparedPromptData& prompt,
                                         const RequestPlanImpl& plan);

    void release_shared_payload(std::uint32_t slot) noexcept;

    // Whether the payload's captured prefix is exactly the first `frontier` tokens of `prompt`,
    // identity axes included: the index matched whole blocks, this is the authoritative check.
    [[nodiscard]] bool shared_payload_matches(std::uint32_t slot,
                                              const PreparedPromptData& prompt,
                                              std::uint32_t frontier) const;
    [[nodiscard]] bool shared_payload_tail_hidden(std::uint32_t slot) const noexcept;
    [[nodiscard]] std::uint32_t shared_payload_mtp_frontier(std::uint32_t slot) const noexcept;
    [[nodiscard]] RewriteCheckpoint shared_payload_checkpoint(std::uint32_t slot) const noexcept;
    // Best-effort capture on a displacement path: a prefix that could not be captured simply stops
    // being shared, and nothing about admitting the replacement request changes.
    void try_capture_node_payload(std::uint32_t lane) noexcept;
    // Drops lane-free payloads until both pools can hand out the given page counts. Best effort: a
    // payload that cannot be dropped leaves the pool exactly as it was.
    void reclaim_payload_pages(std::uint32_t text_pages, std::uint32_t backend_pages) noexcept;

    // Forgets every promise a lane made. Called before its chain can be replaced, so a promise for
    // a prefix the lane no longer holds cannot outlive it.
    void forget_lane_chain(std::uint32_t lane) noexcept;


    // Content-addressed index of resident prefixes. Its nodes are promises from lanes, so it is
    // sized from the device pages those lanes can ever hold.
    std::unique_ptr<runtime::prefix_cache::SharedBlockStore> shared_blocks_;
    // State images for the nodes the index owns. The arena is the scarce resource -- one image per
    // restorable node, each the whole GDN state of both ranks -- so its capacity is what bounds how
    // many lane-free prefixes the process can hold, and the least recently used image is the one a
    // new capture evicts.
    std::unique_ptr<HostLinearStateArena> shared_linear_arena_;
    // The per-node KV host tier: one byte arena shared by the pool geometries, exactly like the lane
    // tier's, so the budget bounds every parked chain together.
    std::unique_ptr<HostKVArena> shared_kv_arena_;
    std::vector<std::unique_ptr<HostKVOffload>> shared_kv_offloads_;
    std::unique_ptr<PinnedHostBuffer> shared_hidden_;
    std::vector<std::optional<SharedPayloadImage>> shared_payloads_;
    // The node each image slot belongs to, so a drop can take the node's payload with it.
    std::vector<std::optional<runtime::prefix_cache::BlockRef>> shared_payload_nodes_;
    std::size_t shared_hidden_row_bytes_ = 0;
    // Page groups the store holds out of the pools on a payload's behalf, per pool family. They are
    // what makes the bytes outlive their lane, and what has to come back before a pool can promise
    // that capacity again.
    std::uint32_t shared_held_text_pages_    = 0;
    std::uint32_t shared_held_backend_pages_ = 0;
    // Monotonic stamp ordering owner promises and retained-lane eviction.
    std::uint64_t shared_epoch_ = 0;
    std::array<std::uint64_t, kMaximumConcurrency> retained_epoch_{};
    // Index usage counters: one lookup per plan that asks the tree where a prompt is resident, one
    // hit per plan the tree answered, and the blocks reclaimed when no lane served them any more.
    mutable std::uint64_t shared_lookups_ = 0;
    mutable std::uint64_t shared_hits_    = 0;
    mutable std::uint64_t shared_pruned_  = 0;

    void drop_parked_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] KVPageGeometry text_kv_geometry() const;
    [[nodiscard]] KVPageGeometry backend_kv_geometry() const;
    void reserve_sequence_kv(SequenceState& sequence, std::uint32_t text_pages,
                             std::uint32_t backend_pages);
    void resize_sequence_kv_entitlement(SequenceState& sequence, std::uint32_t text_pages,
                                        std::uint32_t backend_pages);
    void bind_sequence_kv(SequenceState& sequence);
    void unbind_sequence_kv(SequenceState& sequence) noexcept;
    void materialize_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                 std::uint32_t backend_tokens = 0);
    void trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                          std::uint32_t backend_tokens = 0);
    void release_sequence_growth_entitlement(SequenceState& sequence) noexcept;
    [[nodiscard]] qwen3_6::PagedKVCache* backend_kv_cache() noexcept;
    [[nodiscard]] const qwen3_6::PagedKVCache* backend_kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState& sequence) const noexcept;
    [[nodiscard]] qwen3_6::PagedKVCacheView text_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_6::PagedKVCacheView mtp_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_6::PagedKVCacheView mtp_kv_view_peer(const SequenceState& sequence) const;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
class ProgramImpl<NINFER_QWEN36_VARIANT> final : public NINFER_QWEN36_RUNTIME_NS::ProgramImplCore {
public:
    using NINFER_QWEN36_RUNTIME_NS::ProgramImplCore::ProgramImplCore;
};

} // namespace ninfer::targets::qwen3_6::detail
