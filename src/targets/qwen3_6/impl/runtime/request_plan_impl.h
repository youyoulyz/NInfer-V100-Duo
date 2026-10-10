#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include "targets/qwen3_6/impl/runtime/schedule.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

void validate_sampling(const ResolvedSamplingParameters& sampling) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }
}

ops::SamplingConfig translate_sampling(const ResolvedSamplingParameters& source) {
    ops::SamplingConfig out;
    out.temperature       = source.temperature;
    out.top_k             = source.top_k;
    out.top_p             = source.top_p;
    out.min_p             = source.min_p;
    out.presence_penalty  = source.presence_penalty;
    out.frequency_penalty = source.frequency_penalty;
    out.seed              = source.seed;
    out.token_counts      = nullptr;
    return out;
}

std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
    return 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::uint64_t projected_service_work(const runtime::RequestPlanSummary& summary,
                                     std::uint32_t reuse_base, std::uint32_t prefill_chunk,
                                     std::size_t prefill_splits) noexcept {
    const std::uint32_t suffix = summary.prompt_tokens - reuse_base;
    const std::uint64_t prefill_units =
        suffix == 0
            ? 1ULL
            : 1ULL + (static_cast<std::uint64_t>(suffix) - 1ULL) / prefill_chunk + prefill_splits;
    const std::uint64_t decode_units =
        summary.effective_output_tokens == 0 ? 0ULL : summary.effective_output_tokens - 1ULL;
    return prefill_units + decode_units;
}

} // namespace

RequestBasePlan
ProgramImplCore::plan_request_base(const PreparedPromptData& prompt,
                                   const runtime::ResolvedExecutionOptions& options) {
    if (prompt.token_ids.empty()) { throw std::invalid_argument("prompt must contain tokens"); }
    if (prompt.token_ids.size() > capacity) {
        throw std::invalid_argument("prompt exceeds configured context capacity");
    }
    if (prompt.token_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("prompt token count exceeds uint32");
    }
    for (const TokenId id : prompt.token_ids) {
        if (id < 0 || id >= TextConfig::token_domain) {
            throw std::invalid_argument("prompt contains token outside the 248077-token domain");
        }
    }
    if (prompt.token_types.size() != prompt.token_ids.size() ||
        prompt.positions.size() != 3ULL * prompt.token_ids.size()) {
        throw std::invalid_argument("prepared prompt token metadata has an invalid shape");
    }
    if (prompt.has_media() != !prompt.media_payloads.empty() ||
        prompt.media_payloads.size() != prompt.vision_items.size()) {
        throw std::invalid_argument("prepared prompt media payload is incomplete");
    }
    for (std::size_t i = 0; i < prompt.media_payloads.size(); ++i) {
        if (!prompt.media_payloads[i] ||
            prompt.media_payloads[i]->patch_elements !=
                prompt.vision_items[i].patch_count * kPreparedVisionPatchFeatures) {
            throw std::invalid_argument("prepared prompt media item payload has an invalid shape");
        }
    }
    if (prompt.has_media() && !vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }
    validate_sampling(options.sampling);

    auto base                             = std::make_unique<RequestBasePlanImpl>();
    base->summary.prompt_tokens           = static_cast<std::uint32_t>(prompt.token_ids.size());
    base->summary.requested_output_tokens = options.requested_output_tokens;
    const std::uint32_t capacity_output =
        capacity - base->summary.prompt_tokens + static_cast<std::uint32_t>(1);
    base->summary.effective_output_tokens =
        std::min(options.requested_output_tokens, capacity_output);
    base->summary.effective_limit_reason = options.requested_output_tokens <= capacity_output
                                               ? FinishReason::OutputLimit
                                               : FinishReason::ContextCapacity;
    base->summary.transient_alignment    = 1;
    base->summary.transient_bytes        = 0;
    base->sampling                       = translate_sampling(options.sampling);
    base->allow_prefix_reuse             = options.allow_prefix_reuse;
    const std::uint32_t reserved_context_tokens =
        base->summary.prompt_tokens + (base->summary.effective_output_tokens == 0
                                           ? 0U
                                           : base->summary.effective_output_tokens - 1U);
    base->text_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);
    if (speculative_backend == SpeculativeBackend::Mtp) {
        const std::uint32_t mtp_tokens    = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(reserved_context_tokens) + draft_window - 1ULL));
        base->backend_kv_page_entitlement = pages_for_tokens(mtp_tokens);
    } else if (speculative_backend == SpeculativeBackend::DFlash) {
        base->backend_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);
    }
    base->summary.admission = runtime::AdmissionResources{
        .active_lanes     = 1,
        .main_kv_pages    = base->text_kv_page_entitlement,
        .backend_kv_pages = base->backend_kv_page_entitlement,
    };
    if (prompt.has_media()) {
        auto control =
            std::make_shared<qwen3_6::VisionControl>(qwen3_6::build_vision_control(prompt));
        std::size_t max_merged     = 0;
        std::uint32_t previous_end = 0;
        for (const qwen3_6::VisionItemControl& item : control->items) {
            if (item.scatter_indices.empty()) {
                throw std::invalid_argument("vision item has no Text consumer columns");
            }
            const auto first = static_cast<std::uint32_t>(item.scatter_indices.front());
            const auto last  = static_cast<std::uint32_t>(item.scatter_indices.back());
            const std::uint32_t begin =
                speculative_backend == SpeculativeBackend::Mtp && first != 0 ? first - 1 : first;
            const std::uint32_t end = last + 1;
            if (begin < previous_end) {
                throw std::invalid_argument("vision item consumer spans overlap");
            }
            if (end > base->summary.prompt_tokens) {
                throw std::invalid_argument("vision item consumer span exceeds prompt");
            }
            if (schedule::VisionContext::workspace_bytes(item) > work.capacity()) {
                throw std::invalid_argument("vision item exceeds the Program workspace envelope");
            }
            previous_end = end;
            max_merged   = std::max(max_merged, item.merged_count);
        }
        base->vision_transient_bytes = schedule::VisionContext::output_transient_bytes(max_merged);
        base->vision_control         = std::move(control);
    }

    if (prompt.identity.rewrite_checkpoint) {
        const RewriteCheckpointSpec candidate = *prompt.identity.rewrite_checkpoint;
        if (candidate.frontier == 0 || candidate.frontier > base->summary.prompt_tokens) {
            throw std::invalid_argument(
                "rewrite checkpoint frontier must lie at or inside the prompt frontier");
        }
        base->rewrite_checkpoint = candidate;
    }
    const std::size_t cold_prefill_splits =
        (base->vision_control != nullptr ? base->vision_control->items.size() : 0ULL) +
        (base->rewrite_checkpoint &&
                 base->rewrite_checkpoint->frontier < base->summary.prompt_tokens
             ? 1ULL
             : 0ULL);
    base->summary.service_work_quanta =
        projected_service_work(base->summary, 0, prefill_chunk, cold_prefill_splits);

    // The NVMe tier is consulted once per request, against the prompt's own content digests: a
    // prefix another conversation spilled -- or this one spilled before a restart -- is addressed
    // by what it says, not by which lane happens to hold it. Vision prompts stay out of it: their
    // media would have to be re-encoded before a restored block could be trusted, and the plan
    // already prices a full encode.
    if (base->allow_prefix_reuse && disk_tier_ != nullptr && !prompt.has_media()) {
        base->disk_restore_frontier = disk_restorable_frontier(prompt);
    }
    return RequestBasePlan(std::move(base));
}

// The deepest frontier one resident sequence can continue `prompt` from. This is the rule a lane
// has always applied to its own state, lifted out so a sibling lane's state is priced identically:
// the committed frontier when the target backend's append readiness agrees, otherwise the rewrite
// checkpoint the prefill captured, which is the frontier a re-sent prompt resumes from.
ProgramImplCore::ResidentReuse
ProgramImplCore::deepest_resident_reuse(const SequenceState& resume,
                                        const PreparedPromptData& prompt) const {
    ResidentReuse best;
    if (!resume.retained) { return best; }
    const bool dflash_append_ready =
        speculative_backend != SpeculativeBackend::DFlash ||
        resume.dflash_context_frontier == resume.execution_frontier;
    if (resume.execution_frontier != 0 && dflash_append_ready &&
        qwen3_6::detail::prefix_matches(prompt, resume.ledger, resume.prefix_identity,
                                        resume.execution_frontier)) {
        best.path = ReusePath::AppendAtFrontier;
        best.base = resume.execution_frontier;
        return best;
    }
    if (resume.rewrite_checkpoint.valid && resume.rewrite_checkpoint.frontier != 0 &&
        resume.rewrite_checkpoint.frontier <= prompt.token_ids.size() &&
        qwen3_6::detail::prefix_matches(prompt, resume.ledger, resume.prefix_identity,
                                        resume.rewrite_checkpoint.frontier)) {
        best.path = restore_path(resume.rewrite_checkpoint.kind);
        best.base = resume.rewrite_checkpoint.frontier;
    }
    return best;
}

// The frontier a RUNNING lane can hand to a sibling: the rewrite checkpoint its request resumed
// from, and nothing else. Its live frontier, its live pages above that frontier and its `current`
// GDN slot all move every round, but a lane that only appends never writes below the frontier it
// resumed from, and the checkpoint's own state slot is left alone until a later capture replaces
// it -- which lands in this metadata first, so the re-verification at admission sees it.
ProgramImplCore::ResidentReuse
ProgramImplCore::checkpoint_claim(const SequenceState& sibling,
                                  const PreparedPromptData& prompt) const {
    ResidentReuse best;
    const RewriteCheckpoint& checkpoint = sibling.rewrite_checkpoint;
    if (!sibling.kv || !checkpoint.valid || checkpoint.frontier == 0 ||
        checkpoint.frontier > prompt.token_ids.size() ||
        checkpoint.frontier > sibling.text_kv_valid) {
        return best;
    }
    if (speculative_backend == SpeculativeBackend::Mtp &&
        sibling.mtp_kv_valid + 1 < checkpoint.frontier) {
        return best;
    }
    if (!qwen3_6::detail::prefix_matches(prompt, sibling.ledger, sibling.prefix_identity,
                                         checkpoint.frontier)) {
        return best;
    }
    best.path = restore_path(checkpoint.kind);
    best.base = checkpoint.frontier;
    return best;
}

RequestPlan ProgramImplCore::plan_request_for_lane(std::uint32_t lane,
                                                   const PreparedPromptData& prompt,
                                                   const RequestBasePlan& base_plan) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    const RequestControl& request = requests[lane];
    const SequenceState& sequence = sequences[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("cannot plan a request while Program is active or pending");
    }
    if (base_plan.impl_ == nullptr) { throw std::logic_error("request base plan is empty"); }
    const RequestBasePlanImpl& base = *base_plan.impl_;

    auto plan                         = std::make_unique<RequestPlanImpl>();
    plan->summary                     = base.summary;
    plan->sampling                    = base.sampling;
    plan->text_kv_page_entitlement    = base.text_kv_page_entitlement;
    plan->backend_kv_page_entitlement = base.backend_kv_page_entitlement;
    plan->cold_service_work_quanta    = base.summary.service_work_quanta;

    if (base.allow_prefix_reuse && prompt.identity.reusable && sequence.retained) {
        const ResidentReuse own = deepest_resident_reuse(sequence, prompt);
        plan->reuse             = own.path;
        plan->reuse_base        = own.base;
    }

    // Nothing of this lane's own continues the prompt. A sibling lane may still hold it: a caller
    // that repeats one long prompt reaches the pool through whichever lane the last request ran on,
    // so the bytes are there while that lane lives -- idle after its request, or still draining the
    // one that is using it. Adopting them maps the sibling's pages read-only and copies its state
    // instead of paying the same prefill again, which is the whole point of the shared prefix.
    //
    // The shared block index is how that sibling is found. Every lane publishes the chains it holds
    // when it commits them, and a prompt is answered with the deepest chain of its own that any
    // lane still serves -- content addressing instead of scanning every resident prefix. The claim
    // itself is unchanged: a prefix whose KV pages are already written and whose complete
    // linear-attention snapshot exists. For an idle lane that is its resident frontier or the
    // rewrite checkpoint its last turn captured; for a lane still running a request, only the
    // checkpoint qualifies -- its live frontier and its live state move every round, while the
    // checkpoint slot its request resumed from is left alone until that request finishes (and a
    // re-capture, which does move it, updates the metadata the claim is re-verified against).
    //
    // Vision stays out of it: an adopted plan drops the media spans it no longer has to encode, and
    // a fallback that has to encode them again would need to rebuild that planning step. DFlash
    // keeps per-lane context outside the KV bundle, which this path does not clone.
    if (base.allow_prefix_reuse && prompt.identity.reusable && !prompt.has_media() &&
        plan->reuse == ReusePath::FullReset &&
        speculative_backend != SpeculativeBackend::DFlash &&
        speculative_backend != SpeculativeBackend::DFlash2 && shared_blocks_ != nullptr) {
        // The content index holds a promise for every block of every shareable chain a lane has
        // published. Asking it for this prompt's chain names the lanes that can serve it -- and only
        // those -- instead of grading every resident prefix in the pool. What comes back is a
        // candidate, never an answer: the continuation is re-derived from the named lane's live
        // state below, exactly as the lane's own plan always has.
        //
        // That check compares whole prefixes, so it is memoized per lane; the walk asks about the
        // same handful of lanes at every block of the chain.
        const std::vector<std::uint64_t> hashes =
            runtime::prefix_cache::block_lookup_hashes(prompt.token_ids, {});
        std::array<ResidentReuse, kMaximumConcurrency> candidate_reuse{};
        std::array<std::uint8_t, kMaximumConcurrency> candidate_state{};
        const auto evaluate = [&](std::uint32_t other) -> const ResidentReuse& {
            static const ResidentReuse kNotACandidate{};
            if (other >= max_concurrency) { return kNotACandidate; }
            if (candidate_state[other] != 0) { return candidate_reuse[other]; }
            candidate_state[other] = 3; // "checked, not a candidate" until proven otherwise
            if (other == lane || !sequences[other].kv) { return candidate_reuse[other]; }
            const bool idle = sequences[other].retained;
            const bool busy = requests[other].lifecycle == Lifecycle::Prefilling ||
                              requests[other].lifecycle == Lifecycle::Active ||
                              requests[other].lifecycle == Lifecycle::Pending;
            if (idle) {
                candidate_reuse[other] = deepest_resident_reuse(sequences[other], prompt);
            } else if (busy) {
                candidate_reuse[other] = checkpoint_claim(sequences[other], prompt);
            }
            if (candidate_reuse[other].path != ReusePath::FullReset) {
                candidate_state[other] = idle ? 1U : 2U;
            }
            return candidate_reuse[other];
        };
        ++shared_lookups_;
        const runtime::prefix_cache::SharedBlockLookup lookup = shared_blocks_->lookup(
            prompt.token_ids, hashes, {},
            [&evaluate](const runtime::prefix_cache::SharedBlockOwner& owner,
                        std::uint32_t tokens) {
                const ResidentReuse& reuse = evaluate(owner.lane);
                return reuse.path != ReusePath::FullReset && reuse.base >= tokens;
            });
        bool have           = false;
        bool best_idle      = false;
        bool best_owner     = false;
        bool from_payload   = false;
        std::uint32_t donor = 0;
        ResidentReuse best;
        if (lookup.owner.has_value()) {
            ++shared_hits_;
            const runtime::prefix_cache::SharedBlockHit& hit = *lookup.owner;
            for (std::uint32_t index = 0; index < hit.owner_count; ++index) {
                const std::uint32_t other = hit.owners[index].lane;
                if (other >= max_concurrency || other == lane || !sequences[other].kv) { continue; }
                const SequenceState& sibling = sequences[other];
                const ResidentReuse reuse    = evaluate(other);
                if (reuse.path == ReusePath::FullReset || reuse.base < hit.tokens) { continue; }
                // Deepest wins; on a tie an idle lender beats a running one (the claim does not
                // have to share a lane that is writing), and a lender that owns its pages beats one
                // that is itself borrowing them, so claims do not chain.
                const bool idle  = candidate_state[other] == 1U;
                const bool owner = !sibling.kv->text.borrowed();
                const bool better =
                    !have || reuse.base > best.base ||
                    (reuse.base == best.base &&
                     (idle != best_idle ? idle : (owner && !best_owner)));
                if (!better) { continue; }
                have        = true;
                best        = reuse;
                donor       = other;
                best_idle   = idle;
                best_owner  = owner;
            }
        }
        // The other way the index can answer: a node payload, where the store owns the page groups
        // and a captured state image restores the continuation, so no lane has to hold the prefix at
        // all. It is only worth taking when it reaches deeper than a live lane -- a lane is cheaper
        // and fresher -- and only when the captured prefix really is this prompt's, which the tree's
        // whole-block match cannot decide on its own.
        if (lookup.payload.has_value()) {
            const runtime::prefix_cache::SharedBlockPayloadView& view = *lookup.payload;
            const std::uint32_t frontier = view.payload.frontier;
            if (view.payload.role == runtime::prefix_cache::SharedRole::Checkpoint &&
                frontier != 0 && frontier <= prompt.token_ids.size() && !have &&
                frontier >= plan->disk_restore_frontier &&
                shared_payload_matches(view.payload.image, prompt, frontier)) {
                ++shared_hits_;
                have         = true;
                from_payload = true;
                best         = ResidentReuse{
                    .path = restore_path(static_cast<RewriteCheckpointKind>(view.payload.kind)),
                    .base = frontier};
            }
        }
        // A resident claim is worth taking only when it reaches at least as deep as the NVMe record
        // this plan may also be carrying: the bytes are already in device memory, so a tie goes to
        // the resident one, and a shallower one would trade a longer reuse for a shorter one. Taking
        // it makes the plan a resident plan, so the disk record it was built on stops being one.
        if (have && best.base >= plan->disk_restore_frontier) {
            plan->reuse                 = best.path;
            plan->reuse_base            = best.base;
            plan->adopt_lane            = from_payload ? std::nullopt
                                                       : std::optional<std::uint32_t>(donor);
            plan->adopt_node            = from_payload
                                              ? std::optional<runtime::prefix_cache::BlockRef>(
                                                    lookup.payload->node)
                                              : std::nullopt;
            plan->disk_restore_frontier = 0;
        }
    }

    // Everything below is a property of the state the plan continues, whichever holder serves it:
    // the lane this plan adopts from, a node payload that outlived its lane, or this lane's own
    // state. A payload has no sequence left to read, so its captured image stands in for one.
    const bool adopting = plan->adopt_lane.has_value() || plan->adopt_node.has_value();
    const SequenceState& resume =
        plan->adopt_lane.has_value() ? sequences[*plan->adopt_lane] : sequence;
    bool resume_tail_hidden_valid    = resume.tail_hidden_valid;
    std::uint32_t resume_mtp_kv_valid = resume.mtp_kv_valid;
    RewriteCheckpoint resume_checkpoint = resume.rewrite_checkpoint;
    if (plan->adopt_node.has_value()) {
        const std::uint32_t image = shared_blocks_->payload(*plan->adopt_node).image;
        resume_tail_hidden_valid = shared_payload_tail_hidden(image);
        resume_mtp_kv_valid      = shared_payload_mtp_frontier(image);
        resume_checkpoint        = shared_payload_checkpoint(image);
    }
    // Whether the first `count` tokens of the prompt really continue the plan's reuse point, identity
    // axes included. A payload answers from its image; a lane from its live ledger.
    const auto prefix_serves = [&](std::uint32_t count) {
        if (plan->adopt_node.has_value()) {
            return shared_payload_matches(shared_blocks_->payload(*plan->adopt_node).image, prompt,
                                          count);
        }
        return qwen3_6::detail::prefix_matches(prompt, resume.ledger, resume.prefix_identity, count);
    };

    if (speculative_backend == SpeculativeBackend::Mtp) {
        const bool append_ready =
            plan->reuse == ReusePath::AppendAtFrontier && resume_tail_hidden_valid &&
            decoder->mtp_cache() != nullptr &&
            (plan->reuse_base == 0 || resume_mtp_kv_valid >= plan->reuse_base - 1);
        const bool checkpoint_ready = is_rewrite_checkpoint_restore(plan->reuse) &&
                                      decoder->mtp_cache() != nullptr && plan->reuse_base != 0 &&
                                      resume_mtp_kv_valid >= plan->reuse_base - 1;
        if (plan->reuse != ReusePath::FullReset && !append_ready && !checkpoint_ready) {
            plan->reuse      = ReusePath::FullReset;
            plan->reuse_base = 0;
        }
    }

    if (is_rewrite_checkpoint_restore(plan->reuse) &&
        speculative_backend == SpeculativeBackend::DFlash &&
        (!dflash || !resume.kv || !resume.kv->backend ||
         resume.dflash_context_frontier < plan->reuse_base)) {
        plan->reuse      = ReusePath::FullReset;
        plan->reuse_base = 0;
    }

    // A plan that lost its reuse -- the MTP gate above, or the DFlash one -- is no longer an
    // adoption, and nothing below may keep charging it as one.
    if (plan->reuse == ReusePath::FullReset &&
        (plan->adopt_lane.has_value() || plan->adopt_node.has_value())) {
        plan->adopt_lane.reset();
        plan->adopt_node.reset();
        plan->reuse_base = 0;
    }

    // The donor's pages are already covered by its own entitlement, so this lane reserves only the
    // region above `reuse_base` it will actually own, and the pages below that frontier are mapped
    // read-only. Charging the borrowed pages again would double-count the same physical memory,
    // which is what makes an N-lane shared prefix fit where N cold reservations would not.
    // A payload whose KV lives in host RAM does not borrow anything: the claim restores every page
    // into the lane's own allocation, so its entitlement is the cold footprint unchanged.
    const bool payload_owns_pages =
        plan->adopt_node.has_value() && shared_blocks_->payload(*plan->adopt_node).kv_in_host();
    if ((plan->adopt_lane.has_value() || plan->adopt_node.has_value()) && !payload_owns_pages) {
        // The MTP bridge continues one token behind the text frontier, exactly as its own lane
        // would: the backend plane is adopted to the same rule, one token lower.
        const std::uint32_t backend_frontier =
            plan->reuse_base == 0 ? 0 : plan->reuse_base - 1;
        const std::uint32_t adopted_text_pages =
            plan->reuse_base / static_cast<std::uint32_t>(kPagedKVPageSize);
        const std::uint32_t adopted_backend_pages =
            speculative_backend == SpeculativeBackend::Mtp
                ? backend_frontier / static_cast<std::uint32_t>(kPagedKVPageSize)
                : 0;
        const std::uint32_t owned_text =
            plan->text_kv_page_entitlement - adopted_text_pages;
        const std::uint32_t owned_backend =
            plan->backend_kv_page_entitlement - adopted_backend_pages;
        const bool owns_nothing =
            owned_text == 0 ||
            (speculative_backend == SpeculativeBackend::Mtp && owned_backend == 0);
        if (owns_nothing) {
            // The adopted prefix already covers the lane's whole reservation, so the lane would own
            // no page at all -- and a mapping with no entitlement of its own cannot grow. There is
            // nothing to gain over the cold start the base plan priced, so take that one.
            plan->adopt_lane.reset();
            plan->adopt_node.reset();
            plan->reuse      = ReusePath::FullReset;
            plan->reuse_base = 0;
        } else {
            plan->text_kv_page_entitlement    = owned_text;
            plan->backend_kv_page_entitlement = owned_backend;
        }
    }

    // Nothing resident matches, but the prompt's own content may still be on the NVMe tier. The
    // lane is then rebuilt from that record: `disk_restore_frontier` marks it, the reuse base is
    // the frontier the record closes, and the MTP bridge continues exactly where an in-place
    // AppendAtFrontier would. A miss at restore time also lands on the cold prefill this plan
    // priced, so the frontier only ever removes work.
    if (plan->reuse == ReusePath::FullReset && base.disk_restore_frontier != 0) {
        plan->disk_restore_frontier = base.disk_restore_frontier;
        plan->reuse                 = ReusePath::AppendAtFrontier;
        plan->reuse_base            = base.disk_restore_frontier;
        if (speculative_backend == SpeculativeBackend::Mtp) {
            plan->prepare_mtp = true;
            plan->mtp_bridge  = plan->reuse_base < plan->summary.prompt_tokens
                                    ? MtpBridgeMode::BeforeSuffix
                                    : MtpBridgeMode::AfterExactHit;
        }
    }

    const std::optional<RewriteCheckpointSpec>& desired = base.rewrite_checkpoint;
    // A disk-restorable plan carries no resident checkpoint at all, so an existing one can never
    // match it: the suffix prefill captures the desired boundary, or the plan defers exactly as a
    // resident hit past that boundary already does.
    const bool existing_checkpoint_matches =
        desired && plan->reuse != ReusePath::FullReset && plan->disk_restore_frontier == 0 &&
        resume_checkpoint.valid && resume_checkpoint.frontier == desired->frontier &&
        prefix_serves(desired->frontier);
    if (!desired) {
        plan->rewrite_checkpoint_action = RewriteCheckpointAction::Drop;
    } else if (existing_checkpoint_matches) {
        plan->rewrite_checkpoint_action = sequence.rewrite_checkpoint.kind == desired->kind
                                              ? RewriteCheckpointAction::KeepExisting
                                              : RewriteCheckpointAction::ReclassifyExisting;
    } else if (desired->frontier > plan->reuse_base) {
        plan->rewrite_checkpoint_action  = RewriteCheckpointAction::CaptureNew;
        plan->rewrite_checkpoint_capture = desired;
    } else {
        // The selected continuation state is already past the desired boundary. It remains a
        // valid hit; do not replay an otherwise reusable prefix merely to materialize an older
        // auxiliary snapshot. A later request can still use the checkpoint currently retained.
        plan->rewrite_checkpoint_action = RewriteCheckpointAction::DeferCapture;
    }

    // A disk-restorable plan reports no reusable prefix. The frontier it names is a record the store
    // may have lost, so the admission has to run like the prefill it can fall back to -- and it is
    // priced that way below -- while the lane that actually lands is whatever the record answers.
    plan->summary.reusable_prompt_tokens =
        plan->disk_restore_frontier != 0 ? 0U : plan->reuse_base;
    if (speculative_backend == SpeculativeBackend::Mtp) {
        if (plan->reuse == ReusePath::FullReset) {
            plan->prepare_mtp = true;
        } else if (plan->reuse == ReusePath::AppendAtFrontier) {
            plan->prepare_mtp = true;
            plan->mtp_bridge  = plan->reuse_base < plan->summary.prompt_tokens
                                    ? MtpBridgeMode::BeforeSuffix
                                    : MtpBridgeMode::AfterExactHit;
        } else if (is_rewrite_checkpoint_restore(plan->reuse)) {
            plan->prepare_mtp = true;
            plan->mtp_bridge  = plan->reuse_base < plan->summary.prompt_tokens
                                    ? MtpBridgeMode::BeforeSuffix
                                    : MtpBridgeMode::AfterExactHit;
        }
    }

    if (base.vision_control != nullptr) {
        VisionPrefillPlan vision;
        vision.control = base.vision_control;
        vision.uses.reserve(base.vision_control->items.size());
        for (std::size_t index = 0; index < base.vision_control->items.size(); ++index) {
            const qwen3_6::VisionItemControl& item = base.vision_control->items[index];
            const auto first          = static_cast<std::uint32_t>(item.scatter_indices.front());
            const auto last           = static_cast<std::uint32_t>(item.scatter_indices.back());
            const std::uint32_t begin = plan->prepare_mtp && first != 0 ? first - 1 : first;
            const std::uint32_t end   = last + 1;
            if (end <= plan->reuse_base) { continue; }
            vision.uses.push_back(VisionUseSpan{begin, end, static_cast<std::uint32_t>(index)});
        }
        if (!vision.uses.empty()) {
            plan->summary.transient_alignment = 256;
            plan->summary.transient_bytes     = base.vision_transient_bytes;
            plan->vision                      = std::move(vision);
        }
    }

    const std::size_t prefill_splits =
        (plan->vision ? plan->vision->uses.size() : 0ULL) +
        (plan->rewrite_checkpoint_capture &&
                 plan->rewrite_checkpoint_capture->frontier < plan->summary.prompt_tokens
             ? 1ULL
             : 0ULL);
    // A disk-restorable plan is priced as the cold prefill it falls back to: the restore is a
    // transfer, but a miss has to recompute the whole prompt, and only the caller that reads the
    // bytes knows which one happened.
    const std::uint32_t service_base = plan->disk_restore_frontier != 0 ? 0U : plan->reuse_base;
    plan->summary.service_work_quanta =
        projected_service_work(plan->summary, service_base, prefill_chunk, prefill_splits);
    return RequestPlan(std::move(plan));
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
