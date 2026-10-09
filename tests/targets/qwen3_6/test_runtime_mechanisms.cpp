#include "core/layout.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/hybrid_topology.h>
#include <ninfer/targets/qwen3_6/mtp_alignment.h>
#include <ninfer/targets/qwen3_6/round_state.h>
#include <ninfer/targets/qwen3_6/vision_control.h>

#include "runtime/prefix_cache/block_hash.h"
#include "targets/qwen3_6/impl/runtime/identity.h"
#include "targets/qwen3_6/impl/runtime/block_keys.h"
#include "targets/qwen3_6/impl/runtime/identity_tag.h"
#include "targets/qwen3_6/impl/runtime/prefix_digests.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

namespace q36 = ninfer::targets::qwen3_6;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

void test_topology() {
    static_assert(q36::kHybridAttentionInterval == 4);
    static_assert(q36::full_attention_layers(64) == 16);
    static_assert(q36::gdn_layers(64) == 48);
    for (std::int32_t layer = 0; layer < 64; ++layer) {
        expect(q36::is_full_attention_layer(layer) == ((layer + 1) % 4 == 0), "hybrid layer kind");
        if (q36::is_full_attention_layer(layer)) {
            expect(q36::full_attention_index(layer) == layer / 4, "full-attention index");
        } else {
            expect(q36::gdn_index(layer) == layer - layer / 4, "GDN index");
        }
    }
}

q36::DecoderStateSpec decoder_spec(ninfer::DType dtype, bool mtp) {
    return q36::DecoderStateSpec{
        .full_attention_layers     = 2,
        .mtp_layers                = 1,
        .capacity                  = 129,
        .kv_heads                  = 2,
        .attention_head_dim        = 64,
        .kv_dtype                  = dtype,
        .kv_quant_group            = dtype == ninfer::DType::I8 ? q36::kKvQuantGroup : 0,
        .enable_mtp                = mtp,
        .text_physical_page_groups = 5,
        .mtp_physical_page_groups  = mtp ? 4U : 0U,
        .linear_attention =
            {
                .layers         = 3,
                .conv_channels  = 10,
                .conv_width     = 3,
                .value_heads    = 4,
                .value_head_dim = 5,
                .key_head_dim   = 6,
                .slot_count     = 4,
                .conv_dtype     = ninfer::DType::BF16,
            },
    };
}

void test_decoder_layout() {
    ninfer::LayoutBuilder bf16_builder;
    const q36::DecoderStateLayout bf16 =
        q36::plan_decoder_state(bf16_builder, decoder_spec(ninfer::DType::BF16, false));
    (void)bf16_builder.finish(256);
    expect(bf16.text_kv.pool.planes.size() == 4, "BF16 Text KV has K/V planes per layer");
    expect(bf16.text_kv.pool.spec.page_group_count == 5 &&
               bf16.text_kv.pool.spec.logical_page_capacity == 3 &&
               bf16.text_kv.pool.spec.table_rows == 1,
           "Text KV separates five physical pages from three logical pages");
    expect(std::all_of(bf16.text_kv.pool.planes.begin(), bf16.text_kv.pool.planes.end(),
                       [](const ninfer::PagedKVPlaneLayout& plane) {
                           return plane.spec.dtype == ninfer::DType::BF16;
                       }),
           "BF16 KV has no scale planes");
    expect(!bf16.mtp_kv.has_value(), "disabled MTP omits KV storage");
    expect(bf16.linear_attention.conv.size() == 3 && bf16.linear_attention.recurrent.size() == 3,
           "Linear Attention layer storage");
    expect(bf16.linear_attention.spec.slot_count == 4, "Linear Attention slot geometry");
    expect(bf16.kv_payload_bytes() == bf16.text_kv.payload_bytes(), "BF16 KV payload accounting");

    ninfer::LayoutBuilder int8_builder;
    const q36::DecoderStateLayout int8 =
        q36::plan_decoder_state(int8_builder, decoder_spec(ninfer::DType::I8, true));
    (void)int8_builder.finish(256);
    expect(int8.text_kv.pool.planes.size() == 8 &&
               int8.text_kv.pool.planes[2].spec.dtype == ninfer::DType::FP16 &&
               int8.text_kv.pool.planes[3].spec.dtype == ninfer::DType::FP16,
           "INT8 Text KV has code and scale planes per layer");
    expect(int8.mtp_kv.has_value() && int8.mtp_kv->layers == 1 &&
               int8.mtp_kv->pool.planes.size() == 4 &&
               int8.mtp_kv->pool.spec.page_group_count == 4 &&
               int8.mtp_kv->pool.spec.logical_page_capacity == 3,
           "enabled MTP has one paged KV layer");
    expect(int8.mtp_kv && int8.mtp_kv->pool.planes[2].spec.dtype == ninfer::DType::FP16 &&
               int8.mtp_kv->pool.planes[3].spec.dtype == ninfer::DType::FP16,
           "INT8 MTP KV has scale planes");
    expect(int8.kv_payload_bytes() == int8.text_kv.payload_bytes() + int8.mtp_kv->payload_bytes(),
           "INT8 Text/MTP KV payload accounting");
}

void test_round_layout() {
    ninfer::LayoutBuilder builder;
    q36::RoundStateLayout round = q36::begin_round_state_layout(
        builder, q36::RoundStateSpec{
                     .hidden = 32, .output_rows = 128, .draft_window = 5, .enable_mtp = true});
    const ninfer::TensorRegion exact_prefill =
        builder.add_tensor(ninfer::DType::BF16, {32, 16}, 256, "exact prefill hidden");
    q36::complete_round_state_layout(builder, round);
    (void)builder.finish(256);
    expect(round.complete, "round layout completes");
    expect(round.logits.shape[0] == 128 && round.logits.shape[1] == 1, "round logits shape");
    expect(round.mtp.has_value() && round.mtp->draft_tokens.shape[0] == 5 &&
               round.mtp->target_input_ids.shape[0] == 6,
           "MTP prefill scratch shapes");
    expect(round.logits.region.offset < exact_prefill.region.offset &&
               exact_prefill.region.offset < round.mtp->draft_tokens.region.offset,
           "exact prefill extension retains established round-region order");
    expect(round.mtp.has_value() && round.mtp->position.shape[0] == 1,
           "MTP prefill scratch is explicit");
    expect(round.mtp_decode.has_value() && round.mtp_decode->alignment_ids.shape[0] == 6 &&
               round.mtp_decode->alignment_ids.shape[1] == 1,
           "MTP decode frame is explicit");

    ninfer::LayoutBuilder speculative_builder;
    q36::RoundStateLayout dflash = q36::begin_round_state_layout(
        speculative_builder,
        q36::RoundStateSpec{
            .hidden = 32, .output_rows = 128, .draft_window = 15, .enable_dflash = true});
    q36::complete_round_state_layout(speculative_builder, dflash);
    (void)speculative_builder.finish(256);
    expect(dflash.logits.shape[1] == 1 && dflash.dflash_prefill.has_value() &&
               dflash.dflash_prefill->produced_count.shape[0] == 1 &&
               dflash.dflash_decode.has_value() &&
               dflash.dflash_decode->draft_tokens.shape[0] == 15,
           "K=15 DFlash storage is backend-owned");
    expect(!dflash.mtp.has_value() && !dflash.mtp_decode.has_value(),
           "DFlash layout does not allocate MTP storage");

    ninfer::LayoutBuilder dflash2_builder;
    q36::RoundStateLayout dflash2 = q36::begin_round_state_layout(
        dflash2_builder,
        q36::RoundStateSpec{.hidden = 32, .output_rows = 128, .batch_capacity = 3,
                            .draft_window = 15, .enable_dflash2 = true});
    q36::complete_round_state_layout(dflash2_builder, dflash2);
    const auto dflash2_bytes = dflash2_builder.finish(256);
    expect(dflash2.dflash2_decode.has_value() &&
               dflash2.dflash2_decode->target_hidden.shape[2] == 3 &&
               dflash2.dflash2_decode->target_logits.shape[1] == 16 &&
               !dflash2.dflash_decode && !dflash2.ordinary,
           "DFlash2 verification layout must be independent of DFlash1");
    ninfer::DeviceBuffer dflash2_backing(dflash2_bytes);
    q36::RoundState bound_dflash2(
        ninfer::DeviceSpan{dflash2_backing.p, dflash2_backing.bytes}, dflash2);
    expect(bound_dflash2.dflash2_decode.has_value() &&
               bound_dflash2.dflash2_decode->licensed_tokens.ne[0] == 16 &&
               bound_dflash2.dflash2_decode->target_positions.ne[1] == 3,
           "DFlash2 target verification frame failed to bind");
}

void test_mtp_alignment() {
    const std::vector<std::int32_t> scatter{2, 4, 7};
    const q36::MtpAlignmentWindow first = q36::plan_mtp_alignment_window(8, 0, 4);
    expect(first.hidden_begin == 0 && first.position_begin == 0 &&
               first.shifted_embedding_begin == 1 && first.columns == 4 &&
               !first.final_column_uses_generated_token,
           "non-final MTP alignment window");
    const q36::MtpVisualOverlap first_visual = q36::shifted_visual_overlap(scatter, 8, first);
    expect(first_visual.source_begin == 0 &&
               first_visual.destination_columns == std::vector<std::int32_t>({1, 3}),
           "non-final shifted visual overlap");

    const q36::MtpAlignmentWindow final = q36::plan_mtp_alignment_window(8, 4, 4);
    expect(final.shifted_embedding_begin == 5 && final.final_column_uses_generated_token,
           "final MTP alignment window");
    const q36::MtpVisualOverlap final_visual = q36::shifted_visual_overlap(scatter, 8, final);
    expect(final_visual.source_begin == 2 &&
               final_visual.destination_columns == std::vector<std::int32_t>({2}),
           "final shifted visual overlap excludes generated-token column");
}

void test_vision_control() {
    q36::PreparedPromptData prompt;
    prompt.token_ids.resize(7);
    prompt.token_types           = {0, static_cast<std::uint8_t>(q36::PromptModality::Image),
                                    0, static_cast<std::uint8_t>(q36::PromptModality::Video),
                                    0, static_cast<std::uint8_t>(q36::PromptModality::Video),
                                    0};
    prompt.prepare.media_items   = 2;
    prompt.prepare.raw_patches   = 12;
    prompt.prepare.vision_tokens = 3;
    prompt.vision_items          = {
        q36::VisionItem{.modality    = q36::PromptModality::Image,
                                 .grid        = {.temporal = 1, .height = 2, .width = 2},
                                 .patch_begin = 0,
                                 .patch_count = 4,
                                 .token_spans = {{.begin = 1, .count = 1}}},
        q36::VisionItem{.modality    = q36::PromptModality::Video,
                                 .grid        = {.temporal = 2, .height = 2, .width = 2},
                                 .patch_begin = 4,
                                 .patch_count = 8,
                                 .token_spans = {{.begin = 3, .count = 1}, {.begin = 5, .count = 1}}},
    };

    const q36::VisionControl control = q36::build_vision_control(prompt);
    expect(control.items.size() == 2, "Vision per-item control count");
    expect(control.items[0].patch_begin == 0 && control.items[0].patch_count == 4 &&
               control.items[0].merged_count == 1 && control.items[0].segment_length == 4 &&
               control.items[0].segment_count == 1 &&
               control.items[0].cu_seqlens == std::vector<std::int32_t>({0, 4}) &&
               control.items[0].scatter_indices == std::vector<std::int32_t>({1}) &&
               control.items[0].position_ids.size() == 8 &&
               control.items[0].position_table_indices.size() == 16 &&
               control.items[0].position_table_weights.size() == 16,
           "image item control offsets");
    expect(control.items[1].patch_begin == 4 && control.items[1].patch_count == 8 &&
               control.items[1].merged_count == 2 && control.items[1].segment_length == 4 &&
               control.items[1].segment_count == 2 &&
               control.items[1].cu_seqlens == std::vector<std::int32_t>({0, 4, 8}) &&
               control.items[1].scatter_indices == std::vector<std::int32_t>({3, 5}) &&
               control.items[1].position_ids.size() == 16 &&
               control.items[1].position_table_indices.size() == 32 &&
               control.items[1].position_table_weights.size() == 32,
           "video item control offsets");
}

q36::PreparedPromptData identity_prompt(std::uint8_t digest_byte = 1) {
    q36::PreparedPromptData prompt;
    prompt.token_ids   = {10, 248056, 248056, 11};
    prompt.token_types = {0, static_cast<std::uint8_t>(q36::PromptModality::Image),
                          static_cast<std::uint8_t>(q36::PromptModality::Image), 0};
    prompt.positions   = {0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 2, 3};
    prompt.rope_delta  = 0;
    q36::VisionItem item{.modality    = q36::PromptModality::Image,
                         .grid        = {.temporal = 1, .height = 2, .width = 4},
                         .patch_begin = 0,
                         .patch_count = 8,
                         .token_spans = {{.begin = 1, .count = 2}}};
    item.content_digest.fill(digest_byte);
    prompt.vision_items.push_back(std::move(item));
    return prompt;
}

void append_text_token(q36::PreparedPromptData& prompt, ninfer::TokenId token,
                       std::int32_t position) {
    const std::size_t old_tokens = prompt.token_ids.size();
    std::vector<std::int32_t> positions;
    positions.reserve(3 * (old_tokens + 1));
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto begin =
            prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * old_tokens);
        positions.insert(positions.end(), begin, begin + static_cast<std::ptrdiff_t>(old_tokens));
        positions.push_back(position);
    }
    prompt.token_ids.push_back(token);
    prompt.token_types.push_back(0);
    prompt.positions = std::move(positions);
}

void test_prefix_identity() {
    q36::PreparedPromptData original    = identity_prompt();
    std::vector<ninfer::TokenId> ledger = original.token_ids;
    q36::detail::ResidentPrefixIdentity resident;
    resident.reserve(16);
    resident.assign(original);

    expect(q36::detail::prefix_matches(original, ledger, resident, original.token_ids.size()),
           "identical multimodal prefix identity");

    q36::PreparedPromptData changed_media = identity_prompt(2);
    expect(!q36::detail::prefix_matches(changed_media, ledger, resident,
                                        changed_media.token_ids.size()),
           "different media content must not reuse placeholder tokens");
    expect(q36::detail::prefix_matches(changed_media, ledger, resident, 1),
           "media wholly after the frontier does not affect prefix identity");
    expect(!q36::detail::prefix_matches(original, ledger, resident, 2),
           "frontier must not divide one Vision item");

    q36::PreparedPromptData changed_position = identity_prompt();
    changed_position.positions[0] += 1;
    expect(!q36::detail::prefix_matches(changed_position, ledger, resident,
                                        changed_position.token_ids.size()),
           "different MRoPE positions must not reuse resident state");

    resident.append_generated(1, original.rope_delta);
    ledger.push_back(12);
    append_text_token(original, 12, 4);
    expect(q36::detail::prefix_matches(original, ledger, resident, ledger.size()),
           "generated multimodal continuation identity");

    const q36::PreparedPromptData prompt_only = identity_prompt();
    resident.truncate(prompt_only.token_ids.size());
    ledger.resize(prompt_only.token_ids.size());
    expect(q36::detail::prefix_matches(prompt_only, ledger, resident, ledger.size()),
           "truncated multimodal continuation identity");
}


q36::PreparedPromptData block_prompt(std::size_t tokens, std::uint8_t digest_byte,
                                     bool with_vision) {
    q36::PreparedPromptData prompt;
    prompt.token_ids.resize(tokens);
    for (std::size_t index = 0; index < tokens; ++index) {
        prompt.token_ids[index] = static_cast<ninfer::TokenId>(1000 + index);
    }
    prompt.token_types.assign(tokens, 0);
    prompt.positions.resize(3 * tokens);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        for (std::size_t index = 0; index < tokens; ++index) {
            prompt.positions[axis * tokens + index] = static_cast<std::int32_t>(index);
        }
    }
    if (with_vision) {
        q36::VisionItem item{.modality    = q36::PromptModality::Image,
                             .grid        = {.temporal = 1, .height = 4, .width = 4},
                             .patch_begin = 0,
                             .patch_count = 16,
                             .token_spans = {{.begin = 70, .count = 30}}};
        item.content_digest.fill(digest_byte);
        prompt.vision_items.push_back(std::move(item));
    }
    return prompt;
}

void test_block_keys() {
    namespace pc                 = ninfer::runtime::prefix_cache;
    const q36::PreparedPromptData prompt = block_prompt(192, 1, true);
    std::vector<std::uint64_t> hashes;
    std::vector<std::uint64_t> extras;
    q36::detail::prompt_block_keys(prompt, hashes, extras);
    expect(hashes.size() == 3, "three full blocks produce three hashes");
    expect(extras.size() == 3, "a media prompt carries one extra per full block");
    expect(hashes == pc::block_lookup_hashes(prompt.token_ids, extras),
           "prompt_block_keys agrees with the raw chain");
    expect(extras[0] == 0 && extras[1] != 0 && extras[2] != 0,
           "the Vision key starts at the block its token range opens in");

    const std::vector<q36::detail::VisionTokenRange> ranges = q36::detail::vision_ranges(prompt);
    expect(ranges.size() == 1 && ranges[0].begin == 70 && ranges[0].end == 100,
           "a Vision range runs from its first token to its last span end");

    std::vector<std::uint64_t> text_hashes;
    std::vector<std::uint64_t> text_extras;
    q36::detail::prompt_block_keys(block_prompt(192, 1, false), text_hashes, text_extras);
    expect(text_extras.empty(), "a text-only prompt carries no extras");
    expect(text_hashes[0] == hashes[0], "media opening in a later block leaves an earlier hash");
    expect(text_hashes[1] != hashes[1], "media identity enters the block it opens in");

    std::vector<std::uint64_t> other_hashes;
    std::vector<std::uint64_t> other_extras;
    q36::detail::prompt_block_keys(block_prompt(192, 2, true), other_hashes, other_extras);
    expect(other_extras[1] != extras[1], "different media content yields a different block key");
    expect(other_hashes[1] != hashes[1], "different media content changes the chained hash");

    const std::uint64_t first_key  = q36::detail::accumulate_vision(0, 7U);
    const std::uint64_t second_key = q36::detail::accumulate_vision(first_key, 9U);
    expect(first_key != 0 && first_key != 7U, "the cumulative Vision key is seeded");
    expect(second_key != first_key, "each Vision item shifts the cumulative key");

    bool threw = false;
    try {
        q36::PreparedPromptData broken = block_prompt(64, 1, false);
        broken.vision_items.push_back(q36::VisionItem{});
        (void)q36::detail::vision_ranges(broken);
    } catch (const std::logic_error&) {
        threw = true;
    }
    expect(threw, "a Vision item without a token span is rejected");
}

void test_prefix_digests() {
    const q36::PreparedPromptData prompt = identity_prompt();
    q36::detail::PrefixDigests digests;
    digests.assign(prompt);
    expect(digests.size() == prompt.token_ids.size(), "digest has one entry per frontier");
    expect(digests.at(0) == (std::array<std::uint64_t, 2>{1469598103934665603ULL,
                                                          7809847782465536322ULL}),
           "digest frontier zero is the seed");

    // A fixed wire format: the golden below was produced by an independent transcription of the
    // sibling line's algorithm, so a refactor that silently rekeys every stored page fails here.
    const std::array<std::array<std::uint64_t, 2>, 5> golden{{
        {0x14650fb0739d0383ULL, 0x6c62272e07bb0142ULL},
        {0x262fc1d227b27188ULL, 0x87481bd9956012d5ULL},
        {0x7e8c2767f8b60e3aULL, 0x7660bc00fa1210cfULL},
        {0x8f2e57777ab77754ULL, 0xd3531877496f52f0ULL},
        {0x5f39c6bfb181be25ULL, 0x7cb0d137e01b17abULL},
    }};
    for (std::size_t frontier = 0; frontier < golden.size(); ++frontier) {
        expect(digests.at(frontier) == golden[frontier], "digest wire-format golden");
    }

    q36::detail::PrefixDigests repeat;
    repeat.assign(identity_prompt());
    expect(repeat.image() == digests.image(), "digest is a pure function of the prompt");

    // Only the prefix may reach a frontier: a different fourth token leaves frontier 3 intact.
    q36::PreparedPromptData changed_suffix = identity_prompt();
    changed_suffix.token_ids[3] = 12;
    q36::detail::PrefixDigests suffix_digests;
    suffix_digests.assign(changed_suffix);
    expect(suffix_digests.at(3) == digests.at(3),
           "a later token does not disturb an earlier frontier");
    expect(suffix_digests.at(4) != digests.at(4), "a different token changes its frontier");

    q36::PreparedPromptData changed_type = identity_prompt();
    changed_type.token_types[0] = 1;
    q36::detail::PrefixDigests type_digests;
    type_digests.assign(changed_type);
    expect(type_digests.at(1) != digests.at(1), "token type is part of the digest");

    q36::PreparedPromptData changed_position = identity_prompt();
    changed_position.positions[0] += 1;
    q36::detail::PrefixDigests position_digests;
    position_digests.assign(changed_position);
    expect(position_digests.at(1) != digests.at(1), "MRoPE position is part of the digest");

    // A Vision item enters at the frontier its last span closes, not at its first token.
    q36::detail::PrefixDigests media_digests;
    media_digests.assign(identity_prompt(2));
    expect(media_digests.at(2) == digests.at(2), "media folds in only at its span end");
    expect(media_digests.at(3) != digests.at(3), "media content is part of the digest");

    // A rewrite checkpoint keys the frontier it closes, so a replay still shortlists under it.
    q36::PreparedPromptData checkpointed = identity_prompt();
    checkpointed.identity.rewrite_checkpoint =
        q36::RewriteCheckpointSpec{.kind = q36::RewriteCheckpointKind::TurnClosure, .frontier = 2};
    q36::detail::PrefixDigests checkpoint_digests;
    checkpoint_digests.assign(checkpointed);
    expect(checkpoint_digests.at(1) == digests.at(1),
           "a checkpoint does not key an earlier frontier");
    expect(checkpoint_digests.at(2) != digests.at(2), "a rewrite checkpoint keys its frontier");
    expect(checkpoint_digests.at(3) != digests.at(3), "a rewrite checkpoint carries forward");

    // Generated tokens extend the image the same way a rebuilt prompt would.
    q36::detail::PrefixDigests appended;
    appended.assign(prompt);
    const std::array<ninfer::TokenId, 2> generated{7, 8};
    appended.append_generated(generated, 0);
    q36::PreparedPromptData extended = identity_prompt();
    append_text_token(extended, 7, 4);
    append_text_token(extended, 8, 5);
    q36::detail::PrefixDigests rebuilt;
    rebuilt.assign(extended);
    expect(appended.image() == rebuilt.image(), "generated append matches a rebuilt prompt");

    q36::detail::PrefixDigests truncated;
    truncated.assign(extended);
    truncated.truncate(prompt.token_ids.size());
    expect(truncated.image() == digests.image(), "truncate returns to the prompt frontier");

    q36::detail::PrefixDigests restored;
    restored.restore(digests.image());
    for (std::size_t frontier = 0; frontier <= digests.size(); ++frontier) {
        expect(restored.at(frontier) == digests.at(frontier), "restore round-trips every frontier");
    }

    q36::detail::PrefixDigests swapped;
    swapped.assign(prompt);
    q36::detail::PrefixDigests emptied;
    swapped.swap(emptied);
    expect(swapped.size() == 0, "swap moves the image out");
    expect(emptied.image() == digests.image(), "swap moves the image in");
    emptied.clear();
    expect(emptied.size() == 0, "clear drops the resident prefix");

    // Malformed images and out-of-range frontiers are rejected, never silently trusted.
    const auto expect_throws = [](auto&& body, const char* label) {
        bool threw = false;
        try {
            body();
        } catch (const std::exception&) {
            threw = true;
        }
        expect(threw, label);
    };
    expect_throws([&] { q36::detail::PrefixDigests empty; (void)empty.at(0); },
                  "at rejects an absent image");
    expect_throws([&] { (void)digests.at(digests.size() + 1); },
                  "at rejects a frontier past the resident prefix");
    expect_throws([&] { q36::detail::PrefixDigests shallow; shallow.truncate(digests.size() + 1); },
                  "truncate cannot extend");
    expect_throws([&] {
        std::vector<std::array<std::uint64_t, 2>> image = digests.image();
        image.front() = {1U, 2U};
        q36::detail::PrefixDigests bad;
        bad.restore(std::move(image));
    }, "restore rejects a missing seed");
    expect_throws([&] {
        std::vector<std::array<std::uint64_t, 2>> image = digests.image();
        image[1][0] = 0;
        q36::detail::PrefixDigests bad;
        bad.restore(std::move(image));
    }, "restore rejects a zero lane");
    expect_throws([&] {
        const std::array<ninfer::TokenId, 1> fresh_tokens{1};
        q36::detail::PrefixDigests fresh;
        fresh.append_generated(fresh_tokens, 0);
    }, "append requires a resident prefix");

    // The profile tag decides which bytes a key names, so it must pack the three fields exactly.
    const std::uint32_t tag =
        q36::detail::identity_tag(ninfer::SpeculativeBackend::DFlash,
                                  ninfer::ProposalHead::Optimized,
                                  ninfer::KvCacheStorage::Int8Group64);
    expect(tag == (static_cast<std::uint32_t>(ninfer::SpeculativeBackend::DFlash) |
                   (static_cast<std::uint32_t>(ninfer::ProposalHead::Optimized) << 8U) |
                   (static_cast<std::uint32_t>(ninfer::KvCacheStorage::Int8Group64) << 16U)),
           "identity tag packs backend | head << 8 | storage << 16");

    const ninfer::DiskKVIdentity identity = q36::detail::make_identity(digests, tag, 3);
    expect(identity.lo == digests.at(3)[0] && identity.hi == digests.at(3)[1] &&
               identity.tag == tag && identity.frontier == 3,
           "make_identity keys on the frontier digest and the profile tag");
}

} // namespace

int main() {
    test_topology();
    test_decoder_layout();
    test_round_layout();
    test_mtp_alignment();
    test_vision_control();
    test_prefix_identity();
    test_prefix_digests();
    test_block_keys();
    if (failures != 0) {
        std::cerr << failures << " Qwen3.6 runtime mechanism checks failed\n";
        return 1;
    }
    std::cout << "Qwen3.6 runtime mechanism checks passed\n";
    return 0;
}
