// Component check of the pinned-host mirror for Qwen3.6 GDN linear-attention state.
//
// Geometry: 27B at tp2 -- 48 GDN layers, conv [5120 channels x 3 width] BF16, recurrent
// [128 key x 128 value x 24 heads] FP32. The pool holds 2 slots per lane (current state and
// rewrite checkpoint), so this exercises the two-slot image the lane tier uses.
//
// What is checked:
//   1. the image lands in cudaMemoryTypeHost pinned RAM and is charged to the process;
//   2. parking then overwriting the source slots, then restoring, reproduces every byte of every
//      (layer, component, slot) record exactly;
//   3. the same image restores into different slot ids, so the image is keyed on content.

#include "core/arena.h"
#include "core/device.h"
#include "core/host_linear_state.h"
#include "core/linear_attention_state.h"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kLayers       = 48;
constexpr std::int32_t kConvChannels  = 5120;
constexpr std::int32_t kConvWidth     = 3;
constexpr std::int32_t kValueHeads    = 24;
constexpr std::int32_t kValueHeadDim  = 128;
constexpr std::int32_t kKeyHeadDim    = 128;
constexpr std::int32_t kSlots         = 4;  // 2 lanes x 2 roles

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool expect(bool condition, const std::string& label) {
    if (condition) { return true; }
    std::cerr << label << " failed\n";
    return false;
}

bool cuda_ok(cudaError_t error, const char* label) {
    if (error == cudaSuccess) { return true; }
    std::cerr << label << ": " << cudaGetErrorString(error) << '\n';
    return false;
}

std::size_t process_rss_bytes() {
    std::ifstream status("/proc/self/status");
    std::string key;
    while (status >> key) {
        if (key == "VmRSS:") {
            std::size_t kb = 0;
            status >> kb;
            return kb * 1024;
        }
        std::string rest;
        std::getline(status, rest);
    }
    return 0;
}

ninfer::LinearAttentionStatePoolSpec spec() {
    return ninfer::LinearAttentionStatePoolSpec{
        .layers         = kLayers,
        .conv_channels  = kConvChannels,
        .conv_width     = kConvWidth,
        .value_heads    = kValueHeads,
        .value_head_dim = kValueHeadDim,
        .key_head_dim   = kKeyHeadDim,
        .slot_count     = kSlots,
        .conv_dtype     = ninfer::DType::BF16,
    };
}

std::uint8_t pattern(std::int32_t slot, std::uint32_t layer, std::size_t index) {
    return static_cast<std::uint8_t>((slot * 97U + layer * 31U + index * 7U + 3U) & 0xFFU);
}

void fill_slot(ninfer::LinearAttentionStatePool& pool, std::int32_t slot, cudaStream_t stream) {
    std::vector<std::uint8_t> conv(static_cast<std::size_t>(kConvChannels) * kConvWidth * 2);
    std::vector<std::uint8_t> recurrent(static_cast<std::size_t>(kKeyHeadDim) * kValueHeadDim *
                                        kValueHeads * 4);
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        for (std::size_t i = 0; i < conv.size(); ++i) { conv[i] = pattern(slot, layer, i); }
        for (std::size_t i = 0; i < recurrent.size(); ++i) { recurrent[i] = pattern(slot, layer, i); }
        const ninfer::Tensor conv_state      = pool.conv_slot(layer, slot);
        const ninfer::Tensor recurrent_state = pool.recurrent_slot(layer, slot);
        cudaMemcpyAsync(conv_state.data, conv.data(), conv.size(), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(recurrent_state.data, recurrent.data(), recurrent.size(),
                        cudaMemcpyHostToDevice, stream);
    }
    cudaStreamSynchronize(stream);
}

std::size_t count_slot_mismatches(ninfer::LinearAttentionStatePool& pool, std::int32_t slot,
                                  std::int32_t pattern_slot, cudaStream_t stream) {
    std::vector<std::uint8_t> conv(static_cast<std::size_t>(kConvChannels) * kConvWidth * 2);
    std::vector<std::uint8_t> recurrent(static_cast<std::size_t>(kKeyHeadDim) * kValueHeadDim *
                                        kValueHeads * 4);
    std::size_t mismatches = 0;
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        const ninfer::Tensor conv_state      = pool.conv_slot(layer, slot);
        const ninfer::Tensor recurrent_state = pool.recurrent_slot(layer, slot);
        cudaMemcpyAsync(conv.data(), conv_state.data, conv.size(), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(recurrent.data(), recurrent_state.data, recurrent.size(),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (std::size_t i = 0; i < conv.size(); ++i) {
            if (conv[i] != pattern(pattern_slot, layer, i)) { ++mismatches; }
        }
        for (std::size_t i = 0; i < recurrent.size(); ++i) {
            if (recurrent[i] != pattern(pattern_slot, layer, i)) { ++mismatches; }
        }
    }
    return mismatches;
}

} // namespace

int main() {
    int devices                 = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&devices);
    if (count_err == cudaErrorNoDevice || count_err == cudaErrorInsufficientDriver ||
        (count_err == cudaSuccess && devices == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    int failures = 0;
    ninfer::DeviceContext ctx(0);

    const ninfer::LinearAttentionStatePoolSpec pool_spec = spec();
    ninfer::LayoutBuilder builder;
    const ninfer::LinearAttentionStatePoolLayout layout =
        ninfer::plan_linear_attention_state_pool(builder, pool_spec);
    ninfer::DeviceArena arena(builder.finish(256));
    ninfer::LinearAttentionStatePool pool({arena.base(), arena.capacity()}, layout);

    fill_slot(pool, 0, ctx.stream);
    fill_slot(pool, 2, ctx.stream);

    const std::vector<std::int32_t> slots{0, 2};
    const ninfer::HostLinearStateLayout host_layout =
        ninfer::plan_host_linear_state_layout(pool_spec, slots);
    const std::size_t expected_bytes =
        static_cast<std::size_t>(kLayers) * slots.size() *
        (static_cast<std::size_t>(kConvChannels) * kConvWidth * 2 +
         static_cast<std::size_t>(kKeyHeadDim) * kValueHeadDim * kValueHeads * 4);
    failures += expect(host_layout.total_bytes == expected_bytes, "layout byte count") ? 0 : 1;

    const std::size_t rss_before = process_rss_bytes();
    ninfer::HostLinearStateArena host_arena(expected_bytes);
    const auto park_start        = std::chrono::steady_clock::now();
    std::optional<ninfer::HostLinearStateImage> image =
        ninfer::park_linear_state(host_arena, pool, slots, ctx.stream);
    const auto park_end = std::chrono::steady_clock::now();
    if (!image) {
        failures += fail("park returned no image");
        std::cout << "FAIL\n";
        return 1;
    }
    const std::size_t rss_after = process_rss_bytes();

    cudaPointerAttributes attributes{};
    failures += cuda_ok(cudaPointerGetAttributes(&attributes, image->data()),
                        "pointer attributes")
                    ? 0
                    : 1;
    failures += expect(attributes.type == cudaMemoryTypeHost, "image lives in host memory") ? 0 : 1;
    failures += expect(image->bytes() == expected_bytes, "image byte count") ? 0 : 1;
    failures += expect(host_arena.occupied_bytes() == expected_bytes, "arena charged") ? 0 : 1;
    failures += expect(rss_after >= rss_before + expected_bytes,
                       "resident host RAM grew by the image") ? 0 : 1;

    // 2. Overwrite the source slots, then restore and compare byte for byte.
    fill_slot(pool, 0, ctx.stream);
    fill_slot(pool, 2, ctx.stream);
    ninfer::restore_linear_state(*image, pool, ctx.stream);
    failures += cuda_ok(cudaStreamSynchronize(ctx.stream), "restore sync") ? 0 : 1;
    failures += expect(count_slot_mismatches(pool, 0, 0, ctx.stream) == 0, "slot 0 byte-exact") ? 0 : 1;
    failures += expect(count_slot_mismatches(pool, 2, 2, ctx.stream) == 0, "slot 2 byte-exact") ? 0 : 1;

    // 3. The same image restores into a different slot pair.
    const std::array<std::int32_t, 2> relocated{1, 3};
    ninfer::restore_linear_state_to(*image, pool, relocated, ctx.stream);
    failures += cuda_ok(cudaStreamSynchronize(ctx.stream), "relocated restore sync") ? 0 : 1;
    failures += expect(count_slot_mismatches(pool, 1, 0, ctx.stream) == 0, "slot 1 byte-exact") ? 0 : 1;
    failures += expect(count_slot_mismatches(pool, 3, 2, ctx.stream) == 0, "slot 3 byte-exact") ? 0 : 1;

    const double mib = static_cast<double>(expected_bytes) / (1024.0 * 1024.0);
    const double ms =
        std::chrono::duration<double>(park_end - park_start).count() * 1e3;
    std::cout.setf(std::ios::fixed);
    std::cout.precision(1);
    std::cout << "GDN state image " << mib << " MiB/rank (" << kLayers << " layers x "
              << slots.size() << " slots)\n";
    std::cout.precision(2);
    std::cout << "park D2H " << ms << " ms (" << mib / (ms / 1e3) << " MiB/s)\n";
    std::cout.precision(1);
    std::cout << "host RSS " << (rss_before / (1024.0 * 1024.0)) << " -> "
              << (rss_after / (1024.0 * 1024.0)) << " MiB\n";

    image.reset();
    failures += expect(host_arena.occupied_bytes() == 0, "arena released after image drop") ? 0 : 1;

    std::cout << (failures == 0 ? "PASS" : "FAIL") << '\n';
    return failures == 0 ? 0 : 1;
}
