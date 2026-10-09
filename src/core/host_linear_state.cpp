#include "core/host_linear_state.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

void check_cuda(cudaError_t error, const char* label) {
    if (error == cudaSuccess) { return; }
    throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(error));
}

#define CUDA_ASSERT(expression) check_cuda((expression), "Linear Attention state transfer")

std::size_t dtype_bytes(DType dtype) {
    switch (dtype) {
    case DType::BF16: return 2;
    case DType::FP32: return 4;
    default: throw std::invalid_argument("Linear Attention state image dtype is unsupported");
    }
}

bool same_spec(const LinearAttentionStatePoolSpec& lhs, const LinearAttentionStatePoolSpec& rhs) {
    return lhs.layers == rhs.layers && lhs.conv_channels == rhs.conv_channels &&
           lhs.conv_width == rhs.conv_width && lhs.value_heads == rhs.value_heads &&
           lhs.value_head_dim == rhs.value_head_dim && lhs.key_head_dim == rhs.key_head_dim &&
           lhs.slot_count == rhs.slot_count && lhs.conv_dtype == rhs.conv_dtype;
}

constexpr std::size_t kImageAlignment = 256;

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

} // namespace

HostLinearStateLayout plan_host_linear_state_layout(const LinearAttentionStatePoolSpec& spec,
                                                    std::span<const std::int32_t> slots) {
    if (slots.empty()) {
        throw std::invalid_argument("Linear Attention state image needs at least one slot");
    }
    for (std::size_t index = 1; index < slots.size(); ++index) {
        if (std::find(slots.begin(), slots.begin() + static_cast<std::ptrdiff_t>(index),
                      slots[index]) != slots.begin() + static_cast<std::ptrdiff_t>(index)) {
            throw std::invalid_argument("Linear Attention state image slot list has duplicates");
        }
    }

    HostLinearStateLayout layout;
    layout.spec                = spec;
    layout.slots.assign(slots.begin(), slots.end());
    layout.conv_record_bytes =
        static_cast<std::size_t>(spec.conv_channels) * static_cast<std::size_t>(spec.conv_width) *
        dtype_bytes(spec.conv_dtype);
    layout.recurrent_record_bytes =
        static_cast<std::size_t>(spec.key_head_dim) *
        static_cast<std::size_t>(spec.value_head_dim) *
        static_cast<std::size_t>(spec.value_heads) * dtype_bytes(DType::FP32);
    layout.conv_layer_bytes      = layout.conv_record_bytes * slots.size();
    layout.recurrent_layer_bytes = layout.recurrent_record_bytes * slots.size();
    layout.total_bytes           = align_up(
        static_cast<std::size_t>(spec.layers) * (layout.conv_layer_bytes + layout.recurrent_layer_bytes),
        kImageAlignment);
    return layout;
}

HostLinearStateImage::HostLinearStateImage() noexcept = default;

HostLinearStateImage::~HostLinearStateImage() { (void)release(); }

HostLinearStateImage::HostLinearStateImage(HostLinearStateImage&& other) noexcept
    : owner_(other.owner_), data_(other.data_), bytes_(other.bytes_),
      layout_(std::move(other.layout_)) {
    other.owner_ = nullptr;
    other.data_  = nullptr;
    other.bytes_ = 0;
}

HostLinearStateImage& HostLinearStateImage::operator=(HostLinearStateImage&& other) noexcept {
    if (this == &other) { return *this; }
    (void)release();
    owner_  = other.owner_;
    data_   = other.data_;
    bytes_  = other.bytes_;
    layout_ = std::move(other.layout_);
    other.owner_ = nullptr;
    other.data_  = nullptr;
    other.bytes_ = 0;
    return *this;
}

bool HostLinearStateImage::release() noexcept {
    if (owner_ == nullptr) { return false; }
    owner_->release_extent(data_, bytes_);
    owner_ = nullptr;
    data_  = nullptr;
    bytes_ = 0;
    return true;
}

HostLinearStateArena::HostLinearStateArena(std::size_t capacity_bytes) {
    if (capacity_bytes == 0) {
        throw std::invalid_argument("Host Linear Attention state arena needs a positive capacity");
    }
    backing_.emplace(capacity_bytes);
    capacity_bytes_ = capacity_bytes;
    free_extents_.push_back(Extent{.offset = 0, .bytes = capacity_bytes});
}

bool HostLinearStateArena::can_allocate(std::size_t bytes) const noexcept {
    if (bytes == 0) { return false; }
    return std::any_of(free_extents_.begin(), free_extents_.end(),
                       [bytes](const Extent& extent) { return extent.bytes >= bytes; });
}

std::optional<HostLinearStateImage> HostLinearStateArena::allocate(
    const HostLinearStateLayout& layout) noexcept {
    if (layout.total_bytes == 0) { return std::nullopt; }
    for (std::size_t index = 0; index < free_extents_.size(); ++index) {
        Extent& extent = free_extents_[index];
        if (extent.bytes < layout.total_bytes) { continue; }
        const std::size_t offset = extent.offset;
        extent.offset += layout.total_bytes;
        extent.bytes -= layout.total_bytes;
        if (extent.bytes == 0) { free_extents_.erase(free_extents_.begin() + static_cast<std::ptrdiff_t>(index)); }
        occupied_bytes_ += layout.total_bytes;

        HostLinearStateImage image;
        image.owner_  = this;
        image.data_   = static_cast<std::byte*>(backing_->data()) + offset;
        image.bytes_  = layout.total_bytes;
        image.layout_ = layout;
        return image;
    }
    return std::nullopt;
}

void HostLinearStateArena::release_extent(std::byte* data, std::size_t bytes) noexcept {
    if (data == nullptr || bytes == 0) { return; }
    const std::size_t offset = static_cast<std::size_t>(
        data - static_cast<std::byte*>(backing_->data()));
    occupied_bytes_ -= bytes;
    free_extents_.push_back(Extent{.offset = offset, .bytes = bytes});
    std::sort(free_extents_.begin(), free_extents_.end(),
              [](const Extent& lhs, const Extent& rhs) { return lhs.offset < rhs.offset; });
    std::vector<Extent> merged;
    for (const Extent& extent : free_extents_) {
        if (!merged.empty() && merged.back().offset + merged.back().bytes == extent.offset) {
            merged.back().bytes += extent.bytes;
        } else {
            merged.push_back(extent);
        }
    }
    free_extents_ = std::move(merged);
}

std::optional<HostLinearStateImage>
park_linear_state(HostLinearStateArena& arena, const LinearAttentionStatePool& pool,
                  std::span<const std::int32_t> slots, cudaStream_t stream) {
    const HostLinearStateLayout layout = plan_host_linear_state_layout(pool.spec, slots);
    std::optional<HostLinearStateImage> image = arena.allocate(layout);
    if (!image) { return std::nullopt; }
    try {
        park_linear_state_into(const_cast<std::byte*>(image->data()), layout, pool, layout.slots,
                               stream);
    } catch (...) {
        image->release();
        throw;
    }
    return image;
}

void park_linear_state_into(std::byte* destination, const HostLinearStateLayout& layout,
                            const LinearAttentionStatePool& pool,
                            std::span<const std::int32_t> slots, cudaStream_t stream) {
    if (destination == nullptr || slots.size() != layout.slots.size()) {
        throw std::invalid_argument("Linear Attention state harvest needs a destination per slot");
    }
    if (pool.layer_count() != layout.spec.layers) {
        throw std::invalid_argument("Linear Attention state harvest pool mismatch");
    }
    for (const std::int32_t slot : slots) {
        if (slot < 0 || slot >= pool.slot_count()) {
            throw std::out_of_range("Linear Attention state harvest slot is out of range");
        }
    }
    const LinearAttentionStateAllLayersView view = pool.all_layers_view();
    std::byte* const base = destination;
    const std::size_t conv_region =
        static_cast<std::size_t>(layout.spec.layers) * layout.conv_layer_bytes;
    // Layer addresses are affine and each (layer, slot) record is contiguous, so one 2D copy
    // carries every layer of one mirrored slot: the host pitch is the packed per-layer size, the
    // device pitch is the pool's own layer stride.
    for (std::size_t position = 0; position < layout.slots.size(); ++position) {
        const std::int32_t slot = slots[position];
        CUDA_ASSERT(cudaMemcpy2DAsync(
            base + position * layout.conv_record_bytes, layout.conv_layer_bytes,
            static_cast<const std::byte*>(view.conv_layer0.data) +
                static_cast<std::size_t>(slot) * layout.conv_record_bytes,
            static_cast<std::size_t>(view.conv_layer_stride_bytes), layout.conv_record_bytes,
            layout.spec.layers, cudaMemcpyDeviceToHost, stream));
        CUDA_ASSERT(cudaMemcpy2DAsync(
            base + conv_region + position * layout.recurrent_record_bytes,
            layout.recurrent_layer_bytes,
            static_cast<const std::byte*>(view.recurrent_layer0.data) +
                static_cast<std::size_t>(slot) * layout.recurrent_record_bytes,
            static_cast<std::size_t>(view.recurrent_layer_stride_bytes),
            layout.recurrent_record_bytes, layout.spec.layers, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_ASSERT(cudaStreamSynchronize(stream));
}

void restore_linear_state_to(const HostLinearStateImage& image, LinearAttentionStatePool& pool,
                             std::span<const std::int32_t> destination_slots, cudaStream_t stream) {
    if (!image.valid()) { throw std::invalid_argument("Linear Attention state image is empty"); }
    restore_linear_state_into(image.data(), image.layout(), pool, destination_slots, stream);
}

void restore_linear_state_into(const std::byte* image, const HostLinearStateLayout& layout,
                               LinearAttentionStatePool& pool,
                               std::span<const std::int32_t> destination_slots,
                               cudaStream_t stream) {
    if (image == nullptr) { throw std::invalid_argument("Linear Attention state image is empty"); }
    if (!same_spec(pool.spec, layout.spec) || pool.layer_count() != layout.spec.layers) {
        throw std::invalid_argument("Linear Attention state image pool mismatch");
    }
    if (destination_slots.size() != layout.slots.size()) {
        throw std::invalid_argument("Linear Attention state restore slot count mismatch");
    }
    for (const std::int32_t slot : destination_slots) {
        if (slot < 0 || slot >= pool.slot_count()) {
            throw std::out_of_range("Linear Attention state restore slot is out of range");
        }
    }

    const LinearAttentionStateAllLayersView view = pool.all_layers_view();
    const std::byte* const base = image;
    const std::size_t conv_region =
        static_cast<std::size_t>(layout.spec.layers) * layout.conv_layer_bytes;
    for (std::size_t position = 0; position < layout.slots.size(); ++position) {
        const std::int32_t slot = destination_slots[position];
        CUDA_ASSERT(cudaMemcpy2DAsync(
            static_cast<std::byte*>(view.conv_layer0.data) +
                static_cast<std::size_t>(slot) * layout.conv_record_bytes,
            static_cast<std::size_t>(view.conv_layer_stride_bytes),
            base + position * layout.conv_record_bytes, layout.conv_layer_bytes,
            layout.conv_record_bytes, layout.spec.layers, cudaMemcpyHostToDevice, stream));
        CUDA_ASSERT(cudaMemcpy2DAsync(
            static_cast<std::byte*>(view.recurrent_layer0.data) +
                static_cast<std::size_t>(slot) * layout.recurrent_record_bytes,
            static_cast<std::size_t>(view.recurrent_layer_stride_bytes),
            base + conv_region + position * layout.recurrent_record_bytes,
            layout.recurrent_layer_bytes, layout.recurrent_record_bytes, layout.spec.layers,
            cudaMemcpyHostToDevice, stream));
    }
}

void restore_linear_state(const HostLinearStateImage& image, LinearAttentionStatePool& pool,
                          cudaStream_t stream) {
    restore_linear_state_to(image, pool, image.layout().slots, stream);
}

} // namespace ninfer
