#pragma once

#include "core/arena.h"
#include "core/linear_attention_state.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer {

/**
 * Packed pinned-host layout for one LinearAttentionStatePool image.
 *
 * The image mirrors a chosen set of slot ids across every layer: convolution records first, then
 * recurrent records, each region layer-major and slot-major. A mirrored record is therefore
 * identified by (layer, component, position in `slots`), never by the physical slot id it came
 * from, which is what lets a restored image land on a different slot.
 */
struct HostLinearStateLayout {
    LinearAttentionStatePoolSpec spec;
    std::vector<std::int32_t> slots;
    std::size_t conv_record_bytes      = 0;
    std::size_t recurrent_record_bytes = 0;
    std::size_t conv_layer_bytes       = 0;
    std::size_t recurrent_layer_bytes  = 0;
    std::size_t total_bytes            = 0;
};

[[nodiscard]] HostLinearStateLayout
plan_host_linear_state_layout(const LinearAttentionStatePoolSpec& spec,
                              std::span<const std::int32_t> slots);

class HostLinearStateArena;

/**
 * One pool image living in pinned host RAM. Move-only; `HostLinearStateArena` is the only producer
 * and the image returns its extent to that arena when it is destroyed, moved into, or released.
 */
class HostLinearStateImage {
public:
    HostLinearStateImage() noexcept;
    ~HostLinearStateImage();

    HostLinearStateImage(const HostLinearStateImage&)            = delete;
    HostLinearStateImage& operator=(const HostLinearStateImage&) = delete;
    HostLinearStateImage(HostLinearStateImage&& other) noexcept;
    HostLinearStateImage& operator=(HostLinearStateImage&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }
    [[nodiscard]] const HostLinearStateLayout& layout() const noexcept { return layout_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] const std::byte* data() const noexcept { return data_; }
    // Mutable base, for a caller that fills the image itself (the disk tier reading it back).
    [[nodiscard]] std::byte* mutable_data() noexcept { return data_; }

    // Returns the extent to the arena early. Idempotent.
    bool release() noexcept;

private:
    friend class HostLinearStateArena;

    HostLinearStateArena* owner_ = nullptr;
    std::byte* data_             = nullptr;
    std::size_t bytes_           = 0;
    HostLinearStateLayout layout_;
};

/**
 * Bounded pinned host RAM for Linear Attention state images. Every byte is allocated in the
 * constructor, so `allocate` never calls into the CUDA allocator and a park is a pure transfer.
 * First fit over one free list, which is what keeps a byte budget enforceable per Program.
 */
class HostLinearStateArena {
public:
    explicit HostLinearStateArena(std::size_t capacity_bytes);

    HostLinearStateArena(const HostLinearStateArena&)            = delete;
    HostLinearStateArena& operator=(const HostLinearStateArena&) = delete;
    HostLinearStateArena(HostLinearStateArena&&)                 = delete;
    HostLinearStateArena& operator=(HostLinearStateArena&&)      = delete;

    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }
    [[nodiscard]] std::size_t occupied_bytes() const noexcept { return occupied_bytes_; }
    [[nodiscard]] std::size_t free_bytes() const noexcept {
        return capacity_bytes_ - occupied_bytes_;
    }

    [[nodiscard]] bool can_allocate(std::size_t bytes) const noexcept;
    [[nodiscard]] std::optional<HostLinearStateImage>
    allocate(const HostLinearStateLayout& layout) noexcept;

private:
    friend class HostLinearStateImage;

    struct Extent {
        std::size_t offset = 0;
        std::size_t bytes  = 0;
    };

    void release_extent(std::byte* data, std::size_t bytes) noexcept;

    std::optional<PinnedHostBuffer> backing_;
    std::size_t capacity_bytes_ = 0;
    std::size_t occupied_bytes_ = 0;
    std::vector<Extent> free_extents_;
};

/**
 * Copies every mirrored slot of `pool` into a fresh image from `arena` on `stream`, synchronizes,
 * and returns. Returns nullopt when a slot id is outside the pool or the arena cannot fit the
 * image; `slots` must be non-empty and duplicate-free.
 */
[[nodiscard]] std::optional<HostLinearStateImage>
park_linear_state(HostLinearStateArena& arena, const LinearAttentionStatePool& pool,
                  std::span<const std::int32_t> slots, cudaStream_t stream);

/**
 * The same harvest into a caller-owned buffer of `layout.total_bytes`, in the packed layout order.
 * This is what a tier writing a state image straight to disk uses, so the bytes it stores are the
 * bytes a restore would demand; it synchronizes before returning.
 */
void park_linear_state_into(std::byte* destination, const HostLinearStateLayout& layout,
                            const LinearAttentionStatePool& pool,
                            std::span<const std::int32_t> slots, cudaStream_t stream);

/**
 * Copies `image` back into the slot ids recorded in `image.layout()`. The caller owns the ordering
 * contract for the destination state; this function neither zeroes nor synchronizes.
 */
void restore_linear_state(const HostLinearStateImage& image, LinearAttentionStatePool& pool,
                          cudaStream_t stream);

/**
 * The same restore from a packed buffer of `layout.total_bytes` rather than from an arena image,
 * which is how a tier that read the bytes back itself lands them.
 */
void restore_linear_state_into(const std::byte* image, const HostLinearStateLayout& layout,
                               LinearAttentionStatePool& pool,
                               std::span<const std::int32_t> destination_slots, cudaStream_t stream);

/**
 * Copies `image` into `destination_slots`, in the order the image recorded them. This is what lets
 * a parked state land on a different lane's slot pair than the one it came from.
 */
void restore_linear_state_to(const HostLinearStateImage& image, LinearAttentionStatePool& pool,
                             std::span<const std::int32_t> destination_slots, cudaStream_t stream);

} // namespace ninfer
