// Exercise a fixed virtual stride with two slots sharing a physical page budget.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>

#include "../src/kv_vmm.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

namespace vmm = sycl::ext::oneapi::experimental;
using Clock = std::chrono::steady_clock;

struct Reservation {
    const sycl::context& context;
    size_t bytes;
    uintptr_t address;

    Reservation(const sycl::context& ctx, size_t size)
        : context(ctx), bytes(size), address(vmm::reserve_virtual_mem(size, ctx)) {}
    ~Reservation() { vmm::free_virtual_mem(address, bytes, context); }
    Reservation(const Reservation&) = delete;
    Reservation& operator=(const Reservation&) = delete;
};

struct Mapping {
    const sycl::context& context;
    size_t bytes;
    void* address;

    Mapping(const sycl::context& ctx, vmm::physical_mem& page, uintptr_t addr, size_t size)
        : context(ctx), bytes(size), address(page.map(addr, size, vmm::address_access_mode::read_write)) {}
    ~Mapping() { vmm::unmap(address, bytes, context); }
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
};

static double elapsed_us(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

static void probe(sycl::queue& q, size_t page_bytes, size_t reserve_granularity) {
    constexpr size_t pages = 16;
    constexpr size_t desired_stride = size_t(1) << 30;
    const size_t alignment = std::max(page_bytes, reserve_granularity);
    const size_t stride = (desired_stride + alignment - 1) / alignment * alignment;
    if (pages * page_bytes > stride) throw std::runtime_error("page budget exceeds slot stride");

    const auto context = q.get_context();
    const auto device = q.get_device();
    Reservation slots(context, 2 * stride);
    std::vector<std::unique_ptr<vmm::physical_mem>> pool;
    std::vector<std::unique_ptr<Mapping>> left, right;
    pool.reserve(pages);
    left.reserve(pages);
    right.reserve(pages / 2);
    auto start = Clock::now();
    for (size_t p = 0; p < pages; ++p)
        pool.push_back(std::make_unique<vmm::physical_mem>(device, context, page_bytes));
    const double alloc_us = elapsed_us(start) / pages;

    start = Clock::now();
    for (size_t p = 0; p < pages; ++p)
        left.push_back(std::make_unique<Mapping>(context, *pool[p], slots.address + p * page_bytes, page_bytes));
    const double map_us = elapsed_us(start) / pages;

    auto* first = reinterpret_cast<uint32_t*>(slots.address);
    auto* second = reinterpret_cast<uint32_t*>(slots.address + stride);
    const size_t words = page_bytes / sizeof(uint32_t);
    uint32_t* result = sycl::malloc_shared<uint32_t>(4, q);
    if (!result) throw std::runtime_error("probe result allocation failed");
    const auto free_result = [&q](uint32_t* ptr) { sycl::free(ptr, q); };
    const std::unique_ptr<uint32_t, decltype(free_result)> result_owner(result, free_result);

    q.memset(first, 0, pages * page_bytes).wait_and_throw();
    q.parallel_for(sycl::range<1>(pages), [=](sycl::id<1> i) {
        first[i[0] * words] = 100 + i[0];
        first[(i[0] + 1) * words - 1] = 200 + i[0];
    }).wait_and_throw();

    // Both neighboring values straddle a boundary between physical pages.
    q.single_task([=] { result[0] = first[words - 1] + first[words]; }).wait_and_throw();
    if (result[0] != 301) throw std::runtime_error("cross-page load mismatch");

    start = Clock::now();
    left.resize(pages / 2);
    const double unmap_us = elapsed_us(start) / (pages / 2);
    start = Clock::now();
    for (size_t p = 0; p < pages / 2; ++p)
        right.push_back(std::make_unique<Mapping>(context, *pool[p + pages / 2],
                                                  slots.address + stride + p * page_bytes, page_bytes));
    const double reuse_us = elapsed_us(start) / (pages / 2);

    q.single_task([=] {
        result[0] = first[words];
        result[1] = second[0];
        result[2] = second[words - 1] + second[words];
    }).wait_and_throw();
    if (result[0] != 101 || result[1] != 108 || result[2] != 317)
        throw std::runtime_error("physical page reuse mismatch");
    q.memset(second, 0, (pages / 2) * page_bytes).wait_and_throw();
    q.single_task([=] {
        result[0] = second[words - 1] + second[words];
        result[1] = first[words];
    }).wait_and_throw();
    if (result[0] != 0 || result[1] != 101) throw std::runtime_error("slot isolation mismatch");
    std::printf("PASS page=%zu reserved=%zu backed=%zu alloc=%.1f us map=%.1f us unmap=%.1f us remap=%.1f us per page\n",
                page_bytes, 2 * stride, pages * page_bytes, alloc_us, map_us, unmap_us, reuse_us);
}

static void probe_shared_budget(sycl::queue& q, size_t page) {
    // Each head starts partway through a page. Growth must retain existing
    // data and map pages shared by two heads only once.
    constexpr size_t capacity = 777, heads = 3, dim = 256;
    const size_t bytes = (capacity * heads * dim + page - 1) / page * page;
    b70::KvVirtualPool pool(q, capacity, 2, 2 * bytes);
    auto* keys = pool.reserve(heads, dim);
    auto* values = pool.reserve(heads, dim);
    if (!pool.grow(0, 1)) throw std::runtime_error("short slot rejected");
    q.parallel_for(sycl::range<1>(heads), [=](sycl::id<1> i) {
        keys[i[0] * capacity * dim] = uint8_t(1 + i[0]);
        values[i[0] * capacity * dim] = uint8_t(4 + i[0]);
    }).wait_and_throw();
    if (!pool.grow(0, capacity) || pool.mapped_bytes(0) != 2 * bytes || pool.grow(1, 1))
        throw std::runtime_error("shared budget check failed");
    auto* result = sycl::malloc_shared<int>(1, q);
    if (!result) throw std::bad_alloc();
    const auto free_result = [&q](int* ptr) { sycl::free(ptr, q); };
    const std::unique_ptr<int, decltype(free_result)> result_owner(result, free_result);
    q.single_task([=] {
        *result = 0;
        for (size_t i = 0; i < heads; ++i)
            *result += keys[i * capacity * dim] + values[i * capacity * dim];
    }).wait_and_throw();
    if (*result != 21) throw std::runtime_error("slot data changed during growth");
    pool.recycle(0);
    if (!pool.grow(1, 300)) throw std::runtime_error("retired slot did not free budget");
    q.single_task([=] {
        *result = 0;
        for (size_t i = 0; i < heads; ++i)
            *result += keys[bytes + (i * capacity + 299) * dim] +
                       values[bytes + (i * capacity + 299) * dim];
    }).wait_and_throw();
    if (*result) throw std::runtime_error("new slot was not zeroed");
    std::printf("PASS engine pool: full slot used %zu bytes, retired slot reused by a %zu-byte prefix\n",
                2 * bytes, pool.mapped_bytes(1));
}

static void probe_compressed_indexer(sycl::queue& q, size_t page) {
    constexpr size_t ratio = 4;
    const size_t dimension = page / 64;
    b70::KvVirtualPool pool(q, 512, 1, 4 * page);
    auto* indexer = pool.reserve(1, dimension, ratio);
    if (!pool.grow(0, 255) || pool.mapped_bytes(0) != page)
        throw std::runtime_error("compressed indexer mapped too many pages");
    q.single_task([=] { indexer[page - 1] = 93; }).wait_and_throw();
    if (!pool.grow(0, 256) || pool.mapped_bytes(0) != 2 * page)
        throw std::runtime_error("compressed indexer did not grow at the block boundary");
    uint8_t old = 0, next = 1;
    q.memcpy(&old, indexer + page - 1, 1).wait_and_throw();
    q.memcpy(&next, indexer + page, 1).wait_and_throw();
    if (old != 93 || next != 0) throw std::runtime_error("compressed indexer data mismatch");
    std::puts("PASS compressed indexer block-boundary growth");
}

static void probe_long_context(sycl::queue& q) {
    // Real dense KV strides put the next extent off a 2 MiB boundary after
    // serving a short prompt, even when the reservation itself is aligned.
    constexpr size_t capacity = 140000, heads = 4, dim = 256;
    b70::KvVirtualPool pool(q, capacity, 1, size_t(1) << 30);
    auto* keys = pool.reserve(heads, dim);
    auto* values = pool.reserve(heads, dim);
    if (!pool.grow(0, 4096 + 128)) throw std::runtime_error("short prefix allocation failed");
    q.single_task([=] { keys[(capacity + 4095) * dim] = 93; }).wait_and_throw();

    if (!pool.grow(0, 16384 + 128) || !pool.grow(0, 131072 + 128))
        throw std::runtime_error("long context allocation failed");
    uint8_t old = 0, next = 1;
    q.memcpy(&old, keys + (capacity + 4095) * dim, 1).wait_and_throw();
    q.memcpy(&next, values + ((heads - 1) * capacity + 131199) * dim, 1).wait_and_throw();
    if (old != 93 || next != 0) throw std::runtime_error("long context growth changed slot data");
    std::puts("PASS long context pool growth through 128K plus decode budget");
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
        const auto device = q.get_device();
        const auto context = q.get_context();
        std::printf("device: %s\n", device.get_info<sycl::info::device::name>().c_str());
        if (!device.has(sycl::aspect::ext_oneapi_virtual_mem)) {
            std::puts("virtual memory aspect unavailable");
            return 2;
        }
        const size_t minimum = vmm::get_mem_granularity(device, context, vmm::granularity_mode::minimum);
        const size_t recommended = vmm::get_mem_granularity(device, context, vmm::granularity_mode::recommended);
        const size_t reservation = vmm::get_mem_granularity(context, vmm::granularity_mode::minimum);
        std::printf("granularity: minimum=%zu recommended=%zu reservation=%zu\n", minimum, recommended, reservation);
        const size_t huge = std::max(minimum, size_t(2) << 20);
        probe(q, minimum, reservation);
        if (huge != minimum) probe(q, huge, reservation);
        probe_shared_budget(q, std::max(minimum, reservation));
        probe_compressed_indexer(q, std::max(minimum, reservation));
        probe_long_context(q);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "probe failed: %s\n", ex.what());
        return 1;
    }
}
