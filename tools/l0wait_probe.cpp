// l0wait_probe (2026-11-19): can the in-order compute queue WAIT on a host-signaled Level Zero event -- a wait in the
// command stream, no spinning kernel -- and what does a GPU -> host -> GPU round trip cost that way, against today's
// way (the host waits for the GPU's flag, then enqueues the next kernel)?
//
// Why: Flash-Next on two B70s, plain decode: each GPU is busy 12.5 ms of a 20.8 ms token (unitrace 11-19).  The gap
// is the 96 small all-reduces a token: the host waits for each, sums, and only then enqueues the next kernels.  With
// a host-signaled event the host can enqueue the whole token ahead and a helper thread does the sums.
//
// Safety (memory: grimoire-probe-timeout-kills): no kernel spins on host memory.  A watchdog in the helper signals
// EVERY event it owes before its deadline (3 s), so the GPU is never left waiting.  stdout unbuffered, one small
// calibration round first, an internal time budget per test.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <atomic>
#include <dlfcn.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <variant>
#include <vector>

// Level Zero entry points from the loader at run time (as include/b70/peer_copy.hpp): the build image has no loader
#define L0FN(f) static decltype(&::f) p_##f = nullptr;
L0FN(zeEventPoolCreate) L0FN(zeEventCreate) L0FN(zeEventHostSignal) L0FN(zeEventHostReset)
L0FN(zeCommandListAppendWaitOnEvents) L0FN(zeEventDestroy) L0FN(zeEventPoolDestroy)
#undef L0FN
static bool load_l0() {
    void* h = ::dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) return false;
#define L0SYM(f) p_##f = reinterpret_cast<decltype(p_##f)>(::dlsym(h, #f)); if (!p_##f) return false;
    L0SYM(zeEventPoolCreate) L0SYM(zeEventCreate) L0SYM(zeEventHostSignal) L0SYM(zeEventHostReset)
    L0SYM(zeCommandListAppendWaitOnEvents) L0SYM(zeEventDestroy) L0SYM(zeEventPoolDestroy)
#undef L0SYM
    return true;
}

using clk = std::chrono::steady_clock;
static double us_since(clk::time_point t) { return std::chrono::duration<double, std::micro>(clk::now() - t).count(); }

#define ZE_OK(x) do { ze_result_t r_ = (x); if (r_ != ZE_RESULT_SUCCESS) { \
    std::printf("FAIL %s = 0x%x\n", #x, unsigned(r_)); return 3; } } while (0)

// the GPU -> host half, as Grimoire's ksend: one work-item stores the tag after a system-scope fence
static void send_tag(sycl::queue& q, uint64_t* hflag, uint64_t tag) {
    q.single_task([=] {
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        *reinterpret_cast<volatile uint64_t*>(hflag) = tag;
    });
}
// the host -> GPU half: read what the host wrote (pinned RAM) into device memory
// a layer's worth of small kernels: each a bounded arithmetic loop on 64 work-items (no memory polling)
static int g_work_kernels = 0, g_work_iters = 2000;
static float* g_scratch = nullptr;
static void work(sycl::queue& q) {
    float* s = g_scratch; const int it = g_work_iters;
    for (int k = 0; k < g_work_kernels; ++k)
        q.parallel_for(sycl::range<1>(64), [=](sycl::id<1> i) {
            float a = s[i], b = 1.0001f;
            for (int j = 0; j < it; ++j) a = a * b + 1e-7f;
            s[i] = a;
        });
}
static void recv_val(sycl::queue& q, const int* hdata, int i, int* dacc) {
    q.single_task([=] { dacc[i] = reinterpret_cast<const volatile int*>(hdata)[i] + 1; });
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device: %s, backend %s\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
                q.get_backend() == sycl::backend::ext_oneapi_level_zero ? "level_zero" : "OTHER");
    if (q.get_backend() != sycl::backend::ext_oneapi_level_zero) return 2;
    if (!load_l0()) { std::printf("libze_loader.so.1 not loaded -- stop\n"); return 2; }
    auto nq = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q);
    ze_command_list_handle_t cl = nullptr;
    if (auto* p = std::get_if<ze_command_list_handle_t>(&nq)) cl = *p;
    if (!cl) { std::printf("the queue is not an immediate command list -- stop\n"); return 2; }
    ze_context_handle_t ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
    ze_device_handle_t dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());

    constexpr int NE = 256;
    ze_event_pool_desc_t pd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr, ZE_EVENT_POOL_FLAG_HOST_VISIBLE, NE};
    ze_event_pool_handle_t pool = nullptr;
    ZE_OK(p_zeEventPoolCreate(ctx, &pd, 1, &dev, &pool));
    std::vector<ze_event_handle_t> ev(NE);
    for (int i = 0; i < NE; ++i) {
        ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, uint32_t(i), ZE_EVENT_SCOPE_FLAG_HOST,
                           ZE_EVENT_SCOPE_FLAG_HOST};
        ZE_OK(p_zeEventCreate(pool, &ed, &ev[size_t(i)]));
    }
    uint64_t* hflag = sycl::malloc_host<uint64_t>(8, q);
    int* hdata = sycl::malloc_host<int>(NE, q);
    int* dacc = sycl::malloc_device<int>(NE, q);
    std::vector<int> back(NE);

    // rounds: send_tag(i+1); [ahead: wait ev[i]]; recv_val(i).  The helper sees tag i+1, writes hdata[i] = 1000 + i,
    // signals ev[i].  It owes every event 0..n-1: past the deadline it signals the rest and reports the failure.
    auto run = [&](int n, bool ahead, double& us_per_round, bool& ok) -> int {
        *reinterpret_cast<volatile uint64_t*>(hflag) = 0;
        for (int i = 0; i < n; ++i) hdata[i] = -7;
        q.memset(dacc, 0, sizeof(int) * NE).wait();
        for (int i = 0; i < n; ++i) ZE_OK(p_zeEventHostReset(ev[size_t(i)]));
        std::atomic<bool> fail{false};
        const auto t0 = clk::now();
        if (ahead) {
            std::thread helper([&] {
                const auto deadline = clk::now() + std::chrono::seconds(3);
                int i = 0;
                for (; i < n; ++i) {
                    while (*reinterpret_cast<volatile uint64_t*>(hflag) < uint64_t(i + 1)) {
                        if (clk::now() > deadline) { fail = true; break; }
                        __builtin_ia32_pause();
                    }
                    if (fail) break;
                    hdata[i] = 1000 + i;
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    p_zeEventHostSignal(ev[size_t(i)]);
                }
                for (; i < n; ++i) p_zeEventHostSignal(ev[size_t(i)]);   // never leave the GPU waiting
            });
            for (int i = 0; i < n; ++i) {
                work(q);
                send_tag(q, hflag, uint64_t(i + 1));
                if (p_zeCommandListAppendWaitOnEvents(cl, 1, &ev[size_t(i)]) != ZE_RESULT_SUCCESS) {
                    fail = true;
                    std::printf("append wait failed at %d\n", i);
                    break;
                }
                recv_val(q, hdata, i, dacc);
            }
            q.wait();
            helper.join();
        } else {
            // today's way: the host waits for the flag, writes, then enqueues the next kernels
            const auto deadline = clk::now() + std::chrono::seconds(3);
            for (int i = 0; i < n && !fail; ++i) {
                work(q);
                send_tag(q, hflag, uint64_t(i + 1));
                while (*reinterpret_cast<volatile uint64_t*>(hflag) < uint64_t(i + 1)) {
                    if (clk::now() > deadline) { fail = true; break; }
                    __builtin_ia32_pause();
                }
                hdata[i] = 1000 + i;
                std::atomic_thread_fence(std::memory_order_seq_cst);
                recv_val(q, hdata, i, dacc);
            }
            q.wait();
        }
        us_per_round = us_since(t0) / n;
        q.memcpy(back.data(), dacc, sizeof(int) * NE).wait();
        int bad = 0;
        for (int i = 0; i < n; ++i) bad += back[size_t(i)] != 1001 + i;
        ok = !fail && bad == 0;
        if (fail || bad) std::printf("  %s n=%d: helper/host timeout %d, wrong values %d\n",
                                     ahead ? "ahead" : "host-wait", n, int(fail.load()), bad);
        return 0;
    };

    if (const char* e = std::getenv("L0W_ITERS")) g_work_iters = std::atoi(e);
    g_scratch = sycl::malloc_device<float>(64, q);
    q.memset(g_scratch, 0, 64 * sizeof(float)).wait();
    // device time of one work kernel alone (enqueue 200 back to back, no syncs)
    {
        g_work_kernels = 200;
        const auto t = clk::now(); work(q); q.wait();
        std::printf("one work kernel ~%.1f us of device time (200 back to back)\n", us_since(t) / 200);
        g_work_kernels = 0;
    }
    const auto budget = clk::now();
    double us = 0; bool ok = false;
    // calibration: one round each way
    if (run(1, false, us, ok) || !ok) { std::printf("host-wait n=1 FAILED\n"); return 4; }
    std::printf("calibration host-wait 1 round: %.1f us (ok)\n", us);
    if (run(1, true, us, ok) || !ok) { std::printf("event-wait n=1 FAILED -- stop\n"); return 4; }
    std::printf("calibration event-wait 1 round: %.1f us (ok)\n", us);
    for (int rep = 0; rep < 5 && us_since(budget) < 10e6; ++rep) {
        for (int n : {16, 96, 240}) {
            double a = 0, b = 0; bool oa = false, ob = false;
            run(n, false, a, oa);
            run(n, true, b, ob);
            std::printf("rep %d n=%3d: host-wait %.1f us/round (%s), event-wait enqueued ahead %.1f us/round (%s)\n",
                        rep, n, a, oa ? "ok" : "BAD", b, ob ? "ok" : "BAD");
            if (!oa || !ob) { std::printf("stop at the first wrong round\n"); rep = 99; break; }
        }
    }
    // with a layer's worth of kernels between exchanges (15 kernels a round), as a decode token has
    for (int wk : {15}) {
        g_work_kernels = wk;
        for (int rep = 0; rep < 3 && us_since(budget) < 20e6; ++rep) {
            double a = 0, b = 0; bool oa = false, ob = false;
            run(96, false, a, oa);
            run(96, true, b, ob);
            std::printf("work %d kernels/round, 96 rounds: host-wait %.1f us/round (%s), event-wait ahead %.1f us/round (%s)\n",
                        wk, a, oa ? "ok" : "BAD", b, ob ? "ok" : "BAD");
            if (!oa || !ob) break;
        }
        g_work_kernels = 0;
    }
    sycl::free(g_scratch, q);
    for (auto e : ev) p_zeEventDestroy(e);
    p_zeEventPoolDestroy(pool);
    sycl::free(hflag, q); sycl::free(hdata, q); sycl::free(dacc, q);
    std::printf("L0WAIT_PROBE_DONE\n");
    return 0;
}
