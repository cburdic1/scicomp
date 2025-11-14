#include <iostream>
#include <cstdlib>
#include <string>
#include <chrono>
#include <vector>
#include <thread>
#include <atomic>
#include <barrier>
#include <filesystem>
#include "WaveOrthotope.hpp"

namespace fs = std::filesystem;


static int get_threads() {
    if (const char* s = std::getenv("SOLVER_NUM_THREADS")) {
        int t = std::atoi(s);
        return (t > 0 ? t : 1);
    }
    return 1;
}

static double get_interval() {
    if (const char* s = std::getenv("INTVL")) {
        double x = std::atof(s);
        return (x > 0 ? x : -1.0);
    }
    return -1.0;
}

static bool file_exists(const char* p) {
    std::error_code ec;
    return fs::exists(p, ec);
}

static void atomic_write(WaveOrthotope& w, const char* outPath) {
    std::string tmp = std::string(outPath) + ".tmp";
    w.write(tmp.c_str());
    std::error_code ec;
    fs::rename(tmp, outPath, ec);
    if (ec) {
        fs::remove(outPath, ec);
        fs::rename(tmp, outPath, ec);
    }
}


struct ThreadPool {
    WaveOrthotope& w;
    size_t nthreads;

    std::barrier<> phase_barrier;       
    std::vector<std::jthread> workers;

    struct Chunk { size_t r0, r1; };
    std::vector<Chunk> chunks;

    std::atomic<size_t> next{0};
    std::atomic<bool> keep_running{true};

    ThreadPool(WaveOrthotope& w_, size_t nt)
        : w(w_), nthreads(nt), phase_barrier(nt + 1)
    {
        size_t R = w.rows();
        size_t interior = R - 2;

        if (interior < nthreads) nthreads = interior;
        if (nthreads < 1) nthreads = 1;

        size_t base = interior / nthreads;
        size_t extra = interior % nthreads;

        size_t r = 1;
        chunks.reserve(nthreads);

        for (size_t t = 0; t < nthreads; t++) {
            size_t len = base + (t < extra ? 1 : 0);
            chunks.push_back({r, r + len});
            r += len;
        }

        workers.reserve(nthreads);
        for (size_t tid = 0; tid < nthreads; tid++) {
            workers.emplace_back([this]{
                size_t C = w.cols();
                while (keep_running.load(std::memory_order_acquire)) {

                
                    phase_barrier.arrive_and_wait();
                    run_laplacian(C);
                    phase_barrier.arrive_and_wait();

                 
                    phase_barrier.arrive_and_wait();
                    run_velocity(C);
                    phase_barrier.arrive_and_wait();

                   
                    phase_barrier.arrive_and_wait();
                    run_displacement(C);
                    phase_barrier.arrive_and_wait();
                }
            });
        }
    }

    inline void reset_queue() {
        next.store(0, std::memory_order_release);
    }

    inline void run_laplacian(size_t C) {
        for (;;) {
            size_t id = next.fetch_add(1, std::memory_order_acq_rel);
            if (id >= chunks.size()) return;
            auto [r0, r1] = chunks[id];

            for (size_t i = r0; i < r1; i++) {
                size_t base = i * C;
                size_t up   = (i - 1) * C;
                size_t dn   = (i + 1) * C;

                for (size_t j = 1; j < C - 1; j++) {
                    w.lap[base + j] =
                        0.5 * ( w.u[up + j] + w.u[dn + j]
                              + w.u[base + j - 1] + w.u[base + j + 1]
                              - 4.0 * w.u[base + j] );
                }
            }
        }
    }

    inline void run_velocity(size_t C) {
        for (;;) {
            size_t id = next.fetch_add(1, std::memory_order_acq_rel);
            if (id >= chunks.size()) return;
            auto [r0, r1] = chunks[id];

            for (size_t i = r0; i < r1; i++) {
                size_t base = i * C;
                for (size_t j = 1; j < C - 1; j++) {
                    size_t k = base + j;
                    w.v[k] += (w.c2 * w.lap[k] - w.c * w.v[k]) * w.dt;
                }
            }
        }
    }

    inline void run_displacement(size_t C) {
        for (;;) {
            size_t id = next.fetch_add(1, std::memory_order_acq_rel);
            if (id >= chunks.size()) return;
            auto [r0, r1] = chunks[id];

            for (size_t i = r0; i < r1; i++) {
                size_t base = i * C;
                for (size_t j = 1; j < C - 1; j++) {
                    size_t k = base + j;
                    w.u[k] += w.v[k] * w.dt;
                }
            }
        }
    }
};


int main(int argc, char** argv) {
    if (argc < 3) return 1;

    const char* inFile  = argv[1];
    const char* outFile = argv[2];

    int nthreads = get_threads();
    double interval = get_interval();

    WaveOrthotope w(file_exists(outFile) ? outFile : inFile);

 
    nthreads = std::min<int>(nthreads, (int)(w.rows() - 2));
    if (nthreads < 1) nthreads = 1;

    double E_STOP = 0.001 * w.interior_cells();

    ThreadPool pool(w, nthreads);
    auto last_ckpt = std::chrono::steady_clock::now();

    while (true) {
        pool.reset_queue();
        pool.phase_barrier.arrive_and_wait();
        pool.phase_barrier.arrive_and_wait();

        pool.reset_queue();
        pool.phase_barrier.arrive_and_wait();
        pool.phase_barrier.arrive_and_wait();

        pool.reset_queue();
        pool.phase_barrier.arrive_and_wait();
        pool.phase_barrier.arrive_and_wait();

        w.t += w.dt;

        if (w.energy() <= E_STOP)
            break;

        if (interval > 0.0) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_ckpt).count() >= interval) {
                atomic_write(w, outFile);
                atomic_write(w, make_checkpoint_name(w.time()).c_str());
                last_ckpt = now;
            }
        }
    }

    pool.keep_running.store(false, std::memory_order_release);

    for (int i = 0; i < 6; i++)
        pool.phase_barrier.arrive_and_wait();

    atomic_write(w, outFile);
    atomic_write(w, make_checkpoint_name(w.time()).c_str());
    return 0;
}
