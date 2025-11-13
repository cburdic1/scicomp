#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <string>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <barrier>
#include <filesystem>
#include "WaveOrthotope.hpp"

namespace fs = std::filesystem;

static double parse_interval_env() {
    const char* s = std::getenv("INTVL");
    if (!s || !*s) return -1.0;
    char* endp = nullptr;
    double val = std::strtod(s, &endp);
    if (endp == s || !std::isfinite(val) || val <= 0.0) return -1.0;
    return val;
}

static int decide_threads_from_args_env(int, char**) {
    if (const char* s = std::getenv("SOLVER_NUM_THREADS")) {
        int n = std::atoi(s);
        if (n > 0) return n;
    }
    return 1;
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
        if (ec) {
            std::cerr << "Error: checkpoint rename failed: " << ec.message() << "\n";
        }
    }
}

/************* JTHREAD THREAD POOL WITH SHARED CHUNK QUEUE *************/
struct ThreadPool {
    struct Chunk {
        std::size_t first_row;
        std::size_t last_row;
    };

    WaveOrthotope& w;
    std::atomic<bool> running{true};
    std::barrier<> barrier;
    std::vector<std::jthread> workers;
    std::vector<Chunk> chunks;
    std::atomic<std::size_t> next_chunk{0};
    std::size_t nthreads;

    ThreadPool(std::size_t n, WaveOrthotope& w_)
        : w(w_), barrier(n + 1), nthreads(n)
    {
        const std::size_t R = w.rows();
        const std::size_t C = w.cols();
        if (R < 3 || C < 3) throw std::runtime_error("Domain must be at least 3x3");

        const std::size_t interior_rows = R - 2;
        std::size_t nchunks = (nthreads > 0 ? nthreads : 1);
        std::size_t base = interior_rows / nchunks;
        std::size_t extra = interior_rows % nchunks;

        chunks.reserve(nchunks);
        std::size_t row = 1;
        for (std::size_t c = 0; c < nchunks; ++c) {
            std::size_t len = base + (c < extra ? 1 : 0);
            if (len == 0) continue;
            chunks.push_back({row, row + len});
            row += len;
        }

        workers.reserve(nthreads);
        for (std::size_t tid = 0; tid < nthreads; ++tid) {
            workers.emplace_back([this]{
                const std::size_t R = w.rows();
                const std::size_t C = w.cols();
                while (true) {

                    // PHASE 1 (fixed ordering)
                    barrier.arrive_and_wait();
                    run_laplacian(R, C);
                    barrier.arrive_and_wait();
                    if (!running.load(std::memory_order_acquire)) break;

                    // PHASE 2 (fixed ordering)
                    barrier.arrive_and_wait();
                    run_velocity(R, C);
                    barrier.arrive_and_wait();
                    if (!running.load(std::memory_order_acquire)) break;

                    // PHASE 3 (fixed ordering)
                    barrier.arrive_and_wait();
                    run_displacement(R, C);
                    barrier.arrive_and_wait();
                    if (!running.load(std::memory_order_acquire)) break;
                }
            });
        }
    }

    void reset_queue() {
        next_chunk.store(0, std::memory_order_release);
    }

    void run_laplacian(std::size_t R, std::size_t C) {
        for (;;) {
            std::size_t idx = next_chunk.fetch_add(1, std::memory_order_acq_rel);
            if (idx >= chunks.size()) break;
            auto [first, last] = chunks[idx];
            for (std::size_t i = first; i < last; ++i) {
                for (std::size_t j = 1; j < C - 1; ++j) {
                    std::size_t k = i * C + j;
                    w.lap[k] = 0.5 * (
                        w.u[(i - 1) * C + j] + w.u[(i + 1) * C + j] +
                        w.u[i * C + (j - 1)] + w.u[i * C + (j + 1)] -
                        4.0 * w.u[k]
                    );
                }
            }
        }
    }

    void run_velocity(std::size_t R, std::size_t C) {
        for (;;) {
            std::size_t idx = next_chunk.fetch_add(1, std::memory_order_acq_rel);
            if (idx >= chunks.size()) break;
            auto [first, last] = chunks[idx];
            for (std::size_t i = first; i < last; ++i) {
                for (std::size_t j = 1; j < C - 1; ++j) {
                    std::size_t k = i * C + j;
                    w.v[k] += (w.c2 * w.lap[k] - w.c * w.v[k]) * w.dt;
                }
            }
        }
    }

    void run_displacement(std::size_t R, std::size_t C) {
        for (;;) {
            std::size_t idx = next_chunk.fetch_add(1, std::memory_order_acq_rel);
            if (idx >= chunks.size()) break;
            auto [first, last] = chunks[idx];
            for (std::size_t i = first; i < last; ++i) {
                for (std::size_t j = 1; j < C - 1; ++j) {
                    std::size_t k = i * C + j;
                    w.u[k] += w.v[k] * w.dt;
                }
            }
        }
    }
};
/***********************************************************************/

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "wavesolve_thread")
                  << " <input.wo> <output.wo>\n";
        return 1;
    }

    const int threads = decide_threads_from_args_env(argc, argv);
    const char* inFile  = argv[1];
    const char* outFile = argv[2];

    WaveOrthotope w(file_exists(outFile) ? outFile : inFile);

    const double E_STOP  = 0.001 * static_cast<double>(w.interior_cells());
    const double interval = parse_interval_env();

    auto last_ckpt = std::chrono::steady_clock::now();

    ThreadPool pool(threads, w);

    while (w.energy() > E_STOP) {

        // PHASE 1
        pool.reset_queue();
        pool.barrier.arrive_and_wait();
        pool.barrier.arrive_and_wait();

        // PHASE 2
        pool.reset_queue();
        pool.barrier.arrive_and_wait();
        pool.barrier.arrive_and_wait();

        // PHASE 3
        pool.reset_queue();
        pool.barrier.arrive_and_wait();
        pool.barrier.arrive_and_wait();

        w.t += w.dt;

        if (interval > 0.0) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_ckpt).count() >= interval) {
                atomic_write(w, outFile);
                auto chk = make_checkpoint_name(w.time());
                atomic_write(w, chk.c_str());
                last_ckpt = now;
            }
        }
    }

    pool.running.store(false, std::memory_order_release);

    // Flush workers through barriers
    pool.barrier.arrive_and_wait();
    pool.barrier.arrive_and_wait();
    pool.barrier.arrive_and_wait();
    pool.barrier.arrive_and_wait();
    pool.barrier.arrive_and_wait();
    pool.barrier.arrive_and_wait();

    atomic_write(w, outFile);
    auto final_chk = make_checkpoint_name(w.time());
    atomic_write(w, final_chk.c_str());

    return 0;
}
