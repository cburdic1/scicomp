#include <iostream>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include <filesystem>
#include <barrier>
#include <chrono>
#include <atomic>
#include "WaveOrthotope.hpp"

namespace fs = std::filesystem;

static int get_threads() {
    if (const char* s = std::getenv("SOLVER_NUM_THREADS")) {
        int t = std::atoi(s);
        return (t > 0 ? t : 1);
    }
    return 1;
}

static double parse_interval_env() {        
    const char* s = std::getenv("INTVL");   
    if (!s || !*s) return -1.0;
    double v = std::atof(s);
    return (v > 0.0 ? v : -1.0);
}

static void atomic_write(const WaveOrthotope& w, const char* out) {
    std::string tmp = std::string(out) + ".tmp";
    w.write(tmp.c_str());
    fs::rename(tmp, out);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: wavesolve_thread in.wo out.wo\n";
        return 1;
    }

    const char* infile = argv[1];
    const char* outfile = argv[2];

    WaveOrthotope w(infile);

    const double E_STOP = 0.001 * static_cast<double>(w.interior_cells());
    const double dt = w.dt;

    double& t = w.t;

    double interval = parse_interval_env(); 
    auto last_ckpt = std::chrono::steady_clock::now();

    const size_t R = w.rows();
    const size_t C = w.cols();
    const int T = get_threads();

    size_t work = R - 2;
    size_t chunk = work / T;
    size_t rem = work % T;

    struct Chunk { size_t i0, i1; };        
    std::vector<Chunk> chunks(T);

    size_t cur = 1;
    for (int th = 0; th < T; th++) {        
        size_t size = chunk + (th < rem ? 1 : 0);
        chunks[th] = { cur, cur + size };   
        cur += size;
    }

    std::barrier sync(T);
    std::atomic<bool> done(false);

    auto worker = [&](int tid) {
        size_t i0 = chunks[tid].i0;
        size_t i1 = chunks[tid].i1;

        while (true) {
            if (tid == 0) {
                double E = w.energy();      
                if (E <= E_STOP) {
                    done.store(true, std::memory_order_relaxed);
                } else {
                    done.store(false, std::memory_order_relaxed);
                }
            }

            sync.arrive_and_wait();

            if (done.load(std::memory_order_relaxed))
                return;

            double* u = w.u.data();
            double* v = w.v.data();
            double* lap = w.lap.data();     
            const double* ud = w.u.data();

            for (size_t i = i0; i < i1; i++) {
                size_t im1 = (i - 1) * C;   
                size_t ic  =  i      * C;   
                size_t ip1 = (i + 1) * C;   

                for (size_t j = 1; j < C - 1; j++) {
                    size_t k = ic + j;      
                    lap[k] = 0.5 * (ud[im1 + j] + ud[ip1 + j]
                                   + ud[ic + j - 1] + ud[ic + j + 1]
                                   - 4.0 * ud[k]);
                }
            }

            sync.arrive_and_wait();

            for (size_t i = i0; i < i1; i++) {
                size_t ic = i * C;
                for (size_t j = 1; j < C - 1; j++) {
                    size_t k = ic + j;      
                    v[k] += (w.c2 * lap[k] - w.c * v[k]) * dt;
                }
            }

            sync.arrive_and_wait();

            for (size_t i = i0; i < i1; i++) {
                size_t ic = i * C;
                for (size_t j = 1; j < C - 1; j++) {
                    size_t k = ic + j;      
                    u[k] += v[k] * dt;      
                }
            }

            sync.arrive_and_wait();

            if (tid == 0) {
                t += dt;

                if (interval > 0.0) {       
                    auto now = std::chrono::steady_clock::now();
                    double elapsed = std::chrono::duration<double>(now - last_ckpt).count();
                    if (elapsed >= interval) {
                        atomic_write(w, outfile);
                        std::string chk = make_checkpoint_name(w.time());
                        atomic_write(w, chk.c_str());
                        last_ckpt = now;    
                    }
                }
            }

            sync.arrive_and_wait();
        }
    };

    std::vector<std::thread> threads;       
    threads.reserve(T);
    for (int i = 0; i < T; i++)
        threads.emplace_back(worker, i);    
    for (auto& th : threads)
        th.join();

    atomic_write(w, outfile);
    return 0;
}
