#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <string>
#include "WaveOrthotope.hpp"
#include <chrono>
#include <omp.h>
#include <filesystem>

namespace fs = std::filesystem;

static double parse_interval_env() {
    const char* s = std::getenv("INTVL");
    if (!s || !*s) return -1.0;
    char* endp = nullptr;
    double val = std::strtod(s, &endp);
    if (endp == s || !std::isfinite(val) || val <= 0.0) return -1.0;
    return val;
}
static int decide_threads_from_args_env(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--threads") == 0) {
            int n = std::atoi(argv[i+1]);
            if (n > 0) return n;
        }
    }
    if (const char* env_n = std::getenv("OMP_NUM_THREADS")) {
        int n = std::atoi(env_n);
        if (n > 0) return n;
    }
    return 8;
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


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "wavesolve_openmp")
                  << " <input.wo> <output.wo> [--threads N]\n";
        return 1;
    }
     const int threads = decide_threads_from_args_env(argc, argv);
    omp_set_dynamic(0);  
    omp_set_num_threads(threads); 

    #ifdef _OPENMP
    omp_set_max_active_levels(1); 
    omp_set_schedule(omp_sched_static, 0);


   #endif

    const char* inFile  = argv[1];
    const char* outFile = argv[2];
    

     WaveOrthotope w(file_exists(outFile) ? outFile : inFile);


    const double E_STOP  = 0.001 * static_cast<double>(w.interior_cells());
    const double interval = parse_interval_env();

    auto wall0 = std::chrono::steady_clock::now();
    auto last_ckpt = wall0;


    while (w.energy() > E_STOP) {
        w.step();

  
        if (interval > 0.0) {
            auto now = std::chrono::steady_clock::now();
            double since = std::chrono::duration<double>(now - last_ckpt).count();
            if (since >= interval) {
                atomic_write(w, outFile);  
                auto chk = make_checkpoint_name(w.time());
                atomic_write(w, chk.c_str());


                last_ckpt = now;
            }
        }
    }

    atomic_write(w, outFile);
   auto final_chk = make_checkpoint_name(w.time());
   atomic_write(w, final_chk.c_str());

    return 0; 
}




