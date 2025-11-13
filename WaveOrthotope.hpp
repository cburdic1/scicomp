#pragma once
#include <cmath>//
#include <iomanip>//
#include <vector>//
#include <cstdint>//
#include <fstream>//
#include <stdexcept>//
#include <sstream>//
#include <string>//
#include <omp.h>

class WaveOrthotope {
public:
    std::uint64_t N=0;
    std::vector<std::uint64_t> m;
    double c = 0.0;
    double t = 0.0;
    std::vector<double> u, v;
    std::vector<double> lap;

    double dt = 0.01;
    double c2 = 1.0;

    explicit WaveOrthotope(const char* filename) {
        std::ifstream is(filename, std::ios::binary);
        if (!is) throw std::runtime_error("bad input file");
       
        read(is, N);
        if (N != 2) throw std::runtime_error("only 2D supported");

        m.resize(N);
        std::size_t idx_dim = 0;
        while (idx_dim < m.size()) {
            read(is, m[idx_dim]);
            idx_dim++;
        }

        read(is, c);
        read(is, t);

        std::size_t total = m[0] * m[1];
        u.resize(total);
        v.resize(total);
        lap.resize(total);

        is.read(reinterpret_cast<char*>(u.data()), total * sizeof(double));
        is.read(reinterpret_cast<char*>(v.data()), total * sizeof(double));
    }

    void write(const char* filename) const {
        std::ofstream os(filename, std::ios::binary | std::ios::trunc);
        if (!os) throw std::runtime_error("bad output file");

        write(os, N);

        std::size_t idx_dim = 0;
        while (idx_dim < m.size()) {
            write(os, m[idx_dim]);
            idx_dim++;
        }

        write(os, c);
        write(os, t);
        os.write(reinterpret_cast<const char*>(u.data()), u.size() * sizeof(double));
        os.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(double));
    }

    inline std::size_t rows() const { return m[0]; }
    inline std::size_t cols() const { return m[1]; }
    inline std::size_t idx(std::size_t i, std::size_t j) const { return i * cols() + j; }
    inline std::size_t interior() const { return (rows() - 2) * (cols() - 2); }

    inline std::size_t interior_cells() const { return interior(); }
    inline double time() const { return t; }


    void step() {
        const std::size_t R = rows();
        const std::size_t C = cols();

        const double* __restrict ud = u.data();
        double* __restrict lapd = lap.data();

  
        #pragma omp parallel for
        for (std::size_t i = 1; i < R - 1; ++i) {
            const std::size_t im1 = (i - 1) * C;
            const std::size_t i0  =  i      * C;
            const std::size_t ip1 = (i + 1) * C;

            #pragma omp simd
            for (std::size_t j = 1; j < C - 1; ++j) {
                const std::size_t k = i0 + j;
                lapd[k] = 0.5 * (ud[im1 + j] + ud[ip1 + j] +
                                 ud[i0 + (j - 1)] + ud[i0 + (j + 1)] -
                                 4.0 * ud[k]);
            }
        }

    
        double* __restrict vd = v.data();
        #pragma omp parallel for
        for (std::size_t i = 1; i < R - 1; ++i) {
            const std::size_t i0 = i * C;

            #pragma omp simd
            for (std::size_t j = 1; j < C - 1; ++j) {
                const std::size_t k = i0 + j;
                vd[k] += (c2 * lapd[k] - c * vd[k]) * dt;
            }
        }

   
        double* __restrict udw = u.data();
        #pragma omp parallel for
        for (std::size_t i = 1; i < R - 1; ++i) {
            const std::size_t i0 = i * C;

            #pragma omp simd
            for (std::size_t j = 1; j < C - 1; ++j) {
                const std::size_t k = i0 + j;
                udw[k] += vd[k] * dt;
            }
        }

        t += dt;
    }


    double energy() const {
        double E = 0.0;
        const std::size_t R = rows();
        const std::size_t C = cols();

        const double* __restrict ud = u.data();
        const double* __restrict vd = v.data();

       
        #pragma omp parallel for reduction(+:E)
        for (std::size_t i = 1; i < R - 1; ++i) {
            const std::size_t i0 = i * C;

            #pragma omp simd reduction(+:E)
            for (std::size_t j = 1; j < C - 1; ++j) {
                const double vij = vd[i0 + j];
                E += 0.5 * vij * vij;
            }
        }

      
        #pragma omp parallel for reduction(+:E)
        for (std::size_t i = 0; i < R - 1; ++i) {
            const std::size_t i0  =  i      * C;
            const std::size_t ip1 = (i + 1) * C;

            #pragma omp simd reduction(+:E)
            for (std::size_t j = 1; j < C - 1; ++j) {
                const double d = ud[i0 + j] - ud[ip1 + j];
                E += 0.25 * d * d;
            }
        }

 
        #pragma omp parallel for reduction(+:E)
        for (std::size_t i = 1; i < R - 1; ++i) {
            const std::size_t i0 = i * C;

            #pragma omp simd reduction(+:E)
            for (std::size_t j = 0; j < C - 1; ++j) {
                const double d = ud[i0 + j] - ud[i0 + (j + 1)];
                E += 0.25 * d * d;
            }
        }

        return E;
    }

private:
    template<class T>
    static void read(std::ifstream& is, T& x) {
        is.read(reinterpret_cast<char*>(&x), sizeof(T));
    }
    template<class T>
    static void write(std::ofstream& os, const T& x) {
        os.write(reinterpret_cast<const char*>(&x), sizeof(T));
    }
};

inline std::string checkpoint_name(double t) {
    std::ostringstream ss;
    ss << "chk-" << std::setw(7) << std::setfill('0') << std::fixed
       << std::setprecision(2) << t << ".wo";
    return ss.str();
}
inline std::string make_checkpoint_name(double t) {
    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss << "chk-";
    ss << std::setw(7) << std::setfill('0') << std::fixed << std::setprecision(2) << t;
    ss << ".wo";
    return ss.str();
}
