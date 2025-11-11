// optimize.cpp
// Build & run (as you were using):
// g++ -Ofast -std=c++20 -fopenmp -march=native -mtune=native optimize.cpp -o optimize
// export OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=cores
// time ./optimize


#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstddef>//new
#include <algorithm>//new


static inline std::size_t idx(std::size_t i, std::size_t j, std::size_t n) {
    return i * n + j;
}


static inline double energy_floor(std::size_t m, std::size_t n) {
    return (m - 2) * (n - 2) * 0.001;
}


static inline double laplacian_flat(const double* __restrict u,
                                    std::size_t i, std::size_t j, std::size_t n)
{
    const std::size_t c = idx(i, j, n);
    return (u[c - 1] + u[c + 1] + u[c - n] + u[c + n]) * 0.5 - 2.0 * u[c];
}


static double energy_flat(const double* __restrict u,
                          const double* __restrict v,
                          std::size_t m, std::size_t n)
{
    double E = 0.0;


    #pragma omp parallel for reduction(+:E)
    for (std::size_t i = 1; i < m - 1; ++i) {
        const std::size_t base = i * n;
        #pragma omp simd reduction(+:E)
        for (std::size_t j = 1; j < n - 1; ++j) {
            const double vij = v[base + j];
            E += 0.5 * vij * vij;
        }
    }


    #pragma omp parallel for reduction(+:E)
    for (std::size_t i = 0; i < m - 1; ++i) {
        const std::size_t base = i * n;
        const std::size_t base2 = (i + 1) * n;
        #pragma omp simd reduction(+:E)
        for (std::size_t j = 1; j < n - 1; ++j) {
            const double d = u[base + j] - u[base2 + j];
            E += 0.25 * d * d;
        }
    }


    #pragma omp parallel for reduction(+:E)
    for (std::size_t i = 1; i < m - 1; ++i) {
        const std::size_t base = i * n;
        #pragma omp simd reduction(+:E)
        for (std::size_t j = 0; j < n - 1; ++j) {
            const double d = u[base + j] - u[base + j + 1];
            E += 0.25 * d * d;
        }
    }


    return E;
}


static inline void mirror_quadrants_flat(double* __restrict a,
                                         std::size_t m, std::size_t n)
{
    const std::size_t half_i = m / 2;
    const std::size_t half_j = n / 2;


    #pragma omp parallel for collapse(2)
    for (std::size_t i = 1; i <= half_i - 1; ++i) {
        for (std::size_t j = 1; j <= half_j - 1; ++j) {
            const double val = a[idx(i, j, n)];
            a[idx(m - 1 - i, j, n)]         = val; // bottom-left
            a[idx(i, n - 1 - j, n)]         = val; // top-right
            a[idx(m - 1 - i, n - 1 - j, n)] = val; // bottom-right
        }
    }


    if (half_i < m - 1) {
        #pragma omp parallel for
        for (std::size_t j = 1; j <= half_j - 1; ++j) {
            a[idx(half_i, j, n)]             = a[idx(half_i - 1, j, n)];
            a[idx(half_i, n - 1 - j, n)]     = a[idx(half_i - 1, n - 1 - j, n)];
        }
    }
    if (half_j < n - 1) {
        #pragma omp parallel for
        for (std::size_t i = 1; i <= half_i - 1; ++i) {
            a[idx(i, half_j, n)]             = a[idx(i, half_j - 1, n)];
            a[idx(m - 1 - i, half_j, n)]     = a[idx(m - 1 - i, half_j - 1, n)];
        }
    }
}


static inline void step_quadrant_flat(double* __restrict u,
                                      double* __restrict v,
                                      std::size_t m, std::size_t n,
                                      double c, double dt)
{
    const std::size_t half_i = m / 2;
    const std::size_t half_j = n / 2;


    #pragma omp parallel for
    for (std::size_t i = 1; i <= half_i - 1; ++i) {
        #pragma omp simd
        for (std::size_t j = 1; j <= half_j - 1; ++j) {
            const double L = laplacian_flat(u, i, j, n);
            const std::size_t p = idx(i, j, n);
            v[p] = (1.0 - dt * c) * v[p] + dt * L;
        }
    }


    #pragma omp parallel for
    for (std::size_t i = 1; i <= half_i - 1; ++i) {
        #pragma omp simd
        for (std::size_t j = 1; j <= half_j - 1; ++j) {
            const std::size_t p = idx(i, j, n);
            u[p] += dt * v[p];
        }
    }


    mirror_quadrants_flat(u, m, n);
    mirror_quadrants_flat(v, m, n);
}


int main() {
    const std::size_t rows = 800;   // even for quadrant symmetry
    const double c  = 0.05;
    const double dt = 0.01;
    const double u0 = 1.0;


    std::vector<double> u(rows * rows, 0.0);
    std::vector<double> v(rows * rows, 0.0);
    #pragma omp parallel for
    for (std::size_t i = 1; i < rows - 1; ++i) {
        const std::size_t base = i * rows;
        #pragma omp simd
        for (std::size_t j = 1; j < rows - 1; ++j) {
            u[base + j] = u0;
        }
    }


    mirror_quadrants_flat(u.data(), rows, rows);
    mirror_quadrants_flat(v.data(), rows, rows);


    double t = 0.0;
    const double E_floor = energy_floor(rows, rows);


    while (energy_flat(u.data(), v.data(), rows, rows) > E_floor) {
        step_quadrant_flat(u.data(), v.data(), rows, rows, c, dt);
        t += dt;
    }


    t = std::round(t * 100.0) / 100.0; // same rounding/format
    std::cout << std::fixed << std::setprecision(2) << t << '\n';
    return 0;
}




