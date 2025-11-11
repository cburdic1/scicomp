//cd ~/scicomp
//rm -f optimize
//g++ -Ofast -std=c++20 -fopenmp -march=native -mtune=native optimize.cpp -o //optimize
//export OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=cores
//time ./optimize
//INRUN


//salloc -p m9 –acount=ryancox --nodes 1 --ntasks 28 --mem 16G --time 00:15:00 <<’INRUN’


//above is what i compile the code with to get such low run time, please use this and in //this order. (maybe without the ryancox part, because that's because I use the //supercomputer for another purpose) 
#pragma once
#include <omp.h>
#include <iostream>
#include <vector>
#include <span>
#include <cmath>
#include <iomanip>

auto interior(auto& x) {
    return std::span(x.begin()+1, x.end()-1);
}

auto interior(const auto& x) {
    return std::span(x.begin()+1, x.end()-1);
}

auto laplacian(const auto& x, auto i, auto j) {
    return (x[i][j-1] + x[i][j+1] + x[i-1][j] + x[i+1][j]) / 2 - 2 * x[i][j];
}

auto energy_floor(const auto& u) {
    return (u.size() - 2) * (u.front().size() - 2) * 0.001;
}

auto energy(const auto& u, const auto& v) {
    double E{};
    auto m = u.size(), n = u.front().size();

 
    for (const auto& row: interior(v)) {
        for (auto v_ij: interior(row)) {
            E += std::pow(v_ij, 2) / 2;
        }
    }

   
    #pragma omp parallel for reduction(+:E)
    for (size_t i = 0; i < m-1; ++i) {
        #pragma omp simd reduction(+:E)
        for (size_t j = 1; j < n-1; ++j) {
            double d = u[i][j] - u[i+1][j];
            E += (d*d) * 0.25;
        }
    }

   
    #pragma omp parallel for reduction(+:E)
    for (size_t i = 1; i < m-1; ++i) {
        #pragma omp simd reduction(+:E)
        for (size_t j = 0; j < n-1; ++j) {
            double d = u[i][j] - u[i][j+1];
            E += (d*d) * 0.25;
        }
    }

    return E;
}

auto step(auto &u, auto &v, auto c, auto dt) {
    auto m = u.size(), n = u.front().size();


    #pragma omp parallel for
    for (size_t i = 1; i < m-1; ++i) {
        #pragma omp simd
        for (size_t j = 1; j < n-1; ++j) {
            auto L = laplacian(u, i, j);
            v[i][j] = (1 - dt * c) * v[i][j] + dt * L;
        }
    }


    #pragma omp parallel for
    for (size_t i = 1; i < m-1; ++i) {
        #pragma omp simd
        for (size_t j = 1; j < n-1; ++j) {
            u[i][j] += dt * v[i][j];
        }
    }
}

int main() {
 
    const int rows = 800;
    const double c = 0.05,
                 dt = 0.01,
                 u0 = 1, v0 = 0;
    double t = 0;

   
    auto u = std::vector<std::vector<double>>(rows, std::vector<double>(rows));
    auto v = u;
    for (auto &row: interior(u)) std::fill(interior(row).begin(), interior(row).end(), u0);


    while (energy(u, v) > energy_floor(u)) {
        step(u, v, c, dt);
        t += dt;
    }

     t = std::round(t * 100.0) / 100.0;
    std::cout << std::fixed << std::setprecision(2) << t << '\n';
    return 0;

}
