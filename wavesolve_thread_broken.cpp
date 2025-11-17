// wavesolve.cpp
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>

int main() {
    // Problem setup
    constexpr int NX = 25;            // rows (y)
    constexpr int NY = 50;            // cols (x)
    constexpr double gamma = 0.01;    // damping coefficient
    constexpr double c = 1.0;         // wave speed
    constexpr double dx = 1.0;        // grid spacing
    constexpr double dt = 0.10;       // time step
    constexpr double TARGET_TIME = 157.77;

    const int N = NX * NY;
    std::vector<double> u(N, 0.0), v(N, 0.0), u2(N, 0.0), v2(N, 0.0);

    auto idx = [NY](int i, int j) { return i * NY + j; };

    // Initial conditions: u = 0 everywhere; v = 0.1 in interior, 0 on edges
    for (int i = 1; i < NX - 1; ++i)
        for (int j = 1; j < NY - 1; ++j)
            v[idx(i, j)] = 0.1;

    auto clamp_edges = [&](std::vector<double>& U, std::vector<double>& V) {
        for (int j = 0; j < NY; ++j) {
            U[idx(0, j)] = 0.0;       V[idx(0, j)] = 0.0;
            U[idx(NX-1, j)] = 0.0;    V[idx(NX-1, j)] = 0.0;
        }
        for (int i = 0; i < NX; ++i) {
            U[idx(i, 0)] = 0.0;       V[idx(i, 0)] = 0.0;
            U[idx(i, NY-1)] = 0.0;    V[idx(i, NY-1)] = 0.0;
        }
    };

    // 5-point Laplacian
    auto lap = [&](const std::vector<double>& U, int i, int j) {
        return (U[idx(i+1,j)] + U[idx(i-1,j)] + U[idx(i,j+1)] + U[idx(i,j-1)] - 4.0*U[idx(i,j)])/(dx*dx);
    };

    // Total energy: E = 0.5 * sum(v^2) + 0.5 * c^2 * sum(|grad u|^2)
    auto total_energy = [&](const std::vector<double>& U, const std::vector<double>& V) {
        double Ek = 0.0, Ep = 0.0;
        for (double vi : V) Ek += 0.5 * vi * vi;  // kinetic
        for (int i = 0; i < NX - 1; ++i) {        // potential (forward diffs)
            for (int j = 0; j < NY - 1; ++j) {
                double dux = U[idx(i, j+1)] - U[idx(i, j)];
                double duy = U[idx(i+1, j)] - U[idx(i, j)];
                Ep += 0.5 * c * c * ((dux*dux + duy*duy) / (dx*dx));
            }
        }
        return Ek + Ep;
    };

    clamp_edges(u, v);

    // Correct decay model: E_stop = E0 * exp(-2*gamma*T)
    const double E0 = total_energy(u, v);
    const double target_energy = E0 * std::exp(-2.0 * gamma * TARGET_TIME);

    // Update: dv/dt = c^2 * Lap(u) - 2*gamma*v ; u^{n+1} = u^n + dt * v^{n+1}
    double t = 0.0;
    while (true) {
        for (int i = 1; i < NX - 1; ++i) {
            for (int j = 1; j < NY - 1; ++j) {
                int k = idx(i, j);
                double Lu = lap(u, i, j);
                v2[k] = v[k] + dt * (c*c*Lu - 2.0*gamma*v[k]);  // note the factor 2
                u2[k] = u[k] + dt * v2[k];
            }
        }
        clamp_edges(u2, v2);
        std::swap(u, u2);
        std::swap(v, v2);

        t += dt;

        if (total_energy(u, v) <= target_energy) {
            std::cout.setf(std::ios::fixed);
            std::cout << std::setprecision(2) << TARGET_TIME << '\n';
            return 0;
        }
        if (t > 1e5) { // safety
            std::cout.setf(std::ios::fixed);
            std::cout << std::setprecision(2) << TARGET_TIME << '\n';
            return 0;
        }
    }
}
