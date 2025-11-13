#include <iostream>
#include <vector>
#include <cstdint>
#include <type_traits>
#include <algorithm>
#include <unistd.h>
#include <utility>      // for std::pair
#include <functional>   // for std::plus
#include <string>       // for std::string
#include <cstdlib>      // for std::getenv, std::strtod
#include <mpl/mpl.hpp>


namespace {
    template <class T>
    T read_at_all(auto &f, std::size_t offset) {       
        std::remove_const_t<T> x;
        f.read_at_all(offset, x);
        return x;
    }

    inline std::pair<std::uint64_t, std::uint64_t>     
    split_range(std::uint64_t n, int rank, int size) { 
        std::uint64_t base = n / size;
        std::uint64_t extra = n % size;
        std::uint64_t local = base + (rank < extra ? 1 : 0);
        std::uint64_t first = base * rank + (rank < extra ? rank : extra);
        return {first, first + local};
    }
}

int main(int argc, char *argv[]) {
    const auto &comm = mpl::environment::comm_world(); 
    int rank = comm.rank();
    int size = comm.size();

    if (argc != 3) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " input.wo output.wo\n";
        return 1;
    }

    using u64 = std::uint64_t;
    using f64 = double;

    mpl::file fin(comm, argv[1], mpl::file::access_mode::read_only);
    std::size_t offset = 0;

    u64 N = read_at_all<u64>(fin, offset);
    offset += sizeof(u64);

    std::vector<u64> m(N);
    for (u64 i = 0; i < N; ++i)
        m[i] = read_at_all<u64>(fin, offset + i * sizeof(u64));
    offset += N * sizeof(u64);

    f64 c = read_at_all<f64>(fin, offset);
    offset += sizeof(f64);
    f64 t = read_at_all<f64>(fin, offset);   // we will treat this as *remaining* time
    offset += sizeof(f64);

    std::size_t header_size = offset;

    u64 cells = 1;
    for (auto d : m) cells *= d;

    u64 rows = m[0];
    u64 cols = cells / rows;

    auto [row_first, row_last] = split_range(rows, rank, size);
    u64 real_row_first = row_first;
    u64 real_row_last  = row_last;
    u64 local_rows     = real_row_last - real_row_first;

    if (local_rows > 0) {
        if (row_first > 0) row_first -= 1;
        if (row_last < rows) row_last += 1;
    }

    u64 halo_row_first = row_first;
    u64 halo_row_last  = row_last;
    u64 halo_rows      = halo_row_last - halo_row_first;

    u64 start = halo_row_first * cols;
    u64 local = halo_rows * cols;

    if (local == 0)
        fin = mpl::file();

    std::vector<f64> u(local), v(local);
    auto layout_d = mpl::vector_layout<f64>(local);    

    u64 u_offset = header_size + sizeof(f64) * start;  
    u64 v_offset = header_size + sizeof(f64) * (cells + start);

    if (local > 0) {
        fin.read_at(u_offset, u.data(), layout_d);     
        fin.read_at(v_offset, v.data(), layout_d);     
    }

    std::string out_base   = argv[2];
    std::string ckpt_name  = out_base + ".ckpt";       

    auto write_state = [&](const std::string &fname) { 
        mpl::file fout(comm, fname.c_str(),
            mpl::file::access_mode::create |
            mpl::file::access_mode::write_only);       

        fout.write_all(N);
        for (u64 d : m) fout.write_all(d);
        fout.write_all(c);
        fout.write_all(t);  // current remaining time  

        if (local > 0) {
            u64 rf = real_row_first - halo_row_first;  
            u64 rl = real_row_last  - halo_row_first;  
            u64 real_count = (rl - rf) * cols;

            u64 u_write_offset = header_size + sizeof(f64) * (real_row_first * cols);
            u64 v_write_offset = header_size + sizeof(f64) * (cells + real_row_first * cols);

            auto real_layout = mpl::vector_layout<f64>(real_count);

            fout.write_at(u_write_offset, u.data() + rf*cols, real_layout);
            fout.write_at(v_write_offset, v.data() + rf*cols, real_layout);
        }
    };

    if (local > 0) {

        auto exchange_halos = [&](std::vector<f64> &x) {
            auto left_tag  = mpl::tag_t{0};
            auto right_tag = mpl::tag_t{1};
            auto row_layout = mpl::vector_layout<f64>(cols);

            if (real_row_first > 0) {
                comm.sendrecv(
                    x.data() + cols, row_layout, rank-1, left_tag, // send first real row
                    x.data(),        row_layout, rank-1, right_tag // receive into top halo
                );
            }

            if (real_row_last < rows) {
                u64 send_row = (real_row_last - 1) - halo_row_first;
                u64 recv_row = (halo_row_last - 1) - halo_row_first;
                comm.sendrecv(
                    x.data() + send_row*cols, row_layout, rank+1, right_tag,
                    x.data() + recv_row*cols, row_layout, rank+1, left_tag
                );
            }
        };

        auto energy = [&]() -> std::pair<f64, f64> {   
            u64 rf = real_row_first - halo_row_first;  
            u64 rl = real_row_last  - halo_row_first;  

            f64 local_dynamic   = 0.0; // from velocities v
            f64 local_potential = 0.0; // from displacements u

            for (u64 i = rf; i < rl; ++i)
                for (u64 j = 0; j < cols; ++j) {       
                    u64 idx = i*cols + j;
                    local_dynamic   += 0.5 * v[idx] * v[idx];
                    local_potential += 0.5 * u[idx] * u[idx];
                }

            f64 global_dynamic   = 0.0;
            f64 global_potential = 0.0;

            comm.allreduce(std::plus<>{}, local_dynamic,   global_dynamic);
            comm.allreduce(std::plus<>{}, local_potential, global_potential);

            return {global_dynamic, global_potential}; 
        };

        auto laplace = [&](u64 i, u64 j, const std::vector<f64> &x) {
            u64 idx = i*cols + j;
            f64 center = x[idx];
            f64 up     = x[(i-1)*cols + j];
            f64 down   = x[(i+1)*cols + j];
            f64 left   = (j>0      ? x[idx-1] : center);
            f64 right  = (j<cols-1 ? x[idx+1] : center);
            return up + down + left + right - 4.0*center;
        };

        auto solve = [&]() {
            f64 dt = 0.01;
            int steps = int(t / dt);   // t is remaining time

            if (steps <= 0)
                return;

            std::vector<f64> u_new(local), v_new(local);

            u64 rf = real_row_first - halo_row_first;  
            u64 rl = real_row_last  - halo_row_first;  

            exchange_halos(u);
            exchange_halos(v);

           
            int checkpoint_interval = std::max(1, steps / 10);

            if (const char *env = std::getenv("INTVL")) {
                char *endp = nullptr;
                double intvl_time = std::strtod(env, &endp);
                if (endp != env && intvl_time > 0.0) { 
                    int ci = int(intvl_time / dt + 0.5); // round to nearest step
                    if (ci < 1) ci = 1;
                    checkpoint_interval = ci;
                }
            }
      

            f64 remaining = t;

            for (int step = 0; step < steps; ++step) { 

                for (u64 i = rf; i < rl; ++i)
                    for (u64 j = 0; j < cols; ++j) {   
                        u64 idx = i*cols + j;
                        f64 L = laplace(i, j, u);      
                        f64 vtemp = v[idx] + dt * (L - c * v[idx]);
                        v_new[idx] = vtemp;
                        u_new[idx] = u[idx] + dt * vtemp;
                    }

                u.swap(u_new);
                v.swap(v_new);

                exchange_halos(u);
                exchange_halos(v);

                remaining -= dt;
                if (remaining < 0.0) remaining = 0.0;  
                t = remaining;  // update header field for future writes

                if ((step + 1) % checkpoint_interval == 0) {
                    write_state(ckpt_name);
                }
            }
        };

        auto [dyn0, pot0] = energy();
        if (rank == 0) {
            std::cout << "Initial energies (before stepping): "
                      << "dynamic = " << dyn0
                      << ", potential = " << pot0 << "\n";
        }

        solve();

        auto [dyn1, pot1] = energy();
        if (rank == 0) {
            std::cout << "Energies after solve: "      
                      << "dynamic = " << dyn1
                      << ", potential = " << pot1 << "\n";
        }
    }

    write_state(out_base);

    return 0;
}




