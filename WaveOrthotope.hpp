#pragma once
#include <cmath>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <string>

class WaveOrthotope {
public:
    std::uint64_t N = 0;
    std::vector<std::uint64_t> m;
    double c = 0.0;
    double t = 0.0;
    std::vector<double> u, v, lap;

    double dt = 0.01;
    double c2 = 1.0;

    explicit WaveOrthotope(const char* filename) {
        std::ifstream is(filename, std::ios::binary);
        if (!is) throw std::runtime_error("bad input file");

        read(is, N);
        if (N != 2) throw std::runtime_error("only 2D supported");

        m.resize(2);
        read(is, m[0]);
        read(is, m[1]);

        read(is, c);
        read(is, t);

        std::size_t total = m[0] * m[1];
        u.resize(total);
        v.resize(total);
        lap.resize(total);

        is.read((char*)u.data(), total * sizeof(double));
        is.read((char*)v.data(), total * sizeof(double));
    }

    void write(const char* filename) const {
        std::ofstream os(filename, std::ios::binary | std::ios::trunc);
        if (!os) throw std::runtime_error("bad output file");

        write(os, N);
        write(os, m[0]);
        write(os, m[1]);
        write(os, c);
        write(os, t);

        os.write((char*)u.data(), u.size() * sizeof(double));
        os.write((char*)v.data(), v.size() * sizeof(double));
    }

    inline std::size_t rows() const { return m[0]; }
    inline std::size_t cols() const { return m[1]; }
    inline std::size_t interior_cells() const {
        return (rows() - 2) * (cols() - 2);
    }
    inline double time() const { return t; }

    double energy() const {
        double E = 0.0;
        std::size_t R = rows(), C = cols();
        const double* ud = u.data();
        const double* vd = v.data();


        for (std::size_t i = 1; i < R - 1; i++) {
            std::size_t io = i * C;
            for (std::size_t j = 1; j < C - 1; j++) {
                double vv = vd[io + j];
                E += 0.5 * vv * vv;
            }
        }

   
        for (std::size_t i = 1; i < R - 2; i++) {
            std::size_t io = i * C, ip = (i + 1) * C;
            for (std::size_t j = 1; j < C - 1; j++) {
                double d = ud[io + j] - ud[ip + j];
                E += 0.25 * d * d;
            }
        }

       
        for (std::size_t i = 1; i < R - 1; i++) {
            std::size_t io = i * C;
            for (std::size_t j = 1; j < C - 2; j++) {
                double d = ud[io + j] - ud[io + j + 1];
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

inline std::string make_checkpoint_name(double t) {
    std::ostringstream ss;
    ss << "chk-" << std::setw(7) << std::setfill('0')
       << std::fixed << std::setprecision(2) << t << ".wo";
    return ss.str();
}
