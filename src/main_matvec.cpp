// main_matvec.cpp
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include <universal/number/posit/posit.hpp>
using namespace sw::universal;
using namespace std;

#include "kernel.hpp"
#include "stats.hpp"

enum class DistType {
    Normal01 = 0,
    UniformMinus1To1 = 1,
    NormalSmall = 2,
    NormalLarge = 3
};

vector<double> gen_vector(int n, mt19937 &rng, DistType dist_type) {
    vector<double> v(n);
    if (dist_type == DistType::Normal01) {
        normal_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < n; ++i) v[i] = dist(rng);
    } else if (dist_type == DistType::UniformMinus1To1) {
        uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int i = 0; i < n; ++i) v[i] = dist(rng);
    } else if (dist_type == DistType::NormalSmall) {
        normal_distribution<double> dist(0.0, 1e-3);
        for (int i = 0; i < n; ++i) v[i] = dist(rng);
    } else if (dist_type == DistType::NormalLarge) {
        normal_distribution<double> dist(0.0, 1e3);
        for (int i = 0; i < n; ++i) v[i] = dist(rng);
    }
    return v;
}

vector<vector<double>> gen_matrix(int rows, int cols, mt19937 &rng, DistType dist_type) {
    vector<vector<double>> A(rows, vector<double>(cols));
    for (int r = 0; r < rows; ++r) {
        A[r] = gen_vector(cols, rng, dist_type);
    }
    return A;
}

template<typename Real>
vector<Real> convert_vec(const vector<double> &src) {
    vector<Real> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = static_cast<Real>(src[i]);
    }
    return dst;
}

template<typename Real>
vector<vector<Real>> convert_mat(const vector<vector<double>> &src) {
    vector<vector<Real>> dst(src.size());
    for (size_t r = 0; r < src.size(); ++r) {
        dst[r] = convert_vec<Real>(src[r]);
    }
    return dst;
}

double matvec_sum_double(const vector<vector<double>> &A, const vector<double> &x) {
    return matvec_sum<double>(A, x);
}

template<int N, int ES>
double matvec_sum_posit(const vector<vector<double>> &A, const vector<double> &x) {
    using p = posit<N, ES>;
    auto Ap = convert_mat<p>(A);
    auto xp = convert_vec<p>(x);
    p res = matvec_sum<p>(Ap, xp);
    return static_cast<double>(res);
}

struct Format { int n; int es; };

// runtime dispatch: n ∈ {8,16,32}，es ∈ [0, n-2]
template<int N, int ES, typename Fn, typename... Args>
double dispatch_es(int target_es, Fn &&fn, Args&&... args) {
    if (target_es == ES) {
        return fn(std::integral_constant<int, N>{}, std::integral_constant<int, ES>{}, std::forward<Args>(args)...);
    }
    if constexpr (ES + 1 < N - 1) {
        return dispatch_es<N, ES + 1>(target_es, std::forward<Fn>(fn), std::forward<Args>(args)...);
    }
    throw std::runtime_error("unsupported es for posit");
}

template<typename Fn, typename... Args>
double dispatch_posit(int n, int es, Fn &&fn, Args&&... args) {
    switch (n) {
        case 8:  return dispatch_es<8, 0>(es, std::forward<Fn>(fn), std::forward<Args>(args)...);
        case 16: return dispatch_es<16,0>(es, std::forward<Fn>(fn), std::forward<Args>(args)...);
        case 32: return dispatch_es<32,0>(es, std::forward<Fn>(fn), std::forward<Args>(args)...);
        default: throw std::runtime_error("unsupported posit n");
    }
}

int main() {
    mt19937 rng(42);
    vector<int> vec_lens = {16, 128, 1024};
    // 固定行數，讓矩陣維度為 rows x vec_len
    int rows = 16;

    vector<DistType> dists = {
        DistType::Normal01,
        DistType::UniformMinus1To1,
        DistType::NormalSmall,
        DistType::NormalLarge
    };
    int num_samples_per_setting = 100;  // 比其他 kernel 少，但不再低到資料嚴重失衡

    vector<Format> formats;
    for (int n : {8, 16, 32}) {
        int max_es = min(2, n - 2);
        for (int es = 0; es <= max_es; ++es) {
            formats.push_back({n, es});
        }
    }

    filesystem::create_directories("data");
    string out_path = "data/matvec_dataset.csv";
    ofstream ofs(out_path);
    ofs << "sample_id,rows,cols,dist_type,"
        << "posit_n,posit_es,"
        << "abs_err,rel_err,"
        << "fp32_abs_err,fp32_rel_err,"
        << "x_mean,x_std,x_p1,x_p50,x_p99,"
        << "y_mean,y_std,y_p1,y_p50,y_p99\n";

    int gid = 0;
    for (int cols : vec_lens) {
        for (DistType dist : dists) {
            for (int s = 0; s < num_samples_per_setting; ++s) {
                auto A = gen_matrix(rows, cols, rng, dist);
                auto x = gen_vector(cols, rng, dist);
                // matvec: 把 A 當作 input0(x_*)，把向量 x 當作 input1(y_*)
                auto xs = compute_mat_stats_absq(A);
                auto ys = compute_vec_stats_absq(x);

                double ref64 = matvec_sum_double(A, x);

                auto Af = convert_mat<float>(A);
                auto xf = convert_vec<float>(x);
                float ref32f = matvec_sum<float>(Af, xf);
                double ref32 = static_cast<double>(ref32f);

                double fp32_abs_err = fabs(ref32 - ref64);
                double fp32_rel_err = fp32_abs_err / (fabs(ref64) + 1e-12);

                for (auto f : formats) {
                    double rp = 0.0;
                    auto runner = [&](auto n_c, auto es_c) -> double {
                        return matvec_sum_posit<n_c, es_c>(A, x);
                    };
                    try {
                        rp = dispatch_posit(f.n, f.es, runner);
                    } catch (const std::exception&) {
                        continue;
                    }

                    double abs_err = fabs(rp - ref64);
                    double rel_err = abs_err / (fabs(ref64) + 1e-12);

                    ofs << gid << ","
                        << rows << ","
                        << cols << ","
                        << static_cast<int>(dist) << ","
                        << f.n << ","
                        << f.es << ","
                        << abs_err << ","
                        << rel_err << ","
                        << fp32_abs_err << ","
                        << fp32_rel_err << ","
                        << xs.mean << "," << xs.std << "," << xs.p1 << "," << xs.p50 << "," << xs.p99 << ","
                        << ys.mean << "," << ys.std << "," << ys.p1 << "," << ys.p50 << "," << ys.p99 << "\n";
                }
                ++gid;
            }
        }
    }

    ofs.close();
    cout << "matvec dataset written to " << out_path << "\n";
    return 0;
}
