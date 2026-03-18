// main_l1_norm.cpp
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

// 輸入分布
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

template<typename Real>
vector<Real> convert_vec(const vector<double> &src) {
    vector<Real> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = static_cast<Real>(src[i]);
    }
    return dst;
}

// baseline
double l1_norm_double(const vector<double> &x) {
    return l1_norm<double>(x);
}

template<int N, int ES>
double l1_norm_posit(const vector<double> &x) {
    using p = posit<N, ES>;
    auto xp = convert_vec<p>(x);
    p res = l1_norm<p>(xp);
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
    vector<DistType> dists = {
        DistType::Normal01,
        DistType::UniformMinus1To1,
        DistType::NormalSmall,
        DistType::NormalLarge
    };
    int num_samples_per_setting = 500;

    vector<Format> formats;
    for (int n : {8, 16, 32}) {
        int max_es = min(2, n - 2);
        for (int es = 0; es <= max_es; ++es) {
            formats.push_back({n, es});
        }
    }

    filesystem::create_directories("data");
    string out_path = "data/l1_norm_dataset.csv";
    ofstream ofs(out_path);
    ofs << "sample_id,vec_len,dist_type,"
        << "posit_n,posit_es,"
        << "abs_err,rel_err,"
        << "fp32_abs_err,fp32_rel_err,"
        << "x_mean,x_std,x_p1,x_p50,x_p99,"
        << "y_mean,y_std,y_p1,y_p50,y_p99\n";

    int gid = 0;
    for (int len : vec_lens) {
        for (DistType dist : dists) {
            for (int s = 0; s < num_samples_per_setting; ++s) {
                auto x = gen_vector(len, rng, dist);
                auto xs = compute_vec_stats_absq(x);
                VecStats ys{0.0, 0.0, 0.0, 0.0, 0.0};

                double ref64 = l1_norm_double(x);

                auto xf = convert_vec<float>(x);
                float ref32f = l1_norm<float>(xf);
                double ref32 = static_cast<double>(ref32f);

                double fp32_abs_err = fabs(ref32 - ref64);
                double fp32_rel_err = fp32_abs_err / (fabs(ref64) + 1e-12);

                for (auto f : formats) {
                    double rp = 0.0;
                    auto runner = [&](auto n_c, auto es_c) -> double {
                        return l1_norm_posit<n_c, es_c>(x);
                    };
                    try {
                        rp = dispatch_posit(f.n, f.es, runner);
                    } catch (const std::exception&) {
                        continue;
                    }

                    double abs_err = fabs(rp - ref64);
                    double rel_err = abs_err / (fabs(ref64) + 1e-12);

                    ofs << gid << ","
                        << len << ","
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
    cout << "l1_norm dataset written to " << out_path << "\n";
    return 0;
}
