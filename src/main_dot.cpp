// main.cpp
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <limits>
#include <algorithm>

#include <universal/number/posit/posit.hpp>
using namespace sw::universal;
using namespace std;

#include "kernel.hpp"
#include "stats.hpp"


enum class DistType {
    Normal = 0,          // N(0, sigma)
    Uniform = 1,         // U(-a, a)
    Laplace = 2,         // Laplace(0, b)
    GaussianMixture = 3, // (1-p)N(0,s1) + pN(0,s2)
};

static double sample_log_uniform(mt19937 &rng, double lo, double hi) {
    if (!(lo > 0.0) || !(hi > 0.0) || !(hi >= lo)) {
        throw std::runtime_error("invalid log-uniform range");
    }
    uniform_real_distribution<double> u(log10(lo), log10(hi));
    return pow(10.0, u(rng));
}

static double sample_scale(mt19937 &rng, bool random_scale) {
    return random_scale ? sample_log_uniform(rng, 1e-4, 1e3) : 1.0;
}

// 依照指定分布產生 double 向量
vector<double> gen_vector(int n, mt19937 &rng, DistType dist_type, double scale, double param1, double param2) {
    vector<double> v(n);

    if (dist_type == DistType::Normal) {
        normal_distribution<double> dist(0.0, scale);
        for (int i = 0; i < n; ++i) v[i] = dist(rng);
    } else if (dist_type == DistType::Uniform) {
        uniform_real_distribution<double> dist(-scale, scale);
        for (int i = 0; i < n; ++i) v[i] = dist(rng);
    } else if (dist_type == DistType::Laplace) {
        double b = (scale > 0.0) ? scale : 1.0;
        exponential_distribution<double> exp_dist(1.0 / b);
        bernoulli_distribution coin(0.5);
        for (int i = 0; i < n; ++i) {
            double mag = exp_dist(rng);
            v[i] = coin(rng) ? mag : -mag;
        }
    } else if (dist_type == DistType::GaussianMixture) {
        double p = param1;
        if (p < 0.0) p = 0.0;
        if (p > 0.5) p = 0.5;
        double ratio = (param2 > 1.0) ? param2 : 10.0;
        bernoulli_distribution outlier_coin(p);
        normal_distribution<double> n1(0.0, scale);
        normal_distribution<double> n2(0.0, scale * ratio);
        for (int i = 0; i < n; ++i) {
            v[i] = outlier_coin(rng) ? n2(rng) : n1(rng);
        }
    }

    return v;
}

// fp64 baseline 版 dot，當作「近似真值」
double dot_double(const vector<double> &a, const vector<double> &b) {
    return dot<double>(a, b);
}

// 把 double 向量轉成某種 Real（float 或 posit 都可以）
template<typename Real>
vector<Real> convert_vec(const vector<double> &src) {
    vector<Real> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = static_cast<Real>(src[i]);
    }
    return dst;
}

// 用某一種 posit 格式做 dot，只回傳結果（轉回 double）
template<int N, int ES>
double dot_posit(const vector<double> &a,
                 const vector<double> &b) {
    using p = posit<N, ES>;
    auto a_p = convert_vec<p>(a);
    auto b_p = convert_vec<p>(b);
    p sum = dot<p>(a_p, b_p);
    return static_cast<double>(sum);
}

struct Format {
    int n;   // posit 總 bit 數
    int es;  // posit exponent size
};

// runtime dispatch: n ∈ {8,16,32}，es ∈ [0, n-2]
template<int N, int ES, typename Fn, typename... Args>
auto dispatch_es(int target_es, Fn &&fn, Args&&... args) {
    if (target_es == ES) {
        return fn(std::integral_constant<int, N>{}, std::integral_constant<int, ES>{}, std::forward<Args>(args)...);
    }
    if constexpr (ES + 1 < N - 1) {
        return dispatch_es<N, ES + 1>(target_es, std::forward<Fn>(fn), std::forward<Args>(args)...);
    }
    throw std::runtime_error("unsupported es for posit");
}

template<typename Fn, typename... Args>
auto dispatch_posit(int n, int es, Fn &&fn, Args&&... args) {
    switch (n) {
        case 8:  return dispatch_es<8, 0>(es, std::forward<Fn>(fn), std::forward<Args>(args)...);
        case 16: return dispatch_es<16,0>(es, std::forward<Fn>(fn), std::forward<Args>(args)...);
        case 32: return dispatch_es<32,0>(es, std::forward<Fn>(fn), std::forward<Args>(args)...);
        default: throw std::runtime_error("unsupported posit n");
    }
}

int main(int argc, char **argv) {
    int seed = 42;
    int num_samples_per_setting = 2000;
    bool random_scale = true;
    bool independent_scale = true;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--fixed-scale") {
            random_scale = false;
        } else if (arg == "--random-scale") {
            random_scale = true;
        } else if (arg == "--same-scale") {
            independent_scale = false;
        } else if (arg == "--independent-scale") {
            independent_scale = true;
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = stoi(argv[++i]);
        } else if (arg == "--samples" && i + 1 < argc) {
            num_samples_per_setting = stoi(argv[++i]);
        } else {
            cerr << "Usage: " << argv[0]
                 << " [--random-scale|--fixed-scale] [--independent-scale|--same-scale] [--seed N] [--samples N]\n";
            return 2;
        }
    }

    mt19937 rng(seed);

    vector<int> vec_lens = {16, 32, 64, 128, 256, 512, 768, 1024};

    // 每筆 sample 會隨機挑一種分佈家族（dist_type 只是記錄用，不會被回歸模型直接當作 feature）
    vector<DistType> dists = {
        DistType::Normal,
        DistType::Uniform,
        DistType::Laplace,
        DistType::GaussianMixture,
    };

    // 可測的 posit 格式列表（n=8/16/32，es=0..2 或合法範圍內）
    vector<Format> formats;
    for (int n : {8, 16, 32}) {
        int max_es = min(2, n - 2);
        for (int es = 0; es <= max_es; ++es) {
            formats.push_back({n, es});
        }
    }

    string out_path = "data/dot_dataset.csv";
    filesystem::create_directories("data");
    ofstream ofs(out_path);
    // dist_type 用 int 輸出，對應 enum DistType 的值
    ofs << "sample_id,vec_len,dist_type,"
        << "dist_param1,dist_param2,"
        << "x_scale,y_scale,"
        << "posit_n,posit_es,"
        << "abs_err,rel_err,"
        << "fp32_abs_err,fp32_rel_err,";
    write_stats_header(ofs, "x");
    ofs << ",";
    write_stats_header(ofs, "y");
    ofs << ",";
    write_quant_feature_header(ofs, "x");
    ofs << ",";
    write_quant_feature_header(ofs, "y");
    ofs << "\n";

    int global_sample_id = 0;

    for (int len : vec_lens) {
        vector<DistType> pool = dists;
        shuffle(pool.begin(), pool.end(), rng);

        for (int s = 0; s < num_samples_per_setting; ++s) {
            if (s % static_cast<int>(pool.size()) == 0) {
                shuffle(pool.begin(), pool.end(), rng);
            }
            DistType dist = pool[s % static_cast<int>(pool.size())];

            // 分佈參數：不同 family 可能需要 outlier_prob / ratio
            double dist_param1 = 0.0;
            double dist_param2 = 0.0;
            if (dist == DistType::GaussianMixture) {
                uniform_real_distribution<double> pd(0.01, 0.10);
                dist_param1 = pd(rng); // outlier prob
                dist_param2 = sample_log_uniform(rng, 3.0, 30.0); // sigma ratio
            }

            // 1) 產生一組 (x, y) 輸入（同 family，可不同尺度）
            double scale_x = sample_scale(rng, random_scale);
            double scale_y = independent_scale ? sample_scale(rng, random_scale) : scale_x;
            auto x = gen_vector(len, rng, dist, scale_x, dist_param1, dist_param2);
            auto y = gen_vector(len, rng, dist, scale_y, dist_param1, dist_param2);
            auto xs = compute_vec_stats_absq(x);
            auto ys = compute_vec_stats_absq(y);

            // 2) fp64 baseline：optimal 近似真值
            double ref64 = dot_double(x, y);

            // 3) fp32 baseline
            auto x_f = convert_vec<float>(x);
            auto y_f = convert_vec<float>(y);
            float ref32_f = dot<float>(x_f, y_f);
            double ref32 = static_cast<double>(ref32_f);

            double fp32_abs_err = fabs(ref32 - ref64);
            double fp32_rel_err = fp32_abs_err / (fabs(ref64) + 1e-12);

            // 4) 對每種 posit 格式計算誤差
            for (auto f : formats) {
                double rp = 0.0;

                auto runner = [&](auto n_c, auto es_c) -> double {
                    return dot_posit<n_c, es_c>(x, y);
                };
                auto quant_runner = [&](auto n_c, auto es_c) {
                    using p = posit<n_c, es_c>;
                    return std::pair<QuantFeatureStats, QuantFeatureStats>{
                        compute_vec_quant_features<p>(x),
                        compute_vec_quant_features<p>(y),
                    };
                };

                try {
                    rp = dispatch_posit(f.n, f.es, runner);
                } catch (const std::exception&) {
                    continue;
                }
                auto [xq, yq] = dispatch_posit(f.n, f.es, quant_runner);

                double abs_err = fabs(rp - ref64);
                double rel_err = abs_err / (fabs(ref64) + 1e-12);

                ofs << global_sample_id << ","
                    << len << ","
                    << static_cast<int>(dist) << ","
                    << dist_param1 << ","
                    << dist_param2 << ","
                    << scale_x << ","
                    << scale_y << ","
                    << f.n << ","
                    << f.es << ","
                    << abs_err << ","
                    << rel_err << ","
                    << fp32_abs_err << ","
                    << fp32_rel_err << ",";
                write_stats_values(ofs, xs);
                ofs << ",";
                write_stats_values(ofs, ys);
                ofs << ",";
                write_quant_feature_values(ofs, xq);
                ofs << ",";
                write_quant_feature_values(ofs, yq);
                ofs << "\n";
            }

            ++global_sample_id;
        }
    }

    ofs.close();
    cout << "dataset written to " << out_path << "\n";
    return 0;
}
