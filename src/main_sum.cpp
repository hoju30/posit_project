// main_sum.cpp
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

#include "kernel.hpp"  // 需要 template<typename Real> Real sum(const vector<Real>&);
#include "stats.hpp"

// ===== 輸入分布型態 =====
// 存到 CSV 時 dist_type 用 int 表示：
// 0 = Normal01, 1 = UniformMinus1To1, 2 = NormalSmall, 3 = NormalLarge
enum class DistType {
    Normal01 = 0,         // N(0, 1)
    UniformMinus1To1 = 1, // U(-1, 1)
    NormalSmall = 2,      // N(0, 1e-3)
    NormalLarge = 3       // N(0, 1e3)
};

// 依照指定分布產生 double 向量
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

// 把 double 向量轉成某種 Real（float 或 posit 都可以）
template<typename Real>
vector<Real> convert_vec(const vector<double> &src) {
    vector<Real> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = static_cast<Real>(src[i]);
    }
    return dst;
}

// fp64 baseline：sum<double>(x)，當作「近似真值」
double sum_double(const vector<double> &x) {
    return sum<double>(x);
}

// 專給 posit 用的 sum wrapper：先把 double 向量轉成 posit，再跑 sum<Real>
template<int N, int ES>
double sum_posit(const vector<double> &x) {
    using p = posit<N, ES>;
    auto x_p = convert_vec<p>(x);
    p res = sum<p>(x_p);
    return static_cast<double>(res);
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

int main() {
    mt19937 rng(42);

    // 想覆蓋的向量長度
    vector<int> vec_lens = {16, 128, 1024};

    // 想覆蓋的輸入分布型態
    vector<DistType> dists = {
        DistType::Normal01,
        DistType::UniformMinus1To1,
        DistType::NormalSmall,
        DistType::NormalLarge
    };

    // 每個 (vec_len, dist_type) 組合下要產生幾個樣本
    int num_samples_per_setting = 500;  // 可依照機器負載調整

    // 要測的 posit 格式列表（n=8/16/32，es=0..2 或合法範圍內）
    vector<Format> formats;
    for (int n : {8, 16, 32}) {
        int max_es = min(2, n - 2);
        for (int es = 0; es <= max_es; ++es) {
            formats.push_back({n, es});
        }
    }

    string out_path = "data/sum_dataset.csv";
    filesystem::create_directories("data");
    ofstream ofs(out_path);
    // dist_type 存成 int，對應 enum DistType 的值
    ofs << "sample_id,vec_len,dist_type,"
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
        for (DistType dist : dists) {
            for (int s = 0; s < num_samples_per_setting; ++s) {
                // 1) 產生一組 x
                auto x = gen_vector(len, rng, dist);
                auto xs = compute_vec_stats_absq(x);
                VecStats ys{};

                // 2) fp64 baseline：optimal 近似真值
                double ref64 = sum_double(x);

                // 3) fp32 baseline
                auto x_f = convert_vec<float>(x);
                float ref32_f = sum<float>(x_f);
                double ref32 = static_cast<double>(ref32_f);

                double fp32_abs_err = fabs(ref32 - ref64);
                double fp32_rel_err = fp32_abs_err / (fabs(ref64) + 1e-12);

                // 4) 對每種 posit 格式計算誤差
                for (auto f : formats) {
                    double rp = 0.0;

                    auto runner = [&](auto n_c, auto es_c) -> double {
                        return sum_posit<n_c, es_c>(x);
                    };
                    auto quant_runner = [&](auto n_c, auto es_c) {
                        using p = posit<n_c, es_c>;
                        return std::pair<QuantFeatureStats, QuantFeatureStats>{
                            compute_vec_quant_features<p>(x),
                            QuantFeatureStats{},
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
    }

    ofs.close();
    cout << "sum dataset written to " << out_path << "\n";
    return 0;
}
