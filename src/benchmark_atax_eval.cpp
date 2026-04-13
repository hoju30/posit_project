#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <universal/number/posit/posit.hpp>

#include "stats.hpp"

using namespace std;
using namespace sw::universal;

enum class DistType {
    Normal01 = 0,
    UniformMinus1To1 = 1,
    NormalSmall = 2,
    NormalLarge = 3
};

struct Format {
    int n;
    int es;
};

bool operator<(const Format &a, const Format &b) {
    if (a.n != b.n) return a.n < b.n;
    return a.es < b.es;
}

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

vector<double> init_polybench_x(int cols) {
    vector<double> x(cols);
    for (int i = 0; i < cols; ++i) x[i] = static_cast<double>(i) * M_PI;
    return x;
}

vector<vector<double>> init_polybench_A(int rows, int cols) {
    vector<vector<double>> A(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            A[i][j] = (static_cast<double>(i) * static_cast<double>(j + 1)) / static_cast<double>(rows);
        }
    }
    return A;
}

template<typename Real>
vector<Real> convert_vec(const vector<double> &src) {
    vector<Real> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) dst[i] = static_cast<Real>(src[i]);
    return dst;
}

template<typename Real>
vector<vector<Real>> convert_mat(const vector<vector<double>> &src) {
    vector<vector<Real>> dst(src.size());
    for (size_t r = 0; r < src.size(); ++r) dst[r] = convert_vec<Real>(src[r]);
    return dst;
}

template<typename Real>
vector<Real> atax(const vector<vector<Real>> &A, const vector<Real> &x) {
    const size_t rows = A.size();
    const size_t cols = rows ? A[0].size() : 0;
    vector<Real> tmp(rows, Real(0));
    vector<Real> y(cols, Real(0));

    for (size_t i = 0; i < rows; ++i) {
        Real acc = Real(0);
        for (size_t j = 0; j < cols; ++j) {
            acc += A[i][j] * x[j];
        }
        tmp[i] = acc;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            y[j] += A[i][j] * tmp[i];
        }
    }

    return y;
}

double vector_sum(const vector<double> &v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s;
}

double l2_norm(const vector<double> &v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return sqrt(s);
}

double l2_diff_norm(const vector<double> &a, const vector<double> &b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return sqrt(s);
}

template<int N, int ES>
vector<double> atax_posit(const vector<vector<double>> &A, const vector<double> &x) {
    using p = posit<N, ES>;
    auto Ap = convert_mat<p>(A);
    auto xp = convert_vec<p>(x);
    auto yp = atax<p>(Ap, xp);
    vector<double> out(yp.size());
    for (size_t i = 0; i < yp.size(); ++i) out[i] = static_cast<double>(yp[i]);
    return out;
}

template<int N, int ES, typename Fn>
auto dispatch_es(int target_es, Fn &&fn) {
    if (target_es == ES) return fn(std::integral_constant<int, N>{}, std::integral_constant<int, ES>{});
    if constexpr (ES + 1 < N - 1) return dispatch_es<N, ES + 1>(target_es, std::forward<Fn>(fn));
    throw std::runtime_error("unsupported es");
}

template<typename Fn>
auto dispatch_posit(int n, int es, Fn &&fn) {
    switch (n) {
        case 8: return dispatch_es<8, 0>(es, std::forward<Fn>(fn));
        case 16: return dispatch_es<16, 0>(es, std::forward<Fn>(fn));
        case 32: return dispatch_es<32, 0>(es, std::forward<Fn>(fn));
        default: throw std::runtime_error("unsupported posit n");
    }
}

void write_stats_json(ostream &os, const VecStats &s) {
    os << "{"
       << "\"mean\":" << s.mean << ","
       << "\"std\":" << s.std << ","
       << "\"min\":" << s.min << ","
       << "\"max\":" << s.max << ","
       << "\"abs_max\":" << s.abs_max << ","
       << "\"skewness\":" << s.skewness << ","
       << "\"excess_kurtosis\":" << s.excess_kurtosis << ","
       << "\"p01\":" << s.p01 << ","
       << "\"p50\":" << s.p50 << ","
       << "\"p99\":" << s.p99 << ","
       << "\"near_zero_ratio\":" << s.near_zero_ratio << ","
       << "\"pos_ratio\":" << s.pos_ratio << ","
       << "\"neg_ratio\":" << s.neg_ratio
       << "}";
}

void write_quant_feature_json(ostream &os, const QuantFeatureStats &s, const char *prefix) {
    os << "\""
       << prefix << "_oor_ratio\":" << s.oor_ratio << ","
       << "\"" << prefix << "_clip_high_ratio\":" << s.clip_high_ratio << ","
       << "\"" << prefix << "_clip_low_ratio\":" << s.clip_low_ratio << ","
       << "\"" << prefix << "_rel_qerr_mean\":" << s.rel_qerr_mean << ","
       << "\"" << prefix << "_zero_after_quant_ratio\":" << s.zero_after_quant_ratio << ","
       << "\"" << prefix << "_unique_ratio\":" << s.unique_ratio;
}

void write_error_json(
    ostream &os,
    const string &name,
    const vector<double> &approx,
    const vector<double> &ref
) {
    const double sum_ref = vector_sum(ref);
    const double sum_approx = vector_sum(approx);
    const double sum_abs_err = fabs(sum_approx - sum_ref);
    const double sum_rel_err = sum_abs_err / (fabs(sum_ref) + 1e-12);

    const double l2_abs_err = l2_diff_norm(approx, ref);
    const double l2_rel_err = l2_abs_err / (l2_norm(ref) + 1e-12);

    os << "\"" << name << "\":{"
       << "\"sum_abs_err\":" << sum_abs_err << ","
       << "\"sum_rel_err\":" << sum_rel_err << ","
       << "\"l2_abs_err\":" << l2_abs_err << ","
       << "\"l2_rel_err\":" << l2_rel_err
       << "}";
}

int main(int argc, char **argv) {
    int rows = 2;
    int cols = 2;
    int dist_type = 0;
    int seed = 42;
    bool polybench_init = false;
    string out_json = "data/atax_actual.json";
    string formats_arg;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        auto need = [&](const char *name) -> string {
            if (i + 1 >= argc) throw runtime_error(string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--rows") rows = stoi(need("--rows"));
        else if (arg == "--cols") cols = stoi(need("--cols"));
        else if (arg == "--dist-type") dist_type = stoi(need("--dist-type"));
        else if (arg == "--seed") seed = stoi(need("--seed"));
        else if (arg == "--polybench-init") polybench_init = true;
        else if (arg == "--formats") formats_arg = need("--formats");
        else if (arg == "--out-json") out_json = need("--out-json");
        else throw runtime_error("unknown arg: " + arg);
    }

    vector<vector<double>> A;
    vector<double> x;
    if (polybench_init) {
        A = init_polybench_A(rows, cols);
        x = init_polybench_x(cols);
    } else {
        mt19937 rng(seed);
        auto dist = static_cast<DistType>(dist_type);
        A = gen_matrix(rows, cols, rng, dist);
        x = gen_vector(cols, rng, dist);
    }

    auto x_stats = compute_mat_stats_absq(A);
    auto y_stats = compute_vec_stats_absq(x);

    auto ref64 = atax<double>(A, x);

    auto Af = convert_mat<float>(A);
    auto xf = convert_vec<float>(x);
    auto ref32 = atax<float>(Af, xf);
    vector<double> ref32d(ref32.begin(), ref32.end());

    vector<Format> formats;
    for (int n : {8, 16, 32}) {
        for (int es = 0; es <= min(2, n - 2); ++es) formats.push_back({n, es});
    }
    if (!formats_arg.empty()) {
        set<Format> allowed;
        size_t start = 0;
        while (start < formats_arg.size()) {
            size_t end = formats_arg.find(',', start);
            if (end == string::npos) end = formats_arg.size();
            string tok = formats_arg.substr(start, end - start);
            auto pos = tok.find(':');
            if (pos == string::npos) throw runtime_error("bad --formats token: " + tok);
            allowed.insert({stoi(tok.substr(0, pos)), stoi(tok.substr(pos + 1))});
            start = end + 1;
        }
        vector<Format> filtered;
        for (const auto &f : formats) {
            if (allowed.count(f)) filtered.push_back(f);
        }
        formats = filtered;
    }

    filesystem::create_directories(filesystem::path(out_json).parent_path());
    ofstream ofs(out_json);
    ofs << "{\n";
    ofs << "  \"benchmark\":\"atax\",\n";
    ofs << "  \"rows\":" << rows << ",\n";
    ofs << "  \"cols\":" << cols << ",\n";
    ofs << "  \"seed\":" << seed << ",\n";
    ofs << "  \"dist_type\":" << dist_type << ",\n";
    ofs << "  \"polybench_init\":" << (polybench_init ? "true" : "false") << ",\n";
    ofs << "  \"formats\":\"" << formats_arg << "\",\n";
    ofs << "  \"x_stats\":";
    write_stats_json(ofs, x_stats);
    ofs << ",\n";
    ofs << "  \"y_stats\":";
    write_stats_json(ofs, y_stats);
    ofs << ",\n";
    ofs << "  \"format_features\":{\n";
    ofs << "    \"formats\":{\n";
    bool first_fmt = true;
    for (const auto &f : formats) {
        auto quant_runner = [&A, &x](auto n_c, auto es_c) {
            using p = posit<n_c, es_c>;
            return pair<QuantFeatureStats, QuantFeatureStats>{
                compute_mat_quant_features<p>(A),
                compute_vec_quant_features<p>(x),
            };
        };
        auto [xq, yq] = dispatch_posit(f.n, f.es, quant_runner);
        if (!first_fmt) ofs << ",\n";
        first_fmt = false;
        ofs << "      \"posit_" << f.n << "_" << f.es << "\":{";
        write_quant_feature_json(ofs, xq, "x");
        ofs << ",";
        write_quant_feature_json(ofs, yq, "y");
        ofs << "}";
    }
    ofs << "\n    }\n";
    ofs << "  },\n";
    ofs << "  \"actual_errors\":{\n";
    ofs << "    ";
    write_error_json(ofs, "fp32", ref32d, ref64);

    for (const auto &f : formats) {
        auto runner = [&A, &x](auto n_c, auto es_c) {
            return atax_posit<n_c, es_c>(A, x);
        };
        auto out = dispatch_posit(f.n, f.es, runner);
        ofs << ",\n    ";
        write_error_json(ofs, "posit_" + to_string(f.n) + "_" + to_string(f.es), out, ref64);
    }
    ofs << "\n  }\n";
    ofs << "}\n";
    ofs.close();

    cout << "wrote " << out_json << "\n";
    return 0;
}
