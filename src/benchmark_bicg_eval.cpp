#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <universal/number/posit/posit.hpp>

#include "stats.hpp"

using namespace std;
using namespace sw::universal;

struct Format {
    int n;
    int es;
};

bool operator<(const Format &a, const Format &b) {
    if (a.n != b.n) return a.n < b.n;
    return a.es < b.es;
}

vector<double> init_polybench_vec_pi(int n) {
    vector<double> v(n);
    for (int i = 0; i < n; ++i) v[i] = static_cast<double>(i) * M_PI;
    return v;
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
pair<vector<Real>, vector<Real>> bicg(
    const vector<vector<Real>> &A,
    const vector<Real> &r,
    const vector<Real> &p
) {
    const size_t rows = A.size();
    const size_t cols = rows ? A[0].size() : 0;
    vector<Real> s(cols, Real(0));
    vector<Real> q(rows, Real(0));

    for (size_t i = 0; i < rows; ++i) {
        q[i] = Real(0);
        for (size_t j = 0; j < cols; ++j) {
            s[j] += r[i] * A[i][j];
            q[i] += A[i][j] * p[j];
        }
    }
    return {s, q};
}

template<int N, int ES>
pair<vector<double>, vector<double>> bicg_posit(
    const vector<vector<double>> &A,
    const vector<double> &r,
    const vector<double> &p
) {
    using posit_t = posit<N, ES>;
    auto Ap = convert_mat<posit_t>(A);
    auto rp = convert_vec<posit_t>(r);
    auto pp = convert_vec<posit_t>(p);
    auto [s, q] = bicg<posit_t>(Ap, rp, pp);
    vector<double> sd(s.size()), qd(q.size());
    for (size_t i = 0; i < s.size(); ++i) sd[i] = static_cast<double>(s[i]);
    for (size_t i = 0; i < q.size(); ++i) qd[i] = static_cast<double>(q[i]);
    return {sd, qd};
}

template<int N, int ES, typename Fn>
auto dispatch_es(int target_es, Fn &&fn) {
    if (target_es == ES) return fn(std::integral_constant<int, N>{}, std::integral_constant<int, ES>{});
    if constexpr (ES + 1 < N - 1) return dispatch_es<N, ES + 1>(target_es, std::forward<Fn>(fn));
    throw runtime_error("unsupported es");
}

template<typename Fn>
auto dispatch_posit(int n, int es, Fn &&fn) {
    switch (n) {
        case 8: return dispatch_es<8, 0>(es, std::forward<Fn>(fn));
        case 16: return dispatch_es<16, 0>(es, std::forward<Fn>(fn));
        case 32: return dispatch_es<32, 0>(es, std::forward<Fn>(fn));
        default: throw runtime_error("unsupported posit n");
    }
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

VecStats combine_vec_stats(const vector<double> &a, const vector<double> &b) {
    vector<double> merged;
    merged.reserve(a.size() + b.size());
    merged.insert(merged.end(), a.begin(), a.end());
    merged.insert(merged.end(), b.begin(), b.end());
    return compute_vec_stats_absq(merged);
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
    const pair<vector<double>, vector<double>> &approx,
    const pair<vector<double>, vector<double>> &ref
) {
    const double sum_ref = vector_sum(ref.first) + vector_sum(ref.second);
    const double sum_approx = vector_sum(approx.first) + vector_sum(approx.second);
    const double sum_abs_err = fabs(sum_approx - sum_ref);
    const double sum_rel_err = sum_abs_err / (fabs(sum_ref) + 1e-12);

    const double l2_abs_err = sqrt(
        pow(l2_diff_norm(approx.first, ref.first), 2.0) +
        pow(l2_diff_norm(approx.second, ref.second), 2.0)
    );
    const double l2_ref = sqrt(
        pow(l2_norm(ref.first), 2.0) +
        pow(l2_norm(ref.second), 2.0)
    );
    const double l2_rel_err = l2_abs_err / (l2_ref + 1e-12);

    os << "\"" << name << "\":{"
       << "\"sum_abs_err\":" << sum_abs_err << ","
       << "\"sum_rel_err\":" << sum_rel_err << ","
       << "\"l2_abs_err\":" << l2_abs_err << ","
       << "\"l2_rel_err\":" << l2_rel_err
       << "}";
}

int main(int argc, char **argv) {
    int rows = 4000;
    int cols = 4000;
    bool polybench_init = false;
    string out_json = "data/bicg_actual.json";
    string formats_arg;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        auto need = [&](const char *name) -> string {
            if (i + 1 >= argc) throw runtime_error(string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--rows") rows = stoi(need("--rows"));
        else if (arg == "--cols") cols = stoi(need("--cols"));
        else if (arg == "--polybench-init") polybench_init = true;
        else if (arg == "--formats") formats_arg = need("--formats");
        else if (arg == "--out-json") out_json = need("--out-json");
        else throw runtime_error("unknown arg: " + arg);
    }

    if (!polybench_init) {
        throw runtime_error("benchmark_bicg_eval currently supports only --polybench-init");
    }

    auto A = init_polybench_A(rows, cols);
    auto r = init_polybench_vec_pi(rows);
    auto p = init_polybench_vec_pi(cols);

    auto x_stats = compute_mat_stats_absq(A);
    auto y_stats = combine_vec_stats(r, p);

    auto ref64 = bicg<double>(A, r, p);

    auto Af = convert_mat<float>(A);
    auto rf = convert_vec<float>(r);
    auto pf = convert_vec<float>(p);
    auto ref32 = bicg<float>(Af, rf, pf);
    pair<vector<double>, vector<double>> ref32d = {
        vector<double>(ref32.first.begin(), ref32.first.end()),
        vector<double>(ref32.second.begin(), ref32.second.end()),
    };

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
    ofs << "  \"benchmark\":\"bicg\",\n";
    ofs << "  \"rows\":" << rows << ",\n";
    ofs << "  \"cols\":" << cols << ",\n";
    ofs << "  \"polybench_init\":true,\n";
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
    vector<double> rp_merged;
    rp_merged.reserve(r.size() + p.size());
    rp_merged.insert(rp_merged.end(), r.begin(), r.end());
    rp_merged.insert(rp_merged.end(), p.begin(), p.end());
    for (const auto &f : formats) {
        auto quant_runner = [&A, &rp_merged](auto n_c, auto es_c) {
            using p_t = posit<n_c, es_c>;
            return pair<QuantFeatureStats, QuantFeatureStats>{
                compute_mat_quant_features<p_t>(A),
                compute_vec_quant_features<p_t>(rp_merged),
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
        auto runner = [&A, &r, &p](auto n_c, auto es_c) {
            return bicg_posit<n_c, es_c>(A, r, p);
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
