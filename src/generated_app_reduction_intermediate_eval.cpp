#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <universal/number/posit/posit.hpp>

using namespace std;
using namespace sw::universal;

namespace {

struct Format {
    int n;
    int es;
};

template <typename Tmp>
double run_dot_product(int vid) {
    const int n = 96 + 32 * (vid % 9);
    const double a = 0.15 + 0.03 * (vid % 11);
    const double b = 0.2 + 0.04 * ((vid + 3) % 7);
    vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = cos(a * static_cast<double>(i + 1));
        y[i] = sin(b * static_cast<double>(i + 3));
    }
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        const double prod = static_cast<double>(Tmp(x[i] * y[i]));
        acc += prod;
    }
    return acc;
}

template <typename Tmp>
double run_weighted_sum(int vid) {
    const int n = 128 + 8 * (vid % 17);
    const double alpha = 0.5 + 0.1 * (vid % 5);
    const double beta = -0.25 + 0.05 * ((vid + 1) % 7);
    vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = sin(0.07 * static_cast<double>(i + 1));
        y[i] = cos(0.11 * static_cast<double>(i + 2));
    }
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        const double term = static_cast<double>(Tmp(alpha * x[i] + beta * y[i]));
        acc += term;
    }
    return acc;
}

template <typename Tmp>
double run_prefix_recurrence(int vid) {
    const int n = 96 + 16 * (vid % 12);
    const double gamma = 0.85 + 0.01 * (vid % 8);
    vector<double> x(n);
    for (int i = 0; i < n; ++i) x[i] = sin(0.05 * static_cast<double>(i + 1));
    double state = 0.0;
    double out = 0.0;
    for (int i = 0; i < n; ++i) {
        const double term = static_cast<double>(Tmp(gamma * state + x[i]));
        state = term;
        out = state;
    }
    return out;
}

template <typename Tmp>
double run_ema_recurrence(int vid) {
    const int n = 100 + 20 * (vid % 10);
    const double alpha = 0.05 + 0.03 * (vid % 6);
    vector<double> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = cos(0.09 * static_cast<double>(i + 1)) + 0.01 * static_cast<double>(i % 5);
    }
    double ema = x[0];
    for (int i = 1; i < n; ++i) {
        const double term = static_cast<double>(Tmp(alpha * x[i] + (1.0 - alpha) * ema));
        ema = term;
    }
    return ema;
}

template <typename Tmp>
double run_two_stage_reduce(int vid) {
    const int n = 128 + 16 * (vid % 10);
    vector<double> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = sin(0.04 * static_cast<double>(i + 1)) + cos(0.02 * static_cast<double>(i + 1));
    }
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        const double tmp = static_cast<double>(Tmp(x[i] * x[i] + 0.125 * x[i]));
        acc += tmp;
    }
    return acc;
}

template <typename Tmp>
double run_stencil1d_like(int vid) {
    const int n = 96 + 16 * (vid % 11);
    const int steps = 4 + (vid % 4);
    vector<double> a(n), b(n, 0.0);
    for (int i = 0; i < n; ++i) a[i] = sin(0.06 * static_cast<double>(i + 1));
    for (int t = 0; t < steps; ++t) {
        for (int i = 1; i < n - 1; ++i) {
            const double tmp = static_cast<double>(Tmp(0.25 * a[i - 1] + 0.5 * a[i] + 0.25 * a[i + 1]));
            b[i] = tmp;
        }
        for (int i = 1; i < n - 1; ++i) a[i] = b[i];
    }
    double acc = 0.0;
    for (int i = 1; i < n - 1; ++i) acc += a[i];
    return acc;
}

template <typename Tmp>
double run_family_variant(const string &family, int vid) {
    if (family == "dot_product") return run_dot_product<Tmp>(vid);
    if (family == "weighted_sum") return run_weighted_sum<Tmp>(vid);
    if (family == "prefix_recurrence") return run_prefix_recurrence<Tmp>(vid);
    if (family == "ema_recurrence") return run_ema_recurrence<Tmp>(vid);
    if (family == "two_stage_reduce") return run_two_stage_reduce<Tmp>(vid);
    if (family == "stencil1d_like") return run_stencil1d_like<Tmp>(vid);
    throw runtime_error("unsupported family for reduction_intermediate eval: " + family);
}

template <int N, int ES, typename Fn>
auto dispatch_es(int target_es, Fn &&fn) {
    if (target_es == ES) return fn(std::integral_constant<int, N>{}, std::integral_constant<int, ES>{});
    if constexpr (ES + 1 < N - 1) return dispatch_es<N, ES + 1>(target_es, std::forward<Fn>(fn));
    throw runtime_error("unsupported es");
}

template <typename Fn>
auto dispatch_posit(int n, int es, Fn &&fn) {
    switch (n) {
        case 8: return dispatch_es<8, 0>(es, std::forward<Fn>(fn));
        case 16: return dispatch_es<16, 0>(es, std::forward<Fn>(fn));
        case 32: return dispatch_es<32, 0>(es, std::forward<Fn>(fn));
        default: throw runtime_error("unsupported posit n");
    }
}

vector<Format> parse_formats_csv(const string &text) {
    vector<Format> out;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find(',', start);
        if (end == string::npos) end = text.size();
        string token = text.substr(start, end - start);
        if (!token.empty()) {
            size_t sep = token.find(':');
            if (sep == string::npos) throw runtime_error("bad format token: " + token);
            out.push_back({stoi(token.substr(0, sep)), stoi(token.substr(sep + 1))});
        }
        start = end + 1;
    }
    return out;
}

}  // namespace

int main(int argc, char **argv) {
    string family;
    int variant = -1;
    string out_json = "entity_eval.json";
    string formats_arg = "8:0,8:1,8:2,16:0,16:1,16:2,32:0,32:1,32:2";

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        auto need = [&](const char *name) -> string {
            if (i + 1 >= argc) throw runtime_error(string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--family") family = need("--family");
        else if (arg == "--variant") variant = stoi(need("--variant"));
        else if (arg == "--formats") formats_arg = need("--formats");
        else if (arg == "--out-json") out_json = need("--out-json");
        else throw runtime_error("unknown arg: " + arg);
    }

    if (family.empty() || variant < 0) throw runtime_error("need --family and --variant");

    const double ref = run_family_variant<double>(family, variant);
    const auto formats = parse_formats_csv(formats_arg);

    filesystem::create_directories(filesystem::path(out_json).parent_path());
    ofstream os(out_json);
    os << "{\n";
    os << "  \"family\": \"" << family << "\",\n";
    os << "  \"variant\": " << variant << ",\n";
    os << "  \"reference_output\": " << ref << ",\n";
    os << "  \"results\": {\n";
    for (size_t i = 0; i < formats.size(); ++i) {
        const auto fmt = formats[i];
        const string name = "posit_" + to_string(fmt.n) + "_" + to_string(fmt.es);
        const double approx = dispatch_posit(fmt.n, fmt.es, [&](auto n_c, auto es_c) {
            using p = posit<n_c, es_c>;
            return run_family_variant<p>(family, variant);
        });
        const double rel = fabs(approx - ref) / (fabs(ref) + 1e-12);
        os << "    \"" << name << "\": {"
           << "\"approx_output\": " << approx << ", "
           << "\"actual_whole_app_rel_err\": " << rel
           << "}";
        if (i + 1 != formats.size()) os << ",";
        os << "\n";
    }
    os << "  }\n";
    os << "}\n";
    return 0;
}
