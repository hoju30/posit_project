#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>
#include <tuple>
#include <unordered_set>
#include <vector>

struct VecStats {
    double mean;
    double std;
    double min;
    double max;
    double abs_max;
    double skewness;
    double excess_kurtosis;
    double p01;
    double p50;
    double p99;
    double near_zero_ratio;
    double pos_ratio;
    double neg_ratio;
};

struct QuantFeatureStats {
    double oor_ratio;
    double clip_high_ratio;
    double clip_low_ratio;
    double rel_qerr_mean;
    double zero_after_quant_ratio;
    double unique_ratio;
};

inline std::tuple<double, double, double> percentiles_nearest(const std::vector<double> &v) {
    if (v.empty()) return {0.0, 0.0, 0.0};
    std::vector<double> tmp(v.begin(), v.end());
    size_t n = tmp.size();
    size_t k1 = static_cast<size_t>(std::floor(0.01 * static_cast<double>(n - 1)));
    size_t k50 = static_cast<size_t>(std::floor(0.50 * static_cast<double>(n - 1)));
    size_t k99 = static_cast<size_t>(std::floor(0.99 * static_cast<double>(n - 1)));
    if (k1 >= n) k1 = n - 1;
    if (k50 >= n) k50 = n - 1;
    if (k99 >= n) k99 = n - 1;

    std::nth_element(tmp.begin(), tmp.begin() + k1, tmp.end());
    double p01 = tmp[k1];
    std::nth_element(tmp.begin(), tmp.begin() + k50, tmp.end());
    double p50 = tmp[k50];
    std::nth_element(tmp.begin(), tmp.begin() + k99, tmp.end());
    double p99 = tmp[k99];
    return {p01, p50, p99};
}

inline void write_stats_header(std::ostream &os, const char *prefix) {
    os << prefix << "_mean,"
       << prefix << "_std,"
       << prefix << "_min,"
       << prefix << "_max,"
       << prefix << "_abs_max,"
       << prefix << "_skewness,"
       << prefix << "_excess_kurtosis,"
       << prefix << "_p01,"
       << prefix << "_p50,"
       << prefix << "_p99,"
       << prefix << "_near_zero_ratio,"
       << prefix << "_pos_ratio,"
       << prefix << "_neg_ratio";
}

inline void write_stats_values(std::ostream &os, const VecStats &s) {
    os << s.mean << ","
       << s.std << ","
       << s.min << ","
       << s.max << ","
       << s.abs_max << ","
       << s.skewness << ","
       << s.excess_kurtosis << ","
       << s.p01 << ","
       << s.p50 << ","
       << s.p99 << ","
       << s.near_zero_ratio << ","
       << s.pos_ratio << ","
       << s.neg_ratio;
}

inline void write_quant_feature_header(std::ostream &os, const char *prefix) {
    os << prefix << "_oor_ratio,"
       << prefix << "_clip_high_ratio,"
       << prefix << "_clip_low_ratio,"
       << prefix << "_rel_qerr_mean,"
       << prefix << "_zero_after_quant_ratio,"
       << prefix << "_unique_ratio";
}

inline void write_quant_feature_values(std::ostream &os, const QuantFeatureStats &s) {
    os << s.oor_ratio << ","
       << s.clip_high_ratio << ","
       << s.clip_low_ratio << ","
       << s.rel_qerr_mean << ","
       << s.zero_after_quant_ratio << ","
       << s.unique_ratio;
}

inline std::uint64_t double_bit_pattern(double v) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(double));
    return bits;
}

inline VecStats compute_vec_stats_absq(const std::vector<double> &v) {
    if (v.empty()) return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double mean = 0.0;
    double sq = 0.0;
    double minv = std::numeric_limits<double>::infinity();
    double maxv = -std::numeric_limits<double>::infinity();
    double abs_max = 0.0;
    double pos = 0.0;
    double neg = 0.0;
    for (double val : v) {
        mean += val;
        sq += val * val;
        minv = std::min(minv, val);
        maxv = std::max(maxv, val);
        abs_max = std::max(abs_max, std::fabs(val));
        if (val > 0.0) pos += 1.0;
        else if (val < 0.0) neg += 1.0;
    }
    mean /= static_cast<double>(v.size());
    double var = sq / static_cast<double>(v.size()) - mean * mean;
    double stdv = (var > 0.0) ? std::sqrt(var) : 0.0;
    auto [p01, p50, p99] = percentiles_nearest(v);

    double m3 = 0.0;
    double m4 = 0.0;
    double near_zero = 0.0;
    const double near_zero_thresh = std::max(1e-12, 0.01 * stdv);
    for (double val : v) {
        double d = val - mean;
        m3 += d * d * d;
        m4 += d * d * d * d;
        if (std::fabs(val) <= near_zero_thresh) near_zero += 1.0;
    }
    m3 /= static_cast<double>(v.size());
    m4 /= static_cast<double>(v.size());
    double skewness = 0.0;
    double excess_kurtosis = 0.0;
    if (stdv > 0.0) {
        skewness = m3 / (stdv * stdv * stdv);
        excess_kurtosis = m4 / (stdv * stdv * stdv * stdv) - 3.0;
    }

    double inv_n = 1.0 / static_cast<double>(v.size());
    return {
        mean,
        stdv,
        minv,
        maxv,
        abs_max,
        skewness,
        excess_kurtosis,
        p01,
        p50,
        p99,
        near_zero * inv_n,
        pos * inv_n,
        neg * inv_n,
    };
}

inline VecStats compute_mat_stats_absq(const std::vector<std::vector<double>> &A) {
    size_t rows = A.size();
    if (rows == 0) return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    size_t cols = A[0].size();
    if (cols == 0) return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    double mean = 0.0;
    double sq = 0.0;
    double minv = std::numeric_limits<double>::infinity();
    double maxv = -std::numeric_limits<double>::infinity();
    double abs_max = 0.0;
    double pos = 0.0;
    double neg = 0.0;
    std::vector<double> vals;
    vals.reserve(rows * cols);
    size_t count = 0;
    for (const auto &row : A) {
        for (double val : row) {
            mean += val;
            sq += val * val;
            minv = std::min(minv, val);
            maxv = std::max(maxv, val);
            abs_max = std::max(abs_max, std::fabs(val));
            if (val > 0.0) pos += 1.0;
            else if (val < 0.0) neg += 1.0;
            vals.push_back(val);
            ++count;
        }
    }
    if (count == 0) return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    mean /= static_cast<double>(count);
    double var = sq / static_cast<double>(count) - mean * mean;
    double stdv = (var > 0.0) ? std::sqrt(var) : 0.0;
    auto [p01, p50, p99] = percentiles_nearest(vals);

    double m3 = 0.0;
    double m4 = 0.0;
    double near_zero = 0.0;
    const double near_zero_thresh = std::max(1e-12, 0.01 * stdv);
    for (double val : vals) {
        double d = val - mean;
        m3 += d * d * d;
        m4 += d * d * d * d;
        if (std::fabs(val) <= near_zero_thresh) near_zero += 1.0;
    }
    m3 /= static_cast<double>(count);
    m4 /= static_cast<double>(count);
    double skewness = 0.0;
    double excess_kurtosis = 0.0;
    if (stdv > 0.0) {
        skewness = m3 / (stdv * stdv * stdv);
        excess_kurtosis = m4 / (stdv * stdv * stdv * stdv) - 3.0;
    }

    double inv_n = 1.0 / static_cast<double>(count);
    return {
        mean,
        stdv,
        minv,
        maxv,
        abs_max,
        skewness,
        excess_kurtosis,
        p01,
        p50,
        p99,
        near_zero * inv_n,
        pos * inv_n,
        neg * inv_n,
    };
}

template<typename Posit>
inline QuantFeatureStats compute_vec_quant_features(const std::vector<double> &v) {
    if (v.empty()) return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    const double maxpos = static_cast<double>(std::numeric_limits<Posit>::max());
    const double minpos = static_cast<double>(std::numeric_limits<Posit>::min());
    const double eps = 1e-12;

    double oor = 0.0;
    double clip_high = 0.0;
    double clip_low = 0.0;
    double rel_qerr_sum = 0.0;
    double zero_after_quant = 0.0;
    std::unordered_set<std::uint64_t> uniq;

    for (double val : v) {
        const double av = std::fabs(val);
        const bool is_high = av > maxpos;
        const bool is_low = (av > 0.0) && (av < minpos);
        if (is_high || is_low) oor += 1.0;
        if (is_high) clip_high += 1.0;
        if (is_low) clip_low += 1.0;

        Posit q = static_cast<Posit>(val);
        double qd = static_cast<double>(q);
        rel_qerr_sum += std::fabs(qd - val) / (std::fabs(val) + eps);
        if (qd == 0.0 && val != 0.0) zero_after_quant += 1.0;
        uniq.insert(double_bit_pattern(qd));
    }

    const double inv_n = 1.0 / static_cast<double>(v.size());
    return {
        oor * inv_n,
        clip_high * inv_n,
        clip_low * inv_n,
        rel_qerr_sum * inv_n,
        zero_after_quant * inv_n,
        static_cast<double>(uniq.size()) * inv_n,
    };
}

template<typename Posit>
inline QuantFeatureStats compute_mat_quant_features(const std::vector<std::vector<double>> &A) {
    std::vector<double> vals;
    for (const auto &row : A) {
        vals.insert(vals.end(), row.begin(), row.end());
    }
    return compute_vec_quant_features<Posit>(vals);
}
