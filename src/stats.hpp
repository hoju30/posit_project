#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>
#include <vector>

struct VecStats {
    double mean;
    double std;
    double p1;
    double p50;
    double p99;
};

inline std::tuple<double, double, double> abs_percentiles_nearest(const std::vector<double> &v) {
    if (v.empty()) return {0.0, 0.0, 0.0};
    std::vector<double> tmp(v.size());
    for (size_t i = 0; i < v.size(); ++i) tmp[i] = std::fabs(v[i]);
    size_t n = tmp.size();
    size_t k1 = static_cast<size_t>(std::floor(0.01 * static_cast<double>(n - 1)));
    size_t k50 = static_cast<size_t>(std::floor(0.50 * static_cast<double>(n - 1)));
    size_t k99 = static_cast<size_t>(std::floor(0.99 * static_cast<double>(n - 1)));
    if (k1 >= n) k1 = n - 1;
    if (k50 >= n) k50 = n - 1;
    if (k99 >= n) k99 = n - 1;

    std::nth_element(tmp.begin(), tmp.begin() + k1, tmp.end());
    double p1 = tmp[k1];
    std::nth_element(tmp.begin(), tmp.begin() + k50, tmp.end());
    double p50 = tmp[k50];
    std::nth_element(tmp.begin(), tmp.begin() + k99, tmp.end());
    double p99 = tmp[k99];
    return {p1, p50, p99};
}

inline VecStats compute_vec_stats_absq(const std::vector<double> &v) {
    if (v.empty()) return {0.0, 0.0, 0.0, 0.0, 0.0};
    double mean = 0.0;
    double sq = 0.0;
    for (double val : v) {
        mean += val;
        sq += val * val;
    }
    mean /= static_cast<double>(v.size());
    double var = sq / static_cast<double>(v.size()) - mean * mean;
    double stdv = (var > 0.0) ? std::sqrt(var) : 0.0;
    auto [p1, p50, p99] = abs_percentiles_nearest(v);
    return {mean, stdv, p1, p50, p99};
}

inline VecStats compute_mat_stats_absq(const std::vector<std::vector<double>> &A) {
    size_t rows = A.size();
    if (rows == 0) return {0.0, 0.0, 0.0, 0.0, 0.0};
    size_t cols = A[0].size();
    if (cols == 0) return {0.0, 0.0, 0.0, 0.0, 0.0};

    double mean = 0.0;
    double sq = 0.0;
    std::vector<double> abs_vals;
    abs_vals.reserve(rows * cols);
    size_t count = 0;
    for (const auto &row : A) {
        for (double val : row) {
            mean += val;
            sq += val * val;
            abs_vals.push_back(std::fabs(val));
            ++count;
        }
    }
    if (count == 0) return {0.0, 0.0, 0.0, 0.0, 0.0};
    mean /= static_cast<double>(count);
    double var = sq / static_cast<double>(count) - mean * mean;
    double stdv = (var > 0.0) ? std::sqrt(var) : 0.0;

    size_t n = abs_vals.size();
    size_t k1 = static_cast<size_t>(std::floor(0.01 * static_cast<double>(n - 1)));
    size_t k50 = static_cast<size_t>(std::floor(0.50 * static_cast<double>(n - 1)));
    size_t k99 = static_cast<size_t>(std::floor(0.99 * static_cast<double>(n - 1)));
    if (k1 >= n) k1 = n - 1;
    if (k50 >= n) k50 = n - 1;
    if (k99 >= n) k99 = n - 1;

    std::nth_element(abs_vals.begin(), abs_vals.begin() + k1, abs_vals.end());
    double p1 = abs_vals[k1];
    std::nth_element(abs_vals.begin(), abs_vals.begin() + k50, abs_vals.end());
    double p50 = abs_vals[k50];
    std::nth_element(abs_vals.begin(), abs_vals.begin() + k99, abs_vals.end());
    double p99 = abs_vals[k99];

    return {mean, stdv, p1, p50, p99};
}

