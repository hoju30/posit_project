#include <vector>
using namespace std;

// ===== dot: sum_i x[i] * y[i] =====
template<typename Real>
Real dot(const vector<Real> &a, const vector<Real> &b) {
    Real s = Real(0);
    for (size_t i = 0; i < a.size(); ++i) {
        s += a[i] * b[i];
    }
    return s;
}

// ===== sum: sum_i x[i] =====
template<typename Real>
Real sum(const vector<Real> &x) {
    Real s = Real(0);
    for (size_t i = 0; i < x.size(); ++i) {
        s += x[i];
    }
    return s;
}

// ===== l1_norm: sum_i |x[i]| =====
template<typename Real>
Real l1_norm(const vector<Real> &x) {
    Real s = Real(0);
    for (size_t i = 0; i < x.size(); ++i) {
        s += (x[i] >= Real(0)) ? x[i] : -x[i];
    }
    return s;
}

// ===== sum_squares: sum_i (x[i]^2) =====
template<typename Real>
Real sum_squares(const vector<Real> &x) {
    Real s = Real(0);
    for (size_t i = 0; i < x.size(); ++i) {
        s += x[i] * x[i];
    }
    return s;
}

// ===== axpy_sum: sum_i (alpha * x[i] + y[i]) =====
// 常見 BLAS pattern：y <- alpha * x + y，再把結果加總
template<typename Real>
Real axpy_sum(Real alpha,
              const vector<Real> &x,
              const vector<Real> &y) {
    Real s = Real(0);
    size_t n = x.size();

    // 假設 x.size() == y.size()
    for (size_t i = 0; i < n; ++i) {
        s += alpha * x[i] + y[i];
    }
    return s;
}

// ===== axpby_sum: sum_i (alpha * x[i] + beta * y[i]) =====
template<typename Real>
Real axpby_sum(Real alpha,
               const vector<Real> &x,
               Real beta,
               const vector<Real> &y) {
    Real s = Real(0);
    size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        s += alpha * x[i] + beta * y[i];
    }
    return s;
}

// ===== relu_sum: sum_i max(x[i], 0) =====
// 模擬簡單的 activation + reduction
template<typename Real>
Real relu_sum(const vector<Real> &x) {
    Real s = Real(0);
    Real zero = Real(0);
    for (size_t i = 0; i < x.size(); ++i) {
        s += (x[i] > zero) ? x[i] : zero;
    }
    return s;
}

// ===== matvec_sum: sum over all rows of (A_row dot x) =====
// A: m x n，x: n，回傳所有行 dot 的加總（標量）
template<typename Real>
Real matvec_sum(const vector<vector<Real>> &A, const vector<Real> &x) {
    Real total = Real(0);
    for (const auto &row : A) {
        total += dot<Real>(row, x);
    }
    return total;
}

// ===== prefix_sum_total: sum over prefix sums (triangular sum) =====
template<typename Real>
Real prefix_sum_total(const vector<Real> &x) {
    Real running = Real(0);
    Real total = Real(0);
    for (size_t i = 0; i < x.size(); ++i) {
        running += x[i];
        total += running;
    }
    return total;
}
