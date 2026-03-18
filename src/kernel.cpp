#include <vector>
#include <cstddef> 

// ===== dot: sum_i x[i] * y[i] =====
template<typename Real>
Real dot(const std::vector<Real> &a, const std::vector<Real> &b) {
    Real s = Real(0);
    std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

// ===== sum: sum_i x[i] =====
template<typename Real>
Real sum(const std::vector<Real> &x) {
    Real s = Real(0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        s += x[i];
    }
    return s;
}

// ===== axpy_sum: sum_i (alpha * x[i] + y[i]) =====
template<typename Real>
Real axpy_sum(Real alpha,
              const std::vector<Real> &x,
              const std::vector<Real> &y) {
    Real s = Real(0);
    std::size_t n = x.size();
    for (std::size_t i = 0; i < n; ++i) {
        s += alpha * x[i] + y[i];
    }
    return s;
}

// ===== relu_sum: sum_i max(x[i], 0) =====
// 模擬簡單的 activation + reduction
template<typename Real>
Real relu_sum(const std::vector<Real> &x) {
    Real s = Real(0);
    Real zero = Real(0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        s += (x[i] > zero) ? x[i] : zero;
    }
    return s;
}

// ===== l1_norm: sum_i |x[i]| =====
template<typename Real>
Real l1_norm(const std::vector<Real> &x) {
    Real s = Real(0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        s += (x[i] >= Real(0)) ? x[i] : -x[i];
    }
    return s;
}

// ===== sum_squares: sum_i (x[i]^2) =====
template<typename Real>
Real sum_squares(const std::vector<Real> &x) {
    Real s = Real(0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        s += x[i] * x[i];
    }
    return s;
} 
// ===== axpby_sum: sum_i (alpha * x[i] + beta * y[i]) =====
template<typename Real>
Real axpby_sum(Real alpha,
               const std::vector<Real> &x,
               Real beta,
               const std::vector<Real> &y) {
    Real s = Real(0);
    std::size_t n = x.size();
    for (std::size_t i = 0; i < n; ++i) {
        s += alpha * x[i] + beta * y[i];
    }
    return s;
}

// ===== matvec_sum: sum over all rows of (A_row dot x) =====
template<typename Real>
Real matvec_sum(const std::vector<std::vector<Real>> &A, const std::vector<Real> &x) {
    Real total = Real(0);
    for (const auto &row : A) {
        total += dot<Real>(row, x);
    }
    return total;
}

// ===== prefix_sum_total: sum over prefix sums (triangular sum) =====
template<typename Real>
Real prefix_sum_total(const std::vector<Real> &x) {
    Real running = Real(0);
    Real total = Real(0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        running += x[i];
        total += running;
    }
    return total;
}
