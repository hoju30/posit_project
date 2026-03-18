#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <cmath>

int main() {
    // 建立亂數產生器：使用 Mersenne Twister + 常態分布
    std::mt19937 rng(42); // 固定 seed，方便重現結果
    std::normal_distribution<double> dist(0.0, 1.0);

    const int N = 10;
    std::vector<double> a(N), b(N), c(N);

    // 產生兩組隨機浮點數
    for (int i = 0; i < N; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }

    // 做一些隨便的浮點數運算：
    // c[i] = (a[i] * b[i] + a[i] / (std::abs(b[i]) + 1e-6)) 的平方根
    for (int i = 0; i < N; ++i) {
        double prod = a[i] * b[i];                         // 乘
        double safe_div = a[i] / (std::abs(b[i]) + 1e-6);  // 除（避免除以 0）
        double sum = prod + safe_div;                      // 加
        c[i] = std::sqrt(std::abs(sum));                   // 開根號 + 絕對值
    }

    return 0;
}
