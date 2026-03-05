#pragma once
#include <cmath>
#include <iostream>
#include <algorithm>

inline void validate_max_abs(const float* ref, const float* out, int n) {
  float max_err = 0.0f;
  for (int i = 0; i < n; ++i) {
    float err = std::fabs(ref[i] - out[i]);
    max_err = std::max(max_err, err);
  }
  std::cout << "max_abs_error: " << max_err << std::endl;
}
