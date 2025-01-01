#include <c10/macros/Macros.h>

template <typename T>
inline C10_HOST_DEVICE T calc_mymuladd(T a, T b, T c) {
  return a * b + c;
}

template <typename T>
inline C10_HOST_DEVICE T calc_mymul(T a, T b) {
  return a * b;
}
