#ifndef PYVORO2_NATIVE_FP_CONTRACT_HPP
#define PYVORO2_NATIVE_FP_CONTRACT_HPP

// CMake force-includes this binding-owned guard in every participating 3D
// translation unit, including unchanged vendor sources and standalone tests.
#include <cfloat>
#include <limits>

#if defined(__FAST_MATH__) || defined(_M_FP_FAST)
#error "the 3D native FP contract forbids fast-math"
#endif

static_assert(sizeof(double) == 8 &&
                  std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::radix == 2 &&
                  std::numeric_limits<double>::digits == 53 &&
                  std::numeric_limits<double>::max_exponent == 1024 &&
                  std::numeric_limits<double>::has_denorm == std::denorm_present,
              "the 3D native FP contract requires IEC binary64 double");
static_assert(FLT_EVAL_METHOD == 0,
              "the 3D native FP contract forbids excess expression precision");

#endif
