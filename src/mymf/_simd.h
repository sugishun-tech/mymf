#ifndef MYMF_SIMD_H
#define MYMF_SIMD_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#ifdef _MSC_VER
#define MYMF_RESTRICT __restrict
#else
#define MYMF_RESTRICT restrict
#endif
#if defined(_OPENMP) && !defined(_MSC_VER)
#define MYMF_SIMD _Pragma("omp simd")
#else
#define MYMF_SIMD
#endif
/* Portable wheel: dispatch once per row, never once per nonzero. */
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && !defined(__APPLE__)
#define MYMF_RUNTIME_X86 1
#define MYMF_AVX2 __attribute__((target("avx2,fma")))
#define MYMF_AVX512 __attribute__((target("avx512f,fma")))
#else
#define MYMF_RUNTIME_X86 0
#endif

/* GCC's outer-loop unroll-and-jam can scalarize the SIMD inner loop.
   Disable only that transform, not floating-point correctness checks. */
#if defined(__GNUC__) && !defined(__clang__)
#define MYMF_NOJAM __attribute__((optimize("no-loop-unroll-and-jam")))
#else
#define MYMF_NOJAM
#endif
#define MYMF_ROW(NAME, ATTR, REAL, INDEX) \
ATTR MYMF_NOJAM static void NAME(const REAL * MYMF_RESTRICT data, \
    const INDEX * MYMF_RESTRICT indices, size_t begin, size_t end, \
    const REAL * MYMF_RESTRICT b, REAL * MYMF_RESTRICT out, size_t k) { \
    memset(out, 0, k * sizeof(REAL)); \
    for (size_t p = begin; p < end; ++p) { \
        const REAL a = data[p]; \
        const REAL *x = b + (size_t)indices[p] * k; \
        MYMF_SIMD \
        for (size_t j = 0; j < k; ++j) out[j] += a * x[j]; \
    } \
}

MYMF_ROW(row_f32_i32_base, , float, int32_t)
MYMF_ROW(row_f32_i64_base, , float, int64_t)
MYMF_ROW(row_f64_i32_base, , double, int32_t)
MYMF_ROW(row_f64_i64_base, , double, int64_t)
#if MYMF_RUNTIME_X86
MYMF_ROW(row_f32_i32_avx2, MYMF_AVX2, float, int32_t)
MYMF_ROW(row_f32_i64_avx2, MYMF_AVX2, float, int64_t)
MYMF_ROW(row_f64_i32_avx2, MYMF_AVX2, double, int32_t)
MYMF_ROW(row_f64_i64_avx2, MYMF_AVX2, double, int64_t)
MYMF_ROW(row_f32_i32_avx512, MYMF_AVX512, float, int32_t)
MYMF_ROW(row_f32_i64_avx512, MYMF_AVX512, float, int64_t)
MYMF_ROW(row_f64_i32_avx512, MYMF_AVX512, double, int32_t)
MYMF_ROW(row_f64_i64_avx512, MYMF_AVX512, double, int64_t)
#endif

#include "_reg.h"

#define MYMF_FN(REAL, INDEX) void (*)(const REAL*, const INDEX*, size_t, size_t, const REAL*, REAL*, size_t)
typedef void (*f32_i32_t)(const float*, const int32_t*, size_t, size_t, const float*, float*, size_t);
typedef void (*f32_i64_t)(const float*, const int64_t*, size_t, size_t, const float*, float*, size_t);
typedef void (*f64_i32_t)(const double*, const int32_t*, size_t, size_t, const double*, double*, size_t);
typedef void (*f64_i64_t)(const double*, const int64_t*, size_t, size_t, const double*, double*, size_t);
static f32_i32_t mymf_row_f32_i32 = row_f32_i32_base;
static f32_i64_t mymf_row_f32_i64 = row_f32_i64_base;
static f64_i32_t mymf_row_f64_i32 = row_f64_i32_base;
static f64_i64_t mymf_row_f64_i64 = row_f64_i64_base;
static const char *mymf_simd = "baseline";
/* Called once when the extension is imported, while holding the Python GIL. */
static void mymf_init(void) {
#if MYMF_RUNTIME_X86
    __builtin_cpu_init();
    const char *request = getenv("MYMF_SIMD");
    if (request && strcmp(request, "baseline") == 0) return;
    if ((!request || strcmp(request, "avx2") != 0) && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma")) {
        mymf_row_f32_i32 = reg_f32_i32_avx512;
        mymf_row_f32_i64 = reg_f32_i64_avx512;
        mymf_row_f64_i32 = reg_f64_i32_avx512;
        mymf_row_f64_i64 = reg_f64_i64_avx512;
        mymf_simd = "avx512f+fma";
    } else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        mymf_row_f32_i32 = reg_f32_i32_avx2;
        mymf_row_f32_i64 = reg_f32_i64_avx2;
        mymf_row_f64_i32 = reg_f64_i32_avx2;
        mymf_row_f64_i64 = reg_f64_i64_avx2;
        mymf_simd = "avx2+fma";
    }
#endif
}
static const char *mymf_simd_name(void) { return mymf_simd; }
static int mymf_openmp_enabled(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}
static int mymf_native_enabled(void) { return MYMF_NATIVE_BUILD; }
#endif
