/* Register accumulators for short dense rows. Generated cases differ only
   in vector count; they do not recognize datasets, ranks, or seeds. The final
   vector is masked so no load can cross a row or buffer boundary. */
#ifndef MYMF_REG_H
#define MYMF_REG_H
#if MYMF_RUNTIME_X86
#include <immintrin.h>
MYMF_AVX512 static void reg_f32_i32_avx512(const float * MYMF_RESTRICT data,
    const int32_t * MYMF_RESTRICT indices, size_t begin, size_t end,
    const float * MYMF_RESTRICT b, float * MYMF_RESTRICT out, size_t k) {
    if (k > 128) { row_f32_i32_avx512(data, indices, begin, end, b, out, k); return; }
    const size_t last = (k - 1) % 16 + 1;
    const __mmask16 mask = (__mmask16)((1u << last) - 1u);
    switch ((k + 15) / 16) {
    case 1: {
        __m512 s0 = _mm512_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m512 a = _mm512_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_ps(a, _mm512_maskz_loadu_ps(mask, x + 0), s0);
        }
        _mm512_mask_storeu_ps(out + 0, mask, s0);
        return;
    }
    case 2: {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m512 a = _mm512_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 0), s0);
            s1 = _mm512_fmadd_ps(a, _mm512_maskz_loadu_ps(mask, x + 16), s1);
        }
        _mm512_storeu_ps(out + 0, s0);
        _mm512_mask_storeu_ps(out + 16, mask, s1);
        return;
    }
    case 3: {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m512 a = _mm512_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 0), s0);
            s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 16), s1);
            s2 = _mm512_fmadd_ps(a, _mm512_maskz_loadu_ps(mask, x + 32), s2);
        }
        _mm512_storeu_ps(out + 0, s0);
        _mm512_storeu_ps(out + 16, s1);
        _mm512_mask_storeu_ps(out + 32, mask, s2);
        return;
    }
    case 4: {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m512 a = _mm512_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 0), s0);
            s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 16), s1);
            s2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 32), s2);
            s3 = _mm512_fmadd_ps(a, _mm512_maskz_loadu_ps(mask, x + 48), s3);
        }
        _mm512_storeu_ps(out + 0, s0);
        _mm512_storeu_ps(out + 16, s1);
        _mm512_storeu_ps(out + 32, s2);
        _mm512_mask_storeu_ps(out + 48, mask, s3);
        return;
    }
    case 5: {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        __m512 s4 = _mm512_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m512 a = _mm512_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 0), s0);
            s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 16), s1);
            s2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 32), s2);
            s3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 48), s3);
            s4 = _mm512_fmadd_ps(a, _mm512_maskz_loadu_ps(mask, x + 64), s4);
        }
        _mm512_storeu_ps(out + 0, s0);
        _mm512_storeu_ps(out + 16, s1);
        _mm512_storeu_ps(out + 32, s2);
        _mm512_storeu_ps(out + 48, s3);
        _mm512_mask_storeu_ps(out + 64, mask, s4);
        return;
    }
    case 6: {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        __m512 s4 = _mm512_setzero_ps();
        __m512 s5 = _mm512_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m512 a = _mm512_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 0), s0);
            s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 16), s1);
            s2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 32), s2);
            s3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 48), s3);
            s4 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 64), s4);
            s5 = _mm512_fmadd_ps(a, _mm512_maskz_loadu_ps(mask, x + 80), s5);
        }
        _mm512_storeu_ps(out + 0, s0);
        _mm512_storeu_ps(out + 16, s1);
        _mm512_storeu_ps(out + 32, s2);
        _mm512_storeu_ps(out + 48, s3);
        _mm512_storeu_ps(out + 64, s4);
        _mm512_mask_storeu_ps(out + 80, mask, s5);
        return;
    }
    case 7: {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        __m512 s4 = _mm512_setzero_ps();
        __m512 s5 = _mm512_setzero_ps();
        __m512 s6 = _mm512_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m512 a = _mm512_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 0), s0);
            s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 16), s1);
            s2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 32), s2);
            s3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 48), s3);
            s4 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 64), s4);
            s5 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 80), s5);
            s6 = _mm512_fmadd_ps(a, _mm512_maskz_loadu_ps(mask, x + 96), s6);
        }
        _mm512_storeu_ps(out + 0, s0);
        _mm512_storeu_ps(out + 16, s1);
        _mm512_storeu_ps(out + 32, s2);
        _mm512_storeu_ps(out + 48, s3);
        _mm512_storeu_ps(out + 64, s4);
        _mm512_storeu_ps(out + 80, s5);
        _mm512_mask_storeu_ps(out + 96, mask, s6);
        return;
    }
    case 8: {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        __m512 s4 = _mm512_setzero_ps();
        __m512 s5 = _mm512_setzero_ps();
        __m512 s6 = _mm512_setzero_ps();
        __m512 s7 = _mm512_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m512 a = _mm512_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 0), s0);
            s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 16), s1);
            s2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 32), s2);
            s3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 48), s3);
            s4 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 64), s4);
            s5 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 80), s5);
            s6 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 96), s6);
            s7 = _mm512_fmadd_ps(a, _mm512_maskz_loadu_ps(mask, x + 112), s7);
        }
        _mm512_storeu_ps(out + 0, s0);
        _mm512_storeu_ps(out + 16, s1);
        _mm512_storeu_ps(out + 32, s2);
        _mm512_storeu_ps(out + 48, s3);
        _mm512_storeu_ps(out + 64, s4);
        _mm512_storeu_ps(out + 80, s5);
        _mm512_storeu_ps(out + 96, s6);
        _mm512_mask_storeu_ps(out + 112, mask, s7);
        return;
    }
    }
}
MYMF_AVX512 static void reg_f32_i64_avx512(const float * MYMF_RESTRICT data,
    const int64_t * MYMF_RESTRICT indices, size_t begin, size_t end,
    const float * MYMF_RESTRICT b, float * MYMF_RESTRICT out, size_t k) {
    if (k > 128) { row_f32_i64_avx512(data, indices, begin, end, b, out, k); return; }
    const size_t last = (k - 1) % 16 + 1;
    const __mmask16 mask = (__mmask16)((1u << last) - 1u);
    switch ((k + 15) / 16) {
    case 1: {
        __m512 s0 = _mm512_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m512 a = _mm512_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_ps(a, _mm512_maskz_loadu_ps(mask, x + 0), s0);
        }
        _mm512_mask_storeu_ps(out + 0, mask, s0);
        return;
    }
    case 2: {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m512 a = _mm512_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 0), s0);
            s1 = _mm512_fmadd_ps(a, _mm512_maskz_loadu_ps(mask, x + 16), s1);
        }
        _mm512_storeu_ps(out + 0, s0);
        _mm512_mask_storeu_ps(out + 16, mask, s1);
        return;
    }
    case 3: {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m512 a = _mm512_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 0), s0);
            s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 16), s1);
            s2 = _mm512_fmadd_ps(a, _mm512_maskz_loadu_ps(mask, x + 32), s2);
        }
        _mm512_storeu_ps(out + 0, s0);
        _mm512_storeu_ps(out + 16, s1);
        _mm512_mask_storeu_ps(out + 32, mask, s2);
        return;
    }
    case 4: {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m512 a = _mm512_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 0), s0);
            s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 16), s1);
            s2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 32), s2);
            s3 = _mm512_fmadd_ps(a, _mm512_maskz_loadu_ps(mask, x + 48), s3);
        }
        _mm512_storeu_ps(out + 0, s0);
        _mm512_storeu_ps(out + 16, s1);
        _mm512_storeu_ps(out + 32, s2);
        _mm512_mask_storeu_ps(out + 48, mask, s3);
        return;
    }
    case 5: {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        __m512 s4 = _mm512_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m512 a = _mm512_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 0), s0);
            s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 16), s1);
            s2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 32), s2);
            s3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 48), s3);
            s4 = _mm512_fmadd_ps(a, _mm512_maskz_loadu_ps(mask, x + 64), s4);
        }
        _mm512_storeu_ps(out + 0, s0);
        _mm512_storeu_ps(out + 16, s1);
        _mm512_storeu_ps(out + 32, s2);
        _mm512_storeu_ps(out + 48, s3);
        _mm512_mask_storeu_ps(out + 64, mask, s4);
        return;
    }
    case 6: {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        __m512 s4 = _mm512_setzero_ps();
        __m512 s5 = _mm512_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m512 a = _mm512_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 0), s0);
            s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 16), s1);
            s2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 32), s2);
            s3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 48), s3);
            s4 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 64), s4);
            s5 = _mm512_fmadd_ps(a, _mm512_maskz_loadu_ps(mask, x + 80), s5);
        }
        _mm512_storeu_ps(out + 0, s0);
        _mm512_storeu_ps(out + 16, s1);
        _mm512_storeu_ps(out + 32, s2);
        _mm512_storeu_ps(out + 48, s3);
        _mm512_storeu_ps(out + 64, s4);
        _mm512_mask_storeu_ps(out + 80, mask, s5);
        return;
    }
    case 7: {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        __m512 s4 = _mm512_setzero_ps();
        __m512 s5 = _mm512_setzero_ps();
        __m512 s6 = _mm512_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m512 a = _mm512_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 0), s0);
            s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 16), s1);
            s2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 32), s2);
            s3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 48), s3);
            s4 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 64), s4);
            s5 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 80), s5);
            s6 = _mm512_fmadd_ps(a, _mm512_maskz_loadu_ps(mask, x + 96), s6);
        }
        _mm512_storeu_ps(out + 0, s0);
        _mm512_storeu_ps(out + 16, s1);
        _mm512_storeu_ps(out + 32, s2);
        _mm512_storeu_ps(out + 48, s3);
        _mm512_storeu_ps(out + 64, s4);
        _mm512_storeu_ps(out + 80, s5);
        _mm512_mask_storeu_ps(out + 96, mask, s6);
        return;
    }
    case 8: {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        __m512 s4 = _mm512_setzero_ps();
        __m512 s5 = _mm512_setzero_ps();
        __m512 s6 = _mm512_setzero_ps();
        __m512 s7 = _mm512_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m512 a = _mm512_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 0), s0);
            s1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 16), s1);
            s2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 32), s2);
            s3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 48), s3);
            s4 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 64), s4);
            s5 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 80), s5);
            s6 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x + 96), s6);
            s7 = _mm512_fmadd_ps(a, _mm512_maskz_loadu_ps(mask, x + 112), s7);
        }
        _mm512_storeu_ps(out + 0, s0);
        _mm512_storeu_ps(out + 16, s1);
        _mm512_storeu_ps(out + 32, s2);
        _mm512_storeu_ps(out + 48, s3);
        _mm512_storeu_ps(out + 64, s4);
        _mm512_storeu_ps(out + 80, s5);
        _mm512_storeu_ps(out + 96, s6);
        _mm512_mask_storeu_ps(out + 112, mask, s7);
        return;
    }
    }
}
MYMF_AVX512 static void reg_f64_i32_avx512(const double * MYMF_RESTRICT data,
    const int32_t * MYMF_RESTRICT indices, size_t begin, size_t end,
    const double * MYMF_RESTRICT b, double * MYMF_RESTRICT out, size_t k) {
    if (k > 64) { row_f64_i32_avx512(data, indices, begin, end, b, out, k); return; }
    const size_t last = (k - 1) % 8 + 1;
    const __mmask8 mask = (__mmask8)((1u << last) - 1u);
    switch ((k + 7) / 8) {
    case 1: {
        __m512d s0 = _mm512_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m512d a = _mm512_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_pd(a, _mm512_maskz_loadu_pd(mask, x + 0), s0);
        }
        _mm512_mask_storeu_pd(out + 0, mask, s0);
        return;
    }
    case 2: {
        __m512d s0 = _mm512_setzero_pd();
        __m512d s1 = _mm512_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m512d a = _mm512_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 0), s0);
            s1 = _mm512_fmadd_pd(a, _mm512_maskz_loadu_pd(mask, x + 8), s1);
        }
        _mm512_storeu_pd(out + 0, s0);
        _mm512_mask_storeu_pd(out + 8, mask, s1);
        return;
    }
    case 3: {
        __m512d s0 = _mm512_setzero_pd();
        __m512d s1 = _mm512_setzero_pd();
        __m512d s2 = _mm512_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m512d a = _mm512_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 0), s0);
            s1 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 8), s1);
            s2 = _mm512_fmadd_pd(a, _mm512_maskz_loadu_pd(mask, x + 16), s2);
        }
        _mm512_storeu_pd(out + 0, s0);
        _mm512_storeu_pd(out + 8, s1);
        _mm512_mask_storeu_pd(out + 16, mask, s2);
        return;
    }
    case 4: {
        __m512d s0 = _mm512_setzero_pd();
        __m512d s1 = _mm512_setzero_pd();
        __m512d s2 = _mm512_setzero_pd();
        __m512d s3 = _mm512_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m512d a = _mm512_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 0), s0);
            s1 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 8), s1);
            s2 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 16), s2);
            s3 = _mm512_fmadd_pd(a, _mm512_maskz_loadu_pd(mask, x + 24), s3);
        }
        _mm512_storeu_pd(out + 0, s0);
        _mm512_storeu_pd(out + 8, s1);
        _mm512_storeu_pd(out + 16, s2);
        _mm512_mask_storeu_pd(out + 24, mask, s3);
        return;
    }
    case 5: {
        __m512d s0 = _mm512_setzero_pd();
        __m512d s1 = _mm512_setzero_pd();
        __m512d s2 = _mm512_setzero_pd();
        __m512d s3 = _mm512_setzero_pd();
        __m512d s4 = _mm512_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m512d a = _mm512_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 0), s0);
            s1 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 8), s1);
            s2 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 16), s2);
            s3 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 24), s3);
            s4 = _mm512_fmadd_pd(a, _mm512_maskz_loadu_pd(mask, x + 32), s4);
        }
        _mm512_storeu_pd(out + 0, s0);
        _mm512_storeu_pd(out + 8, s1);
        _mm512_storeu_pd(out + 16, s2);
        _mm512_storeu_pd(out + 24, s3);
        _mm512_mask_storeu_pd(out + 32, mask, s4);
        return;
    }
    case 6: {
        __m512d s0 = _mm512_setzero_pd();
        __m512d s1 = _mm512_setzero_pd();
        __m512d s2 = _mm512_setzero_pd();
        __m512d s3 = _mm512_setzero_pd();
        __m512d s4 = _mm512_setzero_pd();
        __m512d s5 = _mm512_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m512d a = _mm512_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 0), s0);
            s1 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 8), s1);
            s2 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 16), s2);
            s3 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 24), s3);
            s4 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 32), s4);
            s5 = _mm512_fmadd_pd(a, _mm512_maskz_loadu_pd(mask, x + 40), s5);
        }
        _mm512_storeu_pd(out + 0, s0);
        _mm512_storeu_pd(out + 8, s1);
        _mm512_storeu_pd(out + 16, s2);
        _mm512_storeu_pd(out + 24, s3);
        _mm512_storeu_pd(out + 32, s4);
        _mm512_mask_storeu_pd(out + 40, mask, s5);
        return;
    }
    case 7: {
        __m512d s0 = _mm512_setzero_pd();
        __m512d s1 = _mm512_setzero_pd();
        __m512d s2 = _mm512_setzero_pd();
        __m512d s3 = _mm512_setzero_pd();
        __m512d s4 = _mm512_setzero_pd();
        __m512d s5 = _mm512_setzero_pd();
        __m512d s6 = _mm512_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m512d a = _mm512_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 0), s0);
            s1 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 8), s1);
            s2 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 16), s2);
            s3 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 24), s3);
            s4 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 32), s4);
            s5 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 40), s5);
            s6 = _mm512_fmadd_pd(a, _mm512_maskz_loadu_pd(mask, x + 48), s6);
        }
        _mm512_storeu_pd(out + 0, s0);
        _mm512_storeu_pd(out + 8, s1);
        _mm512_storeu_pd(out + 16, s2);
        _mm512_storeu_pd(out + 24, s3);
        _mm512_storeu_pd(out + 32, s4);
        _mm512_storeu_pd(out + 40, s5);
        _mm512_mask_storeu_pd(out + 48, mask, s6);
        return;
    }
    case 8: {
        __m512d s0 = _mm512_setzero_pd();
        __m512d s1 = _mm512_setzero_pd();
        __m512d s2 = _mm512_setzero_pd();
        __m512d s3 = _mm512_setzero_pd();
        __m512d s4 = _mm512_setzero_pd();
        __m512d s5 = _mm512_setzero_pd();
        __m512d s6 = _mm512_setzero_pd();
        __m512d s7 = _mm512_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m512d a = _mm512_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 0), s0);
            s1 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 8), s1);
            s2 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 16), s2);
            s3 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 24), s3);
            s4 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 32), s4);
            s5 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 40), s5);
            s6 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 48), s6);
            s7 = _mm512_fmadd_pd(a, _mm512_maskz_loadu_pd(mask, x + 56), s7);
        }
        _mm512_storeu_pd(out + 0, s0);
        _mm512_storeu_pd(out + 8, s1);
        _mm512_storeu_pd(out + 16, s2);
        _mm512_storeu_pd(out + 24, s3);
        _mm512_storeu_pd(out + 32, s4);
        _mm512_storeu_pd(out + 40, s5);
        _mm512_storeu_pd(out + 48, s6);
        _mm512_mask_storeu_pd(out + 56, mask, s7);
        return;
    }
    }
}
MYMF_AVX512 static void reg_f64_i64_avx512(const double * MYMF_RESTRICT data,
    const int64_t * MYMF_RESTRICT indices, size_t begin, size_t end,
    const double * MYMF_RESTRICT b, double * MYMF_RESTRICT out, size_t k) {
    if (k > 64) { row_f64_i64_avx512(data, indices, begin, end, b, out, k); return; }
    const size_t last = (k - 1) % 8 + 1;
    const __mmask8 mask = (__mmask8)((1u << last) - 1u);
    switch ((k + 7) / 8) {
    case 1: {
        __m512d s0 = _mm512_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m512d a = _mm512_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_pd(a, _mm512_maskz_loadu_pd(mask, x + 0), s0);
        }
        _mm512_mask_storeu_pd(out + 0, mask, s0);
        return;
    }
    case 2: {
        __m512d s0 = _mm512_setzero_pd();
        __m512d s1 = _mm512_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m512d a = _mm512_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 0), s0);
            s1 = _mm512_fmadd_pd(a, _mm512_maskz_loadu_pd(mask, x + 8), s1);
        }
        _mm512_storeu_pd(out + 0, s0);
        _mm512_mask_storeu_pd(out + 8, mask, s1);
        return;
    }
    case 3: {
        __m512d s0 = _mm512_setzero_pd();
        __m512d s1 = _mm512_setzero_pd();
        __m512d s2 = _mm512_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m512d a = _mm512_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 0), s0);
            s1 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 8), s1);
            s2 = _mm512_fmadd_pd(a, _mm512_maskz_loadu_pd(mask, x + 16), s2);
        }
        _mm512_storeu_pd(out + 0, s0);
        _mm512_storeu_pd(out + 8, s1);
        _mm512_mask_storeu_pd(out + 16, mask, s2);
        return;
    }
    case 4: {
        __m512d s0 = _mm512_setzero_pd();
        __m512d s1 = _mm512_setzero_pd();
        __m512d s2 = _mm512_setzero_pd();
        __m512d s3 = _mm512_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m512d a = _mm512_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 0), s0);
            s1 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 8), s1);
            s2 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 16), s2);
            s3 = _mm512_fmadd_pd(a, _mm512_maskz_loadu_pd(mask, x + 24), s3);
        }
        _mm512_storeu_pd(out + 0, s0);
        _mm512_storeu_pd(out + 8, s1);
        _mm512_storeu_pd(out + 16, s2);
        _mm512_mask_storeu_pd(out + 24, mask, s3);
        return;
    }
    case 5: {
        __m512d s0 = _mm512_setzero_pd();
        __m512d s1 = _mm512_setzero_pd();
        __m512d s2 = _mm512_setzero_pd();
        __m512d s3 = _mm512_setzero_pd();
        __m512d s4 = _mm512_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m512d a = _mm512_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 0), s0);
            s1 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 8), s1);
            s2 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 16), s2);
            s3 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 24), s3);
            s4 = _mm512_fmadd_pd(a, _mm512_maskz_loadu_pd(mask, x + 32), s4);
        }
        _mm512_storeu_pd(out + 0, s0);
        _mm512_storeu_pd(out + 8, s1);
        _mm512_storeu_pd(out + 16, s2);
        _mm512_storeu_pd(out + 24, s3);
        _mm512_mask_storeu_pd(out + 32, mask, s4);
        return;
    }
    case 6: {
        __m512d s0 = _mm512_setzero_pd();
        __m512d s1 = _mm512_setzero_pd();
        __m512d s2 = _mm512_setzero_pd();
        __m512d s3 = _mm512_setzero_pd();
        __m512d s4 = _mm512_setzero_pd();
        __m512d s5 = _mm512_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m512d a = _mm512_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 0), s0);
            s1 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 8), s1);
            s2 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 16), s2);
            s3 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 24), s3);
            s4 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 32), s4);
            s5 = _mm512_fmadd_pd(a, _mm512_maskz_loadu_pd(mask, x + 40), s5);
        }
        _mm512_storeu_pd(out + 0, s0);
        _mm512_storeu_pd(out + 8, s1);
        _mm512_storeu_pd(out + 16, s2);
        _mm512_storeu_pd(out + 24, s3);
        _mm512_storeu_pd(out + 32, s4);
        _mm512_mask_storeu_pd(out + 40, mask, s5);
        return;
    }
    case 7: {
        __m512d s0 = _mm512_setzero_pd();
        __m512d s1 = _mm512_setzero_pd();
        __m512d s2 = _mm512_setzero_pd();
        __m512d s3 = _mm512_setzero_pd();
        __m512d s4 = _mm512_setzero_pd();
        __m512d s5 = _mm512_setzero_pd();
        __m512d s6 = _mm512_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m512d a = _mm512_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 0), s0);
            s1 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 8), s1);
            s2 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 16), s2);
            s3 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 24), s3);
            s4 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 32), s4);
            s5 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 40), s5);
            s6 = _mm512_fmadd_pd(a, _mm512_maskz_loadu_pd(mask, x + 48), s6);
        }
        _mm512_storeu_pd(out + 0, s0);
        _mm512_storeu_pd(out + 8, s1);
        _mm512_storeu_pd(out + 16, s2);
        _mm512_storeu_pd(out + 24, s3);
        _mm512_storeu_pd(out + 32, s4);
        _mm512_storeu_pd(out + 40, s5);
        _mm512_mask_storeu_pd(out + 48, mask, s6);
        return;
    }
    case 8: {
        __m512d s0 = _mm512_setzero_pd();
        __m512d s1 = _mm512_setzero_pd();
        __m512d s2 = _mm512_setzero_pd();
        __m512d s3 = _mm512_setzero_pd();
        __m512d s4 = _mm512_setzero_pd();
        __m512d s5 = _mm512_setzero_pd();
        __m512d s6 = _mm512_setzero_pd();
        __m512d s7 = _mm512_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m512d a = _mm512_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 0), s0);
            s1 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 8), s1);
            s2 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 16), s2);
            s3 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 24), s3);
            s4 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 32), s4);
            s5 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 40), s5);
            s6 = _mm512_fmadd_pd(a, _mm512_loadu_pd(x + 48), s6);
            s7 = _mm512_fmadd_pd(a, _mm512_maskz_loadu_pd(mask, x + 56), s7);
        }
        _mm512_storeu_pd(out + 0, s0);
        _mm512_storeu_pd(out + 8, s1);
        _mm512_storeu_pd(out + 16, s2);
        _mm512_storeu_pd(out + 24, s3);
        _mm512_storeu_pd(out + 32, s4);
        _mm512_storeu_pd(out + 40, s5);
        _mm512_storeu_pd(out + 48, s6);
        _mm512_mask_storeu_pd(out + 56, mask, s7);
        return;
    }
    }
}
MYMF_AVX2 static void reg_f32_i32_avx2(const float * MYMF_RESTRICT data,
    const int32_t * MYMF_RESTRICT indices, size_t begin, size_t end,
    const float * MYMF_RESTRICT b, float * MYMF_RESTRICT out, size_t k) {
    if (k > 64) { row_f32_i32_avx2(data, indices, begin, end, b, out, k); return; }
    const size_t last = (k - 1) % 8 + 1;
    const __m256i mask = _mm256_setr_epi32(last > 0 ? -1 : 0, last > 1 ? -1 : 0, last > 2 ? -1 : 0, last > 3 ? -1 : 0, last > 4 ? -1 : 0, last > 5 ? -1 : 0, last > 6 ? -1 : 0, last > 7 ? -1 : 0);
    switch ((k + 7) / 8) {
    case 1: {
        __m256 s0 = _mm256_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m256 a = _mm256_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_ps(a, _mm256_maskload_ps(x + 0, mask), s0);
        }
        _mm256_maskstore_ps(out + 0, mask, s0);
        return;
    }
    case 2: {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m256 a = _mm256_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 0), s0);
            s1 = _mm256_fmadd_ps(a, _mm256_maskload_ps(x + 8, mask), s1);
        }
        _mm256_storeu_ps(out + 0, s0);
        _mm256_maskstore_ps(out + 8, mask, s1);
        return;
    }
    case 3: {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m256 a = _mm256_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 0), s0);
            s1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 8), s1);
            s2 = _mm256_fmadd_ps(a, _mm256_maskload_ps(x + 16, mask), s2);
        }
        _mm256_storeu_ps(out + 0, s0);
        _mm256_storeu_ps(out + 8, s1);
        _mm256_maskstore_ps(out + 16, mask, s2);
        return;
    }
    case 4: {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m256 a = _mm256_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 0), s0);
            s1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 8), s1);
            s2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 16), s2);
            s3 = _mm256_fmadd_ps(a, _mm256_maskload_ps(x + 24, mask), s3);
        }
        _mm256_storeu_ps(out + 0, s0);
        _mm256_storeu_ps(out + 8, s1);
        _mm256_storeu_ps(out + 16, s2);
        _mm256_maskstore_ps(out + 24, mask, s3);
        return;
    }
    case 5: {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        __m256 s4 = _mm256_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m256 a = _mm256_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 0), s0);
            s1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 8), s1);
            s2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 16), s2);
            s3 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 24), s3);
            s4 = _mm256_fmadd_ps(a, _mm256_maskload_ps(x + 32, mask), s4);
        }
        _mm256_storeu_ps(out + 0, s0);
        _mm256_storeu_ps(out + 8, s1);
        _mm256_storeu_ps(out + 16, s2);
        _mm256_storeu_ps(out + 24, s3);
        _mm256_maskstore_ps(out + 32, mask, s4);
        return;
    }
    case 6: {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        __m256 s4 = _mm256_setzero_ps();
        __m256 s5 = _mm256_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m256 a = _mm256_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 0), s0);
            s1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 8), s1);
            s2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 16), s2);
            s3 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 24), s3);
            s4 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 32), s4);
            s5 = _mm256_fmadd_ps(a, _mm256_maskload_ps(x + 40, mask), s5);
        }
        _mm256_storeu_ps(out + 0, s0);
        _mm256_storeu_ps(out + 8, s1);
        _mm256_storeu_ps(out + 16, s2);
        _mm256_storeu_ps(out + 24, s3);
        _mm256_storeu_ps(out + 32, s4);
        _mm256_maskstore_ps(out + 40, mask, s5);
        return;
    }
    case 7: {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        __m256 s4 = _mm256_setzero_ps();
        __m256 s5 = _mm256_setzero_ps();
        __m256 s6 = _mm256_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m256 a = _mm256_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 0), s0);
            s1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 8), s1);
            s2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 16), s2);
            s3 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 24), s3);
            s4 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 32), s4);
            s5 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 40), s5);
            s6 = _mm256_fmadd_ps(a, _mm256_maskload_ps(x + 48, mask), s6);
        }
        _mm256_storeu_ps(out + 0, s0);
        _mm256_storeu_ps(out + 8, s1);
        _mm256_storeu_ps(out + 16, s2);
        _mm256_storeu_ps(out + 24, s3);
        _mm256_storeu_ps(out + 32, s4);
        _mm256_storeu_ps(out + 40, s5);
        _mm256_maskstore_ps(out + 48, mask, s6);
        return;
    }
    case 8: {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        __m256 s4 = _mm256_setzero_ps();
        __m256 s5 = _mm256_setzero_ps();
        __m256 s6 = _mm256_setzero_ps();
        __m256 s7 = _mm256_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m256 a = _mm256_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 0), s0);
            s1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 8), s1);
            s2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 16), s2);
            s3 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 24), s3);
            s4 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 32), s4);
            s5 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 40), s5);
            s6 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 48), s6);
            s7 = _mm256_fmadd_ps(a, _mm256_maskload_ps(x + 56, mask), s7);
        }
        _mm256_storeu_ps(out + 0, s0);
        _mm256_storeu_ps(out + 8, s1);
        _mm256_storeu_ps(out + 16, s2);
        _mm256_storeu_ps(out + 24, s3);
        _mm256_storeu_ps(out + 32, s4);
        _mm256_storeu_ps(out + 40, s5);
        _mm256_storeu_ps(out + 48, s6);
        _mm256_maskstore_ps(out + 56, mask, s7);
        return;
    }
    }
}
MYMF_AVX2 static void reg_f32_i64_avx2(const float * MYMF_RESTRICT data,
    const int64_t * MYMF_RESTRICT indices, size_t begin, size_t end,
    const float * MYMF_RESTRICT b, float * MYMF_RESTRICT out, size_t k) {
    if (k > 64) { row_f32_i64_avx2(data, indices, begin, end, b, out, k); return; }
    const size_t last = (k - 1) % 8 + 1;
    const __m256i mask = _mm256_setr_epi32(last > 0 ? -1 : 0, last > 1 ? -1 : 0, last > 2 ? -1 : 0, last > 3 ? -1 : 0, last > 4 ? -1 : 0, last > 5 ? -1 : 0, last > 6 ? -1 : 0, last > 7 ? -1 : 0);
    switch ((k + 7) / 8) {
    case 1: {
        __m256 s0 = _mm256_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m256 a = _mm256_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_ps(a, _mm256_maskload_ps(x + 0, mask), s0);
        }
        _mm256_maskstore_ps(out + 0, mask, s0);
        return;
    }
    case 2: {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m256 a = _mm256_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 0), s0);
            s1 = _mm256_fmadd_ps(a, _mm256_maskload_ps(x + 8, mask), s1);
        }
        _mm256_storeu_ps(out + 0, s0);
        _mm256_maskstore_ps(out + 8, mask, s1);
        return;
    }
    case 3: {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m256 a = _mm256_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 0), s0);
            s1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 8), s1);
            s2 = _mm256_fmadd_ps(a, _mm256_maskload_ps(x + 16, mask), s2);
        }
        _mm256_storeu_ps(out + 0, s0);
        _mm256_storeu_ps(out + 8, s1);
        _mm256_maskstore_ps(out + 16, mask, s2);
        return;
    }
    case 4: {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m256 a = _mm256_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 0), s0);
            s1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 8), s1);
            s2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 16), s2);
            s3 = _mm256_fmadd_ps(a, _mm256_maskload_ps(x + 24, mask), s3);
        }
        _mm256_storeu_ps(out + 0, s0);
        _mm256_storeu_ps(out + 8, s1);
        _mm256_storeu_ps(out + 16, s2);
        _mm256_maskstore_ps(out + 24, mask, s3);
        return;
    }
    case 5: {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        __m256 s4 = _mm256_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m256 a = _mm256_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 0), s0);
            s1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 8), s1);
            s2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 16), s2);
            s3 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 24), s3);
            s4 = _mm256_fmadd_ps(a, _mm256_maskload_ps(x + 32, mask), s4);
        }
        _mm256_storeu_ps(out + 0, s0);
        _mm256_storeu_ps(out + 8, s1);
        _mm256_storeu_ps(out + 16, s2);
        _mm256_storeu_ps(out + 24, s3);
        _mm256_maskstore_ps(out + 32, mask, s4);
        return;
    }
    case 6: {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        __m256 s4 = _mm256_setzero_ps();
        __m256 s5 = _mm256_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m256 a = _mm256_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 0), s0);
            s1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 8), s1);
            s2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 16), s2);
            s3 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 24), s3);
            s4 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 32), s4);
            s5 = _mm256_fmadd_ps(a, _mm256_maskload_ps(x + 40, mask), s5);
        }
        _mm256_storeu_ps(out + 0, s0);
        _mm256_storeu_ps(out + 8, s1);
        _mm256_storeu_ps(out + 16, s2);
        _mm256_storeu_ps(out + 24, s3);
        _mm256_storeu_ps(out + 32, s4);
        _mm256_maskstore_ps(out + 40, mask, s5);
        return;
    }
    case 7: {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        __m256 s4 = _mm256_setzero_ps();
        __m256 s5 = _mm256_setzero_ps();
        __m256 s6 = _mm256_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m256 a = _mm256_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 0), s0);
            s1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 8), s1);
            s2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 16), s2);
            s3 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 24), s3);
            s4 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 32), s4);
            s5 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 40), s5);
            s6 = _mm256_fmadd_ps(a, _mm256_maskload_ps(x + 48, mask), s6);
        }
        _mm256_storeu_ps(out + 0, s0);
        _mm256_storeu_ps(out + 8, s1);
        _mm256_storeu_ps(out + 16, s2);
        _mm256_storeu_ps(out + 24, s3);
        _mm256_storeu_ps(out + 32, s4);
        _mm256_storeu_ps(out + 40, s5);
        _mm256_maskstore_ps(out + 48, mask, s6);
        return;
    }
    case 8: {
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();
        __m256 s4 = _mm256_setzero_ps();
        __m256 s5 = _mm256_setzero_ps();
        __m256 s6 = _mm256_setzero_ps();
        __m256 s7 = _mm256_setzero_ps();
        for (size_t p = begin; p < end; ++p) {
            const __m256 a = _mm256_set1_ps(data[p]);
            const float *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 0), s0);
            s1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 8), s1);
            s2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 16), s2);
            s3 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 24), s3);
            s4 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 32), s4);
            s5 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 40), s5);
            s6 = _mm256_fmadd_ps(a, _mm256_loadu_ps(x + 48), s6);
            s7 = _mm256_fmadd_ps(a, _mm256_maskload_ps(x + 56, mask), s7);
        }
        _mm256_storeu_ps(out + 0, s0);
        _mm256_storeu_ps(out + 8, s1);
        _mm256_storeu_ps(out + 16, s2);
        _mm256_storeu_ps(out + 24, s3);
        _mm256_storeu_ps(out + 32, s4);
        _mm256_storeu_ps(out + 40, s5);
        _mm256_storeu_ps(out + 48, s6);
        _mm256_maskstore_ps(out + 56, mask, s7);
        return;
    }
    }
}
MYMF_AVX2 static void reg_f64_i32_avx2(const double * MYMF_RESTRICT data,
    const int32_t * MYMF_RESTRICT indices, size_t begin, size_t end,
    const double * MYMF_RESTRICT b, double * MYMF_RESTRICT out, size_t k) {
    if (k > 32) { row_f64_i32_avx2(data, indices, begin, end, b, out, k); return; }
    const size_t last = (k - 1) % 4 + 1;
    const __m256i mask = _mm256_setr_epi64x(last > 0 ? -1 : 0, last > 1 ? -1 : 0, last > 2 ? -1 : 0, last > 3 ? -1 : 0);
    switch ((k + 3) / 4) {
    case 1: {
        __m256d s0 = _mm256_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m256d a = _mm256_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_pd(a, _mm256_maskload_pd(x + 0, mask), s0);
        }
        _mm256_maskstore_pd(out + 0, mask, s0);
        return;
    }
    case 2: {
        __m256d s0 = _mm256_setzero_pd();
        __m256d s1 = _mm256_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m256d a = _mm256_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 0), s0);
            s1 = _mm256_fmadd_pd(a, _mm256_maskload_pd(x + 4, mask), s1);
        }
        _mm256_storeu_pd(out + 0, s0);
        _mm256_maskstore_pd(out + 4, mask, s1);
        return;
    }
    case 3: {
        __m256d s0 = _mm256_setzero_pd();
        __m256d s1 = _mm256_setzero_pd();
        __m256d s2 = _mm256_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m256d a = _mm256_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 0), s0);
            s1 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 4), s1);
            s2 = _mm256_fmadd_pd(a, _mm256_maskload_pd(x + 8, mask), s2);
        }
        _mm256_storeu_pd(out + 0, s0);
        _mm256_storeu_pd(out + 4, s1);
        _mm256_maskstore_pd(out + 8, mask, s2);
        return;
    }
    case 4: {
        __m256d s0 = _mm256_setzero_pd();
        __m256d s1 = _mm256_setzero_pd();
        __m256d s2 = _mm256_setzero_pd();
        __m256d s3 = _mm256_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m256d a = _mm256_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 0), s0);
            s1 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 4), s1);
            s2 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 8), s2);
            s3 = _mm256_fmadd_pd(a, _mm256_maskload_pd(x + 12, mask), s3);
        }
        _mm256_storeu_pd(out + 0, s0);
        _mm256_storeu_pd(out + 4, s1);
        _mm256_storeu_pd(out + 8, s2);
        _mm256_maskstore_pd(out + 12, mask, s3);
        return;
    }
    case 5: {
        __m256d s0 = _mm256_setzero_pd();
        __m256d s1 = _mm256_setzero_pd();
        __m256d s2 = _mm256_setzero_pd();
        __m256d s3 = _mm256_setzero_pd();
        __m256d s4 = _mm256_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m256d a = _mm256_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 0), s0);
            s1 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 4), s1);
            s2 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 8), s2);
            s3 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 12), s3);
            s4 = _mm256_fmadd_pd(a, _mm256_maskload_pd(x + 16, mask), s4);
        }
        _mm256_storeu_pd(out + 0, s0);
        _mm256_storeu_pd(out + 4, s1);
        _mm256_storeu_pd(out + 8, s2);
        _mm256_storeu_pd(out + 12, s3);
        _mm256_maskstore_pd(out + 16, mask, s4);
        return;
    }
    case 6: {
        __m256d s0 = _mm256_setzero_pd();
        __m256d s1 = _mm256_setzero_pd();
        __m256d s2 = _mm256_setzero_pd();
        __m256d s3 = _mm256_setzero_pd();
        __m256d s4 = _mm256_setzero_pd();
        __m256d s5 = _mm256_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m256d a = _mm256_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 0), s0);
            s1 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 4), s1);
            s2 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 8), s2);
            s3 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 12), s3);
            s4 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 16), s4);
            s5 = _mm256_fmadd_pd(a, _mm256_maskload_pd(x + 20, mask), s5);
        }
        _mm256_storeu_pd(out + 0, s0);
        _mm256_storeu_pd(out + 4, s1);
        _mm256_storeu_pd(out + 8, s2);
        _mm256_storeu_pd(out + 12, s3);
        _mm256_storeu_pd(out + 16, s4);
        _mm256_maskstore_pd(out + 20, mask, s5);
        return;
    }
    case 7: {
        __m256d s0 = _mm256_setzero_pd();
        __m256d s1 = _mm256_setzero_pd();
        __m256d s2 = _mm256_setzero_pd();
        __m256d s3 = _mm256_setzero_pd();
        __m256d s4 = _mm256_setzero_pd();
        __m256d s5 = _mm256_setzero_pd();
        __m256d s6 = _mm256_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m256d a = _mm256_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 0), s0);
            s1 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 4), s1);
            s2 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 8), s2);
            s3 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 12), s3);
            s4 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 16), s4);
            s5 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 20), s5);
            s6 = _mm256_fmadd_pd(a, _mm256_maskload_pd(x + 24, mask), s6);
        }
        _mm256_storeu_pd(out + 0, s0);
        _mm256_storeu_pd(out + 4, s1);
        _mm256_storeu_pd(out + 8, s2);
        _mm256_storeu_pd(out + 12, s3);
        _mm256_storeu_pd(out + 16, s4);
        _mm256_storeu_pd(out + 20, s5);
        _mm256_maskstore_pd(out + 24, mask, s6);
        return;
    }
    case 8: {
        __m256d s0 = _mm256_setzero_pd();
        __m256d s1 = _mm256_setzero_pd();
        __m256d s2 = _mm256_setzero_pd();
        __m256d s3 = _mm256_setzero_pd();
        __m256d s4 = _mm256_setzero_pd();
        __m256d s5 = _mm256_setzero_pd();
        __m256d s6 = _mm256_setzero_pd();
        __m256d s7 = _mm256_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m256d a = _mm256_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 0), s0);
            s1 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 4), s1);
            s2 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 8), s2);
            s3 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 12), s3);
            s4 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 16), s4);
            s5 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 20), s5);
            s6 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 24), s6);
            s7 = _mm256_fmadd_pd(a, _mm256_maskload_pd(x + 28, mask), s7);
        }
        _mm256_storeu_pd(out + 0, s0);
        _mm256_storeu_pd(out + 4, s1);
        _mm256_storeu_pd(out + 8, s2);
        _mm256_storeu_pd(out + 12, s3);
        _mm256_storeu_pd(out + 16, s4);
        _mm256_storeu_pd(out + 20, s5);
        _mm256_storeu_pd(out + 24, s6);
        _mm256_maskstore_pd(out + 28, mask, s7);
        return;
    }
    }
}
MYMF_AVX2 static void reg_f64_i64_avx2(const double * MYMF_RESTRICT data,
    const int64_t * MYMF_RESTRICT indices, size_t begin, size_t end,
    const double * MYMF_RESTRICT b, double * MYMF_RESTRICT out, size_t k) {
    if (k > 32) { row_f64_i64_avx2(data, indices, begin, end, b, out, k); return; }
    const size_t last = (k - 1) % 4 + 1;
    const __m256i mask = _mm256_setr_epi64x(last > 0 ? -1 : 0, last > 1 ? -1 : 0, last > 2 ? -1 : 0, last > 3 ? -1 : 0);
    switch ((k + 3) / 4) {
    case 1: {
        __m256d s0 = _mm256_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m256d a = _mm256_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_pd(a, _mm256_maskload_pd(x + 0, mask), s0);
        }
        _mm256_maskstore_pd(out + 0, mask, s0);
        return;
    }
    case 2: {
        __m256d s0 = _mm256_setzero_pd();
        __m256d s1 = _mm256_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m256d a = _mm256_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 0), s0);
            s1 = _mm256_fmadd_pd(a, _mm256_maskload_pd(x + 4, mask), s1);
        }
        _mm256_storeu_pd(out + 0, s0);
        _mm256_maskstore_pd(out + 4, mask, s1);
        return;
    }
    case 3: {
        __m256d s0 = _mm256_setzero_pd();
        __m256d s1 = _mm256_setzero_pd();
        __m256d s2 = _mm256_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m256d a = _mm256_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 0), s0);
            s1 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 4), s1);
            s2 = _mm256_fmadd_pd(a, _mm256_maskload_pd(x + 8, mask), s2);
        }
        _mm256_storeu_pd(out + 0, s0);
        _mm256_storeu_pd(out + 4, s1);
        _mm256_maskstore_pd(out + 8, mask, s2);
        return;
    }
    case 4: {
        __m256d s0 = _mm256_setzero_pd();
        __m256d s1 = _mm256_setzero_pd();
        __m256d s2 = _mm256_setzero_pd();
        __m256d s3 = _mm256_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m256d a = _mm256_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 0), s0);
            s1 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 4), s1);
            s2 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 8), s2);
            s3 = _mm256_fmadd_pd(a, _mm256_maskload_pd(x + 12, mask), s3);
        }
        _mm256_storeu_pd(out + 0, s0);
        _mm256_storeu_pd(out + 4, s1);
        _mm256_storeu_pd(out + 8, s2);
        _mm256_maskstore_pd(out + 12, mask, s3);
        return;
    }
    case 5: {
        __m256d s0 = _mm256_setzero_pd();
        __m256d s1 = _mm256_setzero_pd();
        __m256d s2 = _mm256_setzero_pd();
        __m256d s3 = _mm256_setzero_pd();
        __m256d s4 = _mm256_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m256d a = _mm256_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 0), s0);
            s1 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 4), s1);
            s2 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 8), s2);
            s3 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 12), s3);
            s4 = _mm256_fmadd_pd(a, _mm256_maskload_pd(x + 16, mask), s4);
        }
        _mm256_storeu_pd(out + 0, s0);
        _mm256_storeu_pd(out + 4, s1);
        _mm256_storeu_pd(out + 8, s2);
        _mm256_storeu_pd(out + 12, s3);
        _mm256_maskstore_pd(out + 16, mask, s4);
        return;
    }
    case 6: {
        __m256d s0 = _mm256_setzero_pd();
        __m256d s1 = _mm256_setzero_pd();
        __m256d s2 = _mm256_setzero_pd();
        __m256d s3 = _mm256_setzero_pd();
        __m256d s4 = _mm256_setzero_pd();
        __m256d s5 = _mm256_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m256d a = _mm256_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 0), s0);
            s1 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 4), s1);
            s2 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 8), s2);
            s3 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 12), s3);
            s4 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 16), s4);
            s5 = _mm256_fmadd_pd(a, _mm256_maskload_pd(x + 20, mask), s5);
        }
        _mm256_storeu_pd(out + 0, s0);
        _mm256_storeu_pd(out + 4, s1);
        _mm256_storeu_pd(out + 8, s2);
        _mm256_storeu_pd(out + 12, s3);
        _mm256_storeu_pd(out + 16, s4);
        _mm256_maskstore_pd(out + 20, mask, s5);
        return;
    }
    case 7: {
        __m256d s0 = _mm256_setzero_pd();
        __m256d s1 = _mm256_setzero_pd();
        __m256d s2 = _mm256_setzero_pd();
        __m256d s3 = _mm256_setzero_pd();
        __m256d s4 = _mm256_setzero_pd();
        __m256d s5 = _mm256_setzero_pd();
        __m256d s6 = _mm256_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m256d a = _mm256_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 0), s0);
            s1 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 4), s1);
            s2 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 8), s2);
            s3 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 12), s3);
            s4 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 16), s4);
            s5 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 20), s5);
            s6 = _mm256_fmadd_pd(a, _mm256_maskload_pd(x + 24, mask), s6);
        }
        _mm256_storeu_pd(out + 0, s0);
        _mm256_storeu_pd(out + 4, s1);
        _mm256_storeu_pd(out + 8, s2);
        _mm256_storeu_pd(out + 12, s3);
        _mm256_storeu_pd(out + 16, s4);
        _mm256_storeu_pd(out + 20, s5);
        _mm256_maskstore_pd(out + 24, mask, s6);
        return;
    }
    case 8: {
        __m256d s0 = _mm256_setzero_pd();
        __m256d s1 = _mm256_setzero_pd();
        __m256d s2 = _mm256_setzero_pd();
        __m256d s3 = _mm256_setzero_pd();
        __m256d s4 = _mm256_setzero_pd();
        __m256d s5 = _mm256_setzero_pd();
        __m256d s6 = _mm256_setzero_pd();
        __m256d s7 = _mm256_setzero_pd();
        for (size_t p = begin; p < end; ++p) {
            const __m256d a = _mm256_set1_pd(data[p]);
            const double *x = b + (size_t)indices[p] * k;
            s0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 0), s0);
            s1 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 4), s1);
            s2 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 8), s2);
            s3 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 12), s3);
            s4 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 16), s4);
            s5 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 20), s5);
            s6 = _mm256_fmadd_pd(a, _mm256_loadu_pd(x + 24), s6);
            s7 = _mm256_fmadd_pd(a, _mm256_maskload_pd(x + 28, mask), s7);
        }
        _mm256_storeu_pd(out + 0, s0);
        _mm256_storeu_pd(out + 4, s1);
        _mm256_storeu_pd(out + 8, s2);
        _mm256_storeu_pd(out + 12, s3);
        _mm256_storeu_pd(out + 16, s4);
        _mm256_storeu_pd(out + 20, s5);
        _mm256_storeu_pd(out + 24, s6);
        _mm256_maskstore_pd(out + 28, mask, s7);
        return;
    }
    }
}
#endif
#endif
