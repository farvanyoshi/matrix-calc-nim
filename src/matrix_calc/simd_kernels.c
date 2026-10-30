/* Runtime-dispatched double-precision kernels. The Nim wrapper validates lengths
 * and non-null pointers for n>0. Architecture routines use full lanes plus scalar
 * tails, unaligned loads, and no access outside the n-element buffers. */
#include <stddef.h>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define MC_AVX 1
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#define MC_NEON 1
#endif

int matrix_calc_simd_backend(void) {
#if defined(MC_DISABLE_SIMD)
    return 0;
#endif
#if defined(MC_AVX)
    if (__builtin_cpu_supports("avx")) return 1;
#endif
#if defined(MC_NEON)
#if defined(__linux__)
    if ((getauxval(AT_HWCAP) & HWCAP_ASIMD) != 0) return 2;
#else
    return 2; /* Advanced SIMD is part of the AArch64 platform ABI. */
#endif
#endif
    return 0;
}
static double mc_dot_scalar(const double *a,const double *b,size_t n) {
    double result=0.0; for(size_t i=0;i<n;++i) result+=a[i]*b[i];return result;
}
static void mc_axpy_scalar(double a,const double*x,double*y,size_t n) {
    for(size_t i=0;i<n;++i)y[i]+=a*x[i];
}
#if defined(MC_AVX)
__attribute__((target("avx")))
static double mc_dot_avx(const double*a,const double*b,size_t n) {
    __m256d sum=_mm256_setzero_pd();size_t i=0;
    for(;i+4<=n;i+=4) sum=_mm256_add_pd(sum,_mm256_mul_pd(_mm256_loadu_pd(a+i),_mm256_loadu_pd(b+i)));
    double lanes[4];_mm256_storeu_pd(lanes,sum);
    double result=lanes[0]+lanes[1]+lanes[2]+lanes[3];
    for(;i<n;++i)result+=a[i]*b[i];return result;
}
__attribute__((target("avx")))
static void mc_axpy_avx(double a,const double*x,double*y,size_t n) {
    __m256d scale=_mm256_set1_pd(a);size_t i=0;
    for(;i+4<=n;i+=4)_mm256_storeu_pd(y+i,_mm256_add_pd(_mm256_loadu_pd(y+i),_mm256_mul_pd(scale,_mm256_loadu_pd(x+i))));
    for(;i<n;++i)y[i]+=a*x[i];
}
#endif
#if defined(MC_NEON)
static double mc_dot_neon(const double*a,const double*b,size_t n) {
    float64x2_t sum=vdupq_n_f64(0.0);size_t i=0;
    for(;i+2<=n;i+=2)sum=vaddq_f64(sum,vmulq_f64(vld1q_f64(a+i),vld1q_f64(b+i)));
    double result=vaddvq_f64(sum);for(;i<n;++i)result+=a[i]*b[i];return result;
}
static void mc_axpy_neon(double a,const double*x,double*y,size_t n) {
    float64x2_t scale=vdupq_n_f64(a);size_t i=0;
    for(;i+2<=n;i+=2)vst1q_f64(y+i,vaddq_f64(vld1q_f64(y+i),vmulq_f64(scale,vld1q_f64(x+i))));
    for(;i<n;++i)y[i]+=a*x[i];
}
#endif

double matrix_calc_dot_simd(const double*a,const double*b,size_t n) {
    int backend=matrix_calc_simd_backend();
#if defined(MC_AVX)
    if(backend==1)return mc_dot_avx(a,b,n);
#endif
#if defined(MC_NEON)
    if(backend==2)return mc_dot_neon(a,b,n);
#endif
    return mc_dot_scalar(a,b,n);
}
void matrix_calc_axpy_simd(double a,const double*x,double*y,size_t n) {
    int backend=matrix_calc_simd_backend();
#if defined(MC_AVX)
    if(backend==1){mc_axpy_avx(a,x,y,n);return;}
#endif
#if defined(MC_NEON)
    if(backend==2){mc_axpy_neon(a,x,y,n);return;}
#endif
    mc_axpy_scalar(a,x,y,n);
}
