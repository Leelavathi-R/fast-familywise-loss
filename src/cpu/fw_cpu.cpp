// Familywise loss on CPU: exact, worst-case O(K) per sample, sparse backward.
//
//   L(z, y) = 1 - z_y + tau*,   tau* = root of sum_j max(z_j - t, 0) = 1,
//   grad    = (1/kappa) 1[z > tau*] - e_y,   kappa = |{z > tau*}|.
//
// Per sample: one streaming pass (Condat's algorithm, guarded, with index tracking).  Blocks of 64
// logits with no entry above the running lower bound rho are skipped by a vectorized comparison.
// The gradient buffer is zero-initialised lazily by the OS (mmap) and written only at the kappa
// support indices and at y.  Flat rows use vectorized Newton passes first.  Any floating-point
// breakdown is caught and the row is recomputed by a shifted median-of-medians search.
#include <torch/extension.h>
#include <immintrin.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <sys/mman.h>
#endif

namespace {


// ---------------------------------------------------------------- thread-local scratch (untouched until used)
struct Buf {
  void* p = nullptr; size_t cap = 0;
  void* get(size_t bytes) {
    if (bytes > cap) { std::free(p); p = std::malloc(bytes); cap = bytes; }
    return p;
  }
  ~Buf() { std::free(p); }
};
struct Scratch { Buf v, vt, vi, vti, w, wi, buf, bv, bi; };
thread_local Scratch tls;

// ---------------------------------------------------------------- BFPRT selection + median-pivot search
void insertion_sort(double* a, int64_t lo, int64_t hi) {
  for (int64_t i = lo + 1; i < hi; ++i) {
    double x = a[i]; int64_t j = i - 1;
    while (j >= lo && a[j] > x) { a[j + 1] = a[j]; --j; }
    a[j + 1] = x;
  }
}
void partition3(double* a, int64_t lo, int64_t hi, double p, int64_t& lt, int64_t& gt) {
  lt = lo; gt = hi; int64_t i = lo;
  while (i < gt) {
    double x = a[i];
    if (x < p) { std::swap(a[lt], a[i]); ++lt; ++i; }
    else if (x > p) { --gt; std::swap(a[gt], a[i]); }
    else ++i;
  }
}
// value of the k-th smallest (0-based from lo) of a[lo:hi); worst-case linear (median of medians)
double select_k(double* a, int64_t lo, int64_t hi, int64_t k) {
  for (;;) {
    int64_t n = hi - lo;
    if (n <= 10) { insertion_sort(a, lo, hi); return a[lo + k]; }
    int64_t ng = 0;
    for (int64_t g = lo; g < hi; g += 5) {
      int64_t e = g + 5 < hi ? g + 5 : hi;
      insertion_sort(a, g, e);
      int64_t m = g + (e - g - 1) / 2;
      std::swap(a[lo + ng], a[m]); ++ng;
    }
    double p = select_k(a, lo, lo + ng, (ng - 1) / 2);
    int64_t lt, gt; partition3(a, lo, hi, p, lt, gt);
    if (k < lt - lo) hi = lt;
    else if (k < gt - lo) return p;
    else { k -= gt - lo; lo = gt; }
  }
}
// median-pivot search on u[0:m).  Returns tau, c; max_out = largest discarded value (so that
// {u > max_out} is exactly the set classified "above", whatever the rounding).
void pivot_search(double* u, int64_t m, double& tau, double& max_out, int64_t& c) {
  int64_t lo = 0, hi = m; double s_known = 0.0; int64_t c_known = 0;
  max_out = -std::numeric_limits<double>::infinity();
  while (hi > lo) {
    int64_t n = hi - lo;
    double p = select_k(u, lo, hi, (n - 1) / 2);
    int64_t lt, gt; partition3(u, lo, hi, p, lt, gt);
    double sG = 0.0; for (int64_t i = gt; i < hi; ++i) sG += u[i];
    int64_t cG = hi - gt;
    if (s_known + sG - double(c_known + cG) * p < 1.0) {
      s_known += sG + double(gt - lt) * p; c_known += cG + (gt - lt); hi = lt;
    } else { if (p > max_out) max_out = p; lo = gt; }
  }
  tau = (s_known - 1.0) / double(c_known); c = c_known;
}

// ---------------------------------------------------------------- helpers
inline float round_down(double t, float) {            // largest float <= t:  x > t  <=>  x > round_down(t)
  float r = float(t);
  if (double(r) > t) r = std::nextafterf(r, -std::numeric_limits<float>::infinity());
  return r;
}
inline double round_down(double t, double) { return t; }

// Candidate bitmask of a[0:W) against r (W = lanes).  Bit j set  <=>  a[j] > r.
#if defined(__AVX512F__)
const int64_t LANES_F = 16;
inline uint32_t cand_mask(const float* a, float r) {
  return (uint32_t)_mm512_cmp_ps_mask(_mm512_loadu_ps(a), _mm512_set1_ps(r), _CMP_GT_OQ);
}
const int64_t LANES_D = 8;
inline uint32_t cand_mask(const double* a, double r) {
  return (uint32_t)_mm512_cmp_pd_mask(_mm512_loadu_pd(a), _mm512_set1_pd(r), _CMP_GT_OQ);
}
#elif defined(__AVX2__)
const int64_t LANES_F = 8;
inline uint32_t cand_mask(const float* a, float r) {
  return (uint32_t)_mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(a), _mm256_set1_ps(r), _CMP_GT_OQ));
}
const int64_t LANES_D = 4;
inline uint32_t cand_mask(const double* a, double r) {
  return (uint32_t)_mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(a), _mm256_set1_pd(r), _CMP_GT_OQ));
}
#else
const int64_t LANES_F = 16;
inline uint32_t cand_mask(const float* a, float r) {
  uint32_t m = 0; for (int j = 0; j < 16; ++j) m |= uint32_t(a[j] > r) << j; return m;
}
const int64_t LANES_D = 8;
inline uint32_t cand_mask(const double* a, double r) {
  uint32_t m = 0; for (int j = 0; j < 8; ++j) m |= uint32_t(a[j] > r) << j; return m;
}
#endif
inline int64_t lanes(float) { return LANES_F; }
inline int64_t lanes(double) { return LANES_D; }

// Guarded Condat with index tracking and in-scan pruning.
// Invariants (exact arithmetic and floating point): rho is non-decreasing; every entry that is
// neither in v nor in vt has been compared with a value <= the final rho and found <= it.
// status 0: support = vi[0:c) (= {a > thr});  1: finished by pivot fallback, support = {a > thr};
// 2: floating-point breakdown.
template <class T>
int condat_idx(const T* a, int64_t n, double* v, int64_t* vi, double* vt, int64_t* vti, int64_t cap,
               double& tau, double& thr, int64_t& c) {
  int64_t nv = 1, nvt = 0, next_prune = 34;
  v[0] = double(a[0]); vi[0] = 0;
  double rho = double(a[0]) - 1.0;
  T r = round_down(rho, T(0));
  auto step = [&](int64_t j) {
    double yn = double(a[j]);
    if (yn > rho) {
      rho += (yn - rho) / double(nv + 1);
      if (rho > yn - 1.0) { v[nv] = yn; vi[nv] = j; ++nv; }
      else {
        for (int64_t q = 0; q < nv; ++q) { vt[nvt + q] = v[q]; vti[nvt + q] = vi[q]; }
        nvt += nv; v[0] = yn; vi[0] = j; nv = 1; rho = yn - 1.0; next_prune = 34;
      }
    }
  };
  auto prune = [&]() {                    // one sweep: drop entries <= rho (each drop raises rho)
    int64_t k = 0;
    while (k < nv) {
      double y = v[k];
      if (y <= rho && nv > 1) { --nv; v[k] = v[nv]; vi[k] = vi[nv]; rho += (rho - y) / double(nv); }
      else ++k;
    }
    next_prune = 2 * nv + 32;
  };
  const int64_t W = lanes(T(0));
  int64_t i = 1;
  for (; i + W <= n; i += W) {
    uint32_t m = cand_mask(a + i, r);
    if (m) {
      do { step(i + __builtin_ctz(m)); m &= m - 1; } while (m);
      if (nv >= next_prune) prune();
      r = round_down(rho, T(0));
    }
  }
  for (; i < n; ++i) step(i);
  for (int64_t j = 0; j < nvt; ++j) {
    double y = vt[j];
    if (y > rho) { v[nv] = y; vi[nv] = vti[j]; ++nv; rho += (y - rho) / double(nv); }
  }
  int64_t work = 0; bool changed = true;
  while (changed) {
    changed = false; work += nv;
    if (work > cap) { double mo; pivot_search(v, nv, tau, mo, c); thr = tau; return 1; }
    int64_t k = 0;
    while (k < nv) {
      double y = v[k];
      if (y <= rho) {
        if (nv == 1) return 2;
        --nv; v[k] = v[nv]; vi[k] = vi[nv];
        rho += (rho - y) / double(nv); changed = true;
      } else ++k;
    }
  }
  double acc = 0.0; for (int64_t k = 0; k < nv; ++k) acc += v[k] - rho;
  tau = rho + (acc - 1.0) / double(nv); thr = rho; c = nv;
  return 0;
}

// =====================================================================================================
// float32 fast path: chunked, fully vectorized threshold search.
//
// State: a lower bound t <= tau* and a candidate buffer (values bv, indices bi) that contains every
// entry seen so far that is > t.  Per chunk of CHUNK logits: (1) chunk maximum m, t <- max(t, m - 1);
// (2) append the chunk entries > t (vector compress).  When the buffer has doubled, "tighten": a few
// Newton steps t <- t + (g_buf(t) - 1)/|buf > t| restricted to the buffer, dropping entries <= t.
// Every value assigned to t is H(S) for a subset S of the row, hence <= tau*; dropped entries are
// <= t.  At the end the Newton iteration is run on the buffer to its fixed point (= tau*), with a
// work cap after which the median-of-medians search finishes on the buffer.
// =====================================================================================================
const int64_t CHUNK = 2048;
int64_t g_cap_factor = 4;      // work cap of the final Newton iteration, in units of K (tests set it to 0)

#if defined(__AVX512F__)
inline float chunk_max(const float* a, int64_t n) {
  float m = a[0]; int64_t i = 0;
  if (n >= 16) {
    __m512 vm = _mm512_loadu_ps(a);
    for (i = 16; i + 16 <= n; i += 16) vm = _mm512_max_ps(vm, _mm512_loadu_ps(a + i));
    m = _mm512_reduce_max_ps(vm);
  }
  for (; i < n; ++i) if (a[i] > m) m = a[i];
  return m;
}
// append the entries of a[0:n) that are > r (their indices are base + position) at position m
inline int64_t compress_gt(const float* a, int64_t n, int32_t base, float r, float* bv, int32_t* bi, int64_t m) {
  const __m512 vr = _mm512_set1_ps(r);
  __m512i idx = _mm512_add_epi32(_mm512_set1_epi32(base), _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15));
  const __m512i inc = _mm512_set1_epi32(16);
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 x = _mm512_loadu_ps(a + i);
    __mmask16 k = _mm512_cmp_ps_mask(x, vr, _CMP_GT_OQ);
    if (k) {
      _mm512_storeu_ps(bv + m, _mm512_maskz_compress_ps(k, x));
      _mm512_storeu_si512((void*)(bi + m), _mm512_maskz_compress_epi32(k, idx));
      m += __builtin_popcount((unsigned)k);
    }
    idx = _mm512_add_epi32(idx, inc);
  }
  for (; i < n; ++i) if (a[i] > r) { bv[m] = a[i]; bi[m] = base + int32_t(i); ++m; }
  return m;
}
// in place: keep the buffer entries > t; s = sum over the kept entries of (x - t).  Returns the new size.
inline int64_t filter_sum(float* bv, int32_t* bi, int64_t n, double t, double& s) {
  const float r = round_down(t, 0.0f);
  const __m512 vr = _mm512_set1_ps(r); const __m512d vt = _mm512_set1_pd(t);
  __m512d acc0 = _mm512_setzero_pd(), acc1 = _mm512_setzero_pd();
  int64_t i = 0, m = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 x = _mm512_loadu_ps(bv + i); __m512i id = _mm512_loadu_si512((const void*)(bi + i));
    __mmask16 k = _mm512_cmp_ps_mask(x, vr, _CMP_GT_OQ);
    __m512d lo = _mm512_sub_pd(_mm512_cvtps_pd(_mm512_castps512_ps256(x)), vt);
    __m512d hi = _mm512_sub_pd(_mm512_cvtps_pd(_mm512_extractf32x8_ps(x, 1)), vt);
    acc0 = _mm512_mask_add_pd(acc0, (__mmask8)(k & 0xFF), acc0, lo);
    acc1 = _mm512_mask_add_pd(acc1, (__mmask8)(k >> 8), acc1, hi);
    _mm512_storeu_ps(bv + m, _mm512_maskz_compress_ps(k, x));
    _mm512_storeu_si512((void*)(bi + m), _mm512_maskz_compress_epi32(k, id));
    m += __builtin_popcount((unsigned)k);
  }
  double ss = _mm512_reduce_add_pd(_mm512_add_pd(acc0, acc1));
  for (; i < n; ++i) { float x = bv[i]; int32_t id = bi[i]; if (x > r) { ss += double(x) - t; bv[m] = x; bi[m] = id; ++m; } }
  s = ss; return m;
}
#elif defined(__AVX2__)
struct Lut8 { alignas(32) int32_t p[256][8]; Lut8() { for (int k = 0; k < 256; ++k) { int j = 0;
  for (int b = 0; b < 8; ++b) if (k >> b & 1) p[k][j++] = b; for (; j < 8; ++j) p[k][j] = 0; } } };
static const Lut8 lut8;
inline float chunk_max(const float* a, int64_t n) {
  float m = a[0]; int64_t i = 0;
  if (n >= 8) {
    __m256 vm = _mm256_loadu_ps(a);
    for (i = 8; i + 8 <= n; i += 8) vm = _mm256_max_ps(vm, _mm256_loadu_ps(a + i));
    alignas(32) float tmp[8]; _mm256_store_ps(tmp, vm);
    m = tmp[0]; for (int j = 1; j < 8; ++j) if (tmp[j] > m) m = tmp[j];
  }
  for (; i < n; ++i) if (a[i] > m) m = a[i];
  return m;
}
inline int64_t compress_gt(const float* a, int64_t n, int32_t base, float r, float* bv, int32_t* bi, int64_t m) {
  const __m256 vr = _mm256_set1_ps(r);
  __m256i idx = _mm256_add_epi32(_mm256_set1_epi32(base), _mm256_setr_epi32(0,1,2,3,4,5,6,7));
  const __m256i inc = _mm256_set1_epi32(8);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 x = _mm256_loadu_ps(a + i);
    int k = _mm256_movemask_ps(_mm256_cmp_ps(x, vr, _CMP_GT_OQ));
    if (k) {
      __m256i p = _mm256_load_si256((const __m256i*)lut8.p[k]);
      _mm256_storeu_ps(bv + m, _mm256_permutevar8x32_ps(x, p));
      _mm256_storeu_si256((__m256i*)(bi + m), _mm256_permutevar8x32_epi32(idx, p));
      m += __builtin_popcount((unsigned)k);
    }
    idx = _mm256_add_epi32(idx, inc);
  }
  for (; i < n; ++i) if (a[i] > r) { bv[m] = a[i]; bi[m] = base + int32_t(i); ++m; }
  return m;
}
inline int64_t filter_sum(float* bv, int32_t* bi, int64_t n, double t, double& s) {
  const float r = round_down(t, 0.0f);
  const __m256 vr = _mm256_set1_ps(r); const __m256d vt = _mm256_set1_pd(t);
  __m256d acc0 = _mm256_setzero_pd(), acc1 = _mm256_setzero_pd();
  int64_t i = 0, m = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 x = _mm256_loadu_ps(bv + i); __m256i id = _mm256_loadu_si256((const __m256i*)(bi + i));
    __m256 c = _mm256_cmp_ps(x, vr, _CMP_GT_OQ);
    int k = _mm256_movemask_ps(c);
    __m256d lo = _mm256_sub_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(x)), vt);
    __m256d hi = _mm256_sub_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(x, 1)), vt);
    __m256d mlo = _mm256_castsi256_pd(_mm256_cvtepi32_epi64(_mm_castps_si128(_mm256_castps256_ps128(c))));
    __m256d mhi = _mm256_castsi256_pd(_mm256_cvtepi32_epi64(_mm_castps_si128(_mm256_extractf128_ps(c, 1))));
    acc0 = _mm256_add_pd(acc0, _mm256_and_pd(lo, mlo));
    acc1 = _mm256_add_pd(acc1, _mm256_and_pd(hi, mhi));
    __m256i p = _mm256_load_si256((const __m256i*)lut8.p[k]);
    _mm256_storeu_ps(bv + m, _mm256_permutevar8x32_ps(x, p));
    _mm256_storeu_si256((__m256i*)(bi + m), _mm256_permutevar8x32_epi32(id, p));
    m += __builtin_popcount((unsigned)k);
  }
  alignas(32) double tmp[4]; _mm256_store_pd(tmp, _mm256_add_pd(acc0, acc1));
  double ss = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
  for (; i < n; ++i) { float x = bv[i]; int32_t id = bi[i]; if (x > r) { ss += double(x) - t; bv[m] = x; bi[m] = id; ++m; } }
  s = ss; return m;
}
#else
inline float chunk_max(const float* a, int64_t n) { float m = a[0]; for (int64_t i = 1; i < n; ++i) if (a[i] > m) m = a[i]; return m; }
inline int64_t compress_gt(const float* a, int64_t n, int32_t base, float r, float* bv, int32_t* bi, int64_t m) {
  for (int64_t i = 0; i < n; ++i) if (a[i] > r) { bv[m] = a[i]; bi[m] = base + int32_t(i); ++m; }
  return m;
}
inline int64_t filter_sum(float* bv, int32_t* bi, int64_t n, double t, double& s) {
  const float r = round_down(t, 0.0f); double ss = 0.0; int64_t m = 0;
  for (int64_t i = 0; i < n; ++i) { float x = bv[i]; int32_t id = bi[i]; if (x > r) { ss += double(x) - t; bv[m] = x; bi[m] = id; ++m; } }
  s = ss; return m;
}
#endif

// status 0: support = bi[0:c);  2: floating-point breakdown (caller uses the robust path).
// t_init: any valid lower bound on tau* (-inf if none); entries <= t_init are never buffered.
int chunked_search(const float* z, int64_t K, float* bv, int32_t* bi, Buf& dbuf, int64_t chunk,
                   double& tau, double& thr, int64_t& c, double t_init) {
  int64_t n = 0, next = 256;
  double t = t_init, s = 0.0;
  bool fresh = false;        // true: the buffer holds exactly its entries > t and s = sum(x - t) over it
  for (int64_t off = 0; off < K; off += chunk) {
    const int64_t len = K - off < chunk ? K - off : chunk;
    const double tm = double(chunk_max(z + off, len)) - 1.0;
    if (tm > t) { t = tm; fresh = false; }
    const int64_t n0 = n;
    n = compress_gt(z + off, len, int32_t(off), round_down(t, 0.0f), bv, bi, n);
    if (n != n0) fresh = false;
    if (n >= next) {                                   // tighten: up to 4 Newton steps on the buffer
      n = filter_sum(bv, bi, n, t, s);
      if (n == 0) return 2;
      for (int it = 0; it < 4; ++it) {
        const double tn = t + (s - 1.0) / double(n);
        if (!(tn > t)) break;
        double s2; const int64_t n2 = filter_sum(bv, bi, n, tn, s2);
        if (n2 == 0) return 2;
        const bool same = n2 == n;
        t = tn; n = n2; s = s2;
        if (same) break;
      }
      next = 4 * n + 256;
      fresh = true;
    }
  }
  if (!fresh) n = filter_sum(bv, bi, n, t, s);
  if (n == 0) return 2;
  int64_t work = 0;
  for (;;) {
    const double tn = t + (s - 1.0) / double(n);
    if (!(tn > t)) { thr = t; tau = tn; c = n; return std::isfinite(tn) ? 0 : 2; }
    double s2; const int64_t n2 = filter_sum(bv, bi, n, tn, s2);
    if (n2 == 0) return 2;
    if (n2 == n) { thr = tn; tau = tn + (s2 - 1.0) / double(n2); c = n; return 0; }
    t = tn; n = n2; s = s2;
    work += n;
    if (work > g_cap_factor * K) break;
  }
  // worst-case guard: median-of-medians search on the buffer (every dropped entry is <= t <= tau*)
  double* u = (double*)dbuf.get(size_t(n) * sizeof(double));
  for (int64_t i = 0; i < n; ++i) u[i] = double(bv[i]);
  double mo; int64_t cc; pivot_search(u, n, tau, mo, cc);
  int64_t m = 0;
  for (int64_t i = 0; i < n; ++i) if (double(bv[i]) > mo) { bi[m] = bi[i]; ++m; }
  if (m != cc || cc < 1) return 2;
  thr = tau; c = cc;
  return 0;
}

template <class T>
inline void masked_sum_count(const T* z, int64_t K, double t, double& s, int64_t& c) {
  double ss = 0.0; int64_t cc = 0;
  #pragma omp simd reduction(+ : ss, cc)
  for (int64_t i = 0; i < K; ++i) {
    double d = double(z[i]) - t;
    bool m = d > 0.0;
    ss += m ? d : 0.0; cc += m ? 1 : 0;
  }
  s = ss; c = cc;
}

// Speed heuristic only (never affects correctness).  Strided sample of ~ns entries; "flat" if the
// logits are so close together that a large fraction of them is above tau*:
//   K * mean(sample max - z) < 8.
template <class T>
bool is_flat(const T* z, int64_t K, int64_t ns) {
  int64_t step = K / ns > 1 ? K / ns : 1;
  double m = double(z[0]);
  for (int64_t i = 0; i < K; i += step) if (double(z[i]) > m) m = double(z[i]);
  double gap = 0.0; int64_t tot = 0;
  for (int64_t i = 0; i < K; i += step) { gap += m - double(z[i]); ++tot; }
  return double(K) * gap < 8.0 * double(tot);
}

// one sample; g (length K) must be all zeros on entry.  Returns loss; writes kappa.
template <class T>
double fw_row(const T* z, int64_t K, int64_t yr, T* g, double scale, int64_t flat_kmax, int64_t& kappa) {
  Scratch& S = tls;
  double* v = (double*)S.v.get(K * sizeof(double));
  double* vt = (double*)S.vt.get(K * sizeof(double));
  int64_t* vi = (int64_t*)S.vi.get(K * sizeof(int64_t));
  int64_t* vti = (int64_t*)S.vti.get(K * sizeof(int64_t));
  double tau = NAN, thr = NAN; int64_t c = 0; int status = 0; bool mapped = false;
  int64_t* wi = nullptr;
  if (K >= 64 && K <= flat_kmax && is_flat(z, K, 128)) {
    double zmax = double(z[0]), tot = 0.0;
    for (int64_t i = 0; i < K; ++i) { double x = double(z[i]); tot += x; if (x > zmax) zmax = x; }
    double t = zmax - 1.0, t2 = (tot - 1.0) / double(K);
    if (t2 > t) t = t2;
    double s; masked_sum_count(z, K, t, s, c);
    bool done = false;
    if (c == 0) { status = 2; done = true; }
    for (int p = 0; !done && p < 12 && double(c) >= 0.03 * double(K); ++p) {
      double tn = t + (s - 1.0) / double(c), s2; int64_t c2;
      masked_sum_count(z, K, tn, s2, c2);
      if (c2 == c) { tau = tn + (s2 - 1.0) / double(c2); thr = tn; status = 1; done = true; }
      else if (c2 == 0) { status = 2; done = true; }
      else { t = tn; s = s2; c = c2; }
    }
    if (!done) {
      double* w = (double*)S.w.get(K * sizeof(double));
      wi = (int64_t*)S.wi.get(K * sizeof(int64_t));
      int64_t m = 0;
      for (int64_t i = 0; i < K; ++i) { double x = double(z[i]); if (x > t) { w[m] = x; wi[m] = i; ++m; } }
      if (m == 0) status = 2;
      else { status = condat_idx(w, m, v, vi, vt, vti, 4 * K, tau, thr, c); mapped = true; }
    }
  } else {
    status = condat_idx(z, K, v, vi, vt, vti, 4 * K, tau, thr, c);
  }
  const double zy = double(z[yr]);
  bool ok = status != 2 && c >= 1 && std::isfinite(tau) && std::isfinite(thr);
  if (ok && status == 0) {
    const T inv = T(scale / double(c));
    if (mapped) for (int64_t q = 0; q < c; ++q) g[wi[vi[q]]] = inv;
    else        for (int64_t q = 0; q < c; ++q) g[vi[q]] = inv;
  } else if (ok) {                                   // dense mask by threshold, with a count check
    const T inv = T(scale / double(c)); int64_t cnt = 0;
    for (int64_t j = 0; j < K; ++j) if (double(z[j]) > thr) { g[j] = inv; ++cnt; }
    if (cnt != c) { ok = false; for (int64_t j = 0; j < K; ++j) g[j] = T(0); }
  }
  double loss;
  if (ok) loss = 1.0 + (thr - zy) + (tau - thr);
  else {                                             // robust path: shifted coordinates, median-pivot search
    double* buf = (double*)S.buf.get(K * sizeof(double));
    double zm = double(z[0]);
    for (int64_t i = 0; i < K; ++i) if (double(z[i]) > zm) zm = double(z[i]);
    for (int64_t i = 0; i < K; ++i) buf[i] = double(z[i]) - zm;
    double t, mo; pivot_search(buf, K, t, mo, c);
    const T inv = T(scale / double(c));
    for (int64_t j = 0; j < K; ++j) g[j] = (double(z[j]) - zm) > mo ? inv : T(0);
    loss = (1.0 + t) + (zm - zy);
  }
  g[yr] -= T(scale);
  kappa = c;
  return loss;
}

int64_t g_chunk = CHUNK;

// float32 fast path, split in two so that the gradient buffer can be allocated after the first row:
// solve leaves the support indices in the thread-local buffer, write scatters them into g (all zeros).
// ---- dense path (speed heuristic + exact solve).  Used when a large fraction of the row is above tau*
// (logits very close together): then an index buffer and a scatter are slower than whole-row passes.
// (1) tau is estimated on a strided sample of 128 entries (weights K/128).  (2) Newton steps on the whole
// row, t <- H({z > t}), each one vectorized pass; H(S) <= tau* for every non-empty S, and the iteration
// stops at a fixed point of t -> H({z > t}), which is tau*.  The sample only chooses the starting point.
// s = sum over {x > t} of (x - t) in double precision, c = |{x > t}|.  r = round_down(t): for a float x,
// x > t  <=>  x > r  <=>  double(x) - t > 0, so the count (float compare) and the sum (max(d, 0)) use the same set.
#if defined(__AVX512F__)
inline void dense_sum_count(const float* z, int64_t K, double t, float r, double& s, int64_t& c) {
  const __m512d vt = _mm512_set1_pd(t), zero = _mm512_setzero_pd(); const __m512 vr = _mm512_set1_ps(r);
  __m512d a0 = zero, a1 = zero, a2 = zero, a3 = zero; int64_t cc = 0, i = 0;
  for (; i + 32 <= K; i += 32) {
    const __m512 x = _mm512_loadu_ps(z + i), y = _mm512_loadu_ps(z + i + 16);
    a0 = _mm512_add_pd(a0, _mm512_max_pd(_mm512_sub_pd(_mm512_cvtps_pd(_mm512_castps512_ps256(x)), vt), zero));
    a1 = _mm512_add_pd(a1, _mm512_max_pd(_mm512_sub_pd(_mm512_cvtps_pd(_mm512_extractf32x8_ps(x, 1)), vt), zero));
    a2 = _mm512_add_pd(a2, _mm512_max_pd(_mm512_sub_pd(_mm512_cvtps_pd(_mm512_castps512_ps256(y)), vt), zero));
    a3 = _mm512_add_pd(a3, _mm512_max_pd(_mm512_sub_pd(_mm512_cvtps_pd(_mm512_extractf32x8_ps(y, 1)), vt), zero));
    cc += __builtin_popcount((unsigned)_mm512_cmp_ps_mask(x, vr, _CMP_GT_OQ)) + __builtin_popcount((unsigned)_mm512_cmp_ps_mask(y, vr, _CMP_GT_OQ));
  }
  double ss = _mm512_reduce_add_pd(_mm512_add_pd(_mm512_add_pd(a0, a1), _mm512_add_pd(a2, a3)));
  for (; i < K; ++i) { const double d = double(z[i]) - t; if (d > 0.0) { ss += d; ++cc; } }
  s = ss; c = cc;
}
#elif defined(__AVX2__)
inline void dense_sum_count(const float* z, int64_t K, double t, float r, double& s, int64_t& c) {
  const __m256d vt = _mm256_set1_pd(t), zero = _mm256_setzero_pd(); const __m256 vr = _mm256_set1_ps(r);
  __m256d a0 = zero, a1 = zero, a2 = zero, a3 = zero; int64_t cc = 0, i = 0;
  for (; i + 16 <= K; i += 16) {
    const __m256 x = _mm256_loadu_ps(z + i), y = _mm256_loadu_ps(z + i + 8);
    a0 = _mm256_add_pd(a0, _mm256_max_pd(_mm256_sub_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(x)), vt), zero));
    a1 = _mm256_add_pd(a1, _mm256_max_pd(_mm256_sub_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(x, 1)), vt), zero));
    a2 = _mm256_add_pd(a2, _mm256_max_pd(_mm256_sub_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(y)), vt), zero));
    a3 = _mm256_add_pd(a3, _mm256_max_pd(_mm256_sub_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(y, 1)), vt), zero));
    cc += __builtin_popcount((unsigned)_mm256_movemask_ps(_mm256_cmp_ps(x, vr, _CMP_GT_OQ))) + __builtin_popcount((unsigned)_mm256_movemask_ps(_mm256_cmp_ps(y, vr, _CMP_GT_OQ)));
  }
  alignas(32) double tmp[4]; _mm256_store_pd(tmp, _mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3)));
  double ss = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
  for (; i < K; ++i) { const double d = double(z[i]) - t; if (d > 0.0) { ss += d; ++cc; } }
  s = ss; c = cc;
}
#else
inline void dense_sum_count(const float* z, int64_t K, double t, float r, double& s, int64_t& c) {
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0; int64_t cc = 0, i = 0;
  for (; i + 4 <= K; i += 4) {
    const double d0 = double(z[i]) - t, d1 = double(z[i + 1]) - t, d2 = double(z[i + 2]) - t, d3 = double(z[i + 3]) - t;
    s0 += d0 > 0.0 ? d0 : 0.0; s1 += d1 > 0.0 ? d1 : 0.0; s2 += d2 > 0.0 ? d2 : 0.0; s3 += d3 > 0.0 ? d3 : 0.0;
    cc += (d0 > 0.0) + (d1 > 0.0) + (d2 > 0.0) + (d3 > 0.0);
  }
  double ss = (s0 + s1) + (s2 + s3);
  for (; i < K; ++i) { const double d = double(z[i]) - t; if (d > 0.0) { ss += d; ++cc; } }
  s = ss; c = cc;
}
#endif

int64_t g_dense_min_k = 64, g_small_k = 512, g_warm_max = 65536; double g_dense_frac = 0.08, g_pre_frac = 0.3;
inline bool sample_start(const float* z, int64_t K, double& t0, double& frac) {
  int64_t ns = (K / 16) & ~int64_t(15);                           // 16 ... 128 entries, a multiple of 16
  ns = ns < 16 ? 16 : (ns > 128 ? 128 : ns);
  const int64_t step = K / ns;
  alignas(64) float x[128]; float smax = z[0]; double tot = 0.0;
  for (int64_t i = 0; i < ns; ++i) { const float v = z[i * step]; x[i] = v; tot += double(v); if (v > smax) smax = v; }
  double s; int64_t n;
  const double tg = double(smax) - 1.0;
  dense_sum_count(x, ns, tg, round_down(tg, 0.0f), s, n);
  if (double(n) < g_pre_frac * double(ns)) return false;         // {z > max - 1} contains the support: sparse row
  const double invw = double(ns) / double(K);                    // each sample entry stands for K/ns logits
  double t = (tot - invw) / double(ns);                          // first step of the iteration, from the whole sample
  if (!std::isfinite(t)) return false;
  int64_t prev = -1;
  for (int it = 0; it < 16; ++it) {
    dense_sum_count(x, ns, t, round_down(t, 0.0f), s, n);
    if (n == 0) return false;
    const double tn = t + (s - invw) / double(n);
    if (!(tn > t) || n == prev) break;
    t = tn; prev = n;
  }
  dense_sum_count(x, ns, t, round_down(t, 0.0f), s, n);
  t0 = t; frac = double(n) / double(ns);
  return std::isfinite(t) && n > 0;
}
// status 3: support = {z > thr} (dense), c = its size.  -1: not handled here.
// status 3: support = {z > thr} (dense), c = its size.  -1: not solved here; t_lb is then a valid lower
// bound on tau* (= H(S) for a non-empty S, or -inf) with which the chunked search is started.
inline int dense_search(const float* z, int64_t K, double& tau, double& thr, int64_t& c, double& t_lb) {
  t_lb = -std::numeric_limits<double>::infinity();
  double t, frac;
  if (!sample_start(z, K, t, frac)) return -1;
  if (K > g_warm_max && frac < g_dense_frac) return -1;   // large sparse row: an extra pass over memory costs more than it saves
  double s; int64_t n;
  dense_sum_count(z, K, t, round_down(t, 0.0f), s, n);
  if (n == 0) return -1;
  t = t + (s - 1.0) / double(n);                     // = H({z > t_sample}) <= tau*
  if (!std::isfinite(t)) return -1;
  t_lb = t;
  if (K >= g_small_k && (frac < g_dense_frac || double(n) < 0.5 * g_dense_frac * double(K))) return -1;
  dense_sum_count(z, K, t, round_down(t, 0.0f), s, n);
  if (n == 0) return -1;
  for (int it = 0; it < 24; ++it) {
    const double tn = t + (s - 1.0) / double(n);
    if (!std::isfinite(tn)) return -1;
    if (!(tn > t)) { thr = t; tau = tn; c = n; return 3; }
    double s2; int64_t n2; dense_sum_count(z, K, tn, round_down(tn, 0.0f), s2, n2);
    if (n2 == 0) return -1;
    if (n2 == n) { thr = tn; tau = tn + (s2 - 1.0) / double(n2); c = n; return 3; }
    t = tn; s = s2; n = n2;
  }
  return -1;
}
inline int solve_f32(const float* z, int64_t K, double& tau, double& thr, int64_t& c) {
  double t_lb = -std::numeric_limits<double>::infinity();
  if (K >= g_dense_min_k) { const int st = dense_search(z, K, tau, thr, c, t_lb); if (st == 3) return 3; }
  Scratch& S = tls;
  float* bv = (float*)S.bv.get(size_t(K + 64) * sizeof(float));
  int32_t* bi = (int32_t*)S.bi.get(size_t(K + 64) * sizeof(int32_t));
  return chunked_search(z, K, bv, bi, S.buf, g_chunk, tau, thr, c, t_lb);
}
inline double write_f32(const float* z, int64_t K, int64_t yr, float* g, double scale, double tau, double thr, int64_t c, bool zero, int st) {
  const float inv = float(scale / double(c));
  if (c == K) std::fill(g, g + K, inv);
  else if (st == 3) {                                // x > thr  <=>  x > round_down(thr) for float x
    const float r = round_down(thr, 0.0f);
    #pragma omp simd
    for (int64_t j = 0; j < K; ++j) g[j] = z[j] > r ? inv : 0.0f;
  } else {
    const int32_t* bi = (const int32_t*)tls.bi.p;
    if (zero) std::memset(g, 0, size_t(K) * sizeof(float));
    for (int64_t q = 0; q < c; ++q) g[bi[q]] = inv;
  }
  g[yr] -= float(scale);
  return 1.0 + (thr - double(z[yr])) + (tau - thr);
}

size_t g_mmap_bytes = size_t(32) << 20, g_mmap_min = size_t(8) << 20;   // tunable, see grad_buffer

// Gradient buffer [N,K].  est = expected number of non-zeros per row (-1: unknown).
// Two modes.  zeroed = true: the buffer comes from mmap, its pages are zero without being written
// (one page fault per touched page), and a row only writes its support.  zeroed = false: plain
// malloc, and each row is cleared by the thread that writes it.  mmap is used for buffers >= 32 MB
// (where malloc would mmap anyway) and for buffers >= 8 MB when the expected number of touched
// pages is below 1/8 of all pages.  (Measured: below 8 MB the mmap/munmap round trip costs more
// than clearing the rows.)
torch::Tensor grad_buffer(int64_t N, int64_t K, torch::ScalarType dt, size_t elem, int64_t est, bool& zeroed) {
  size_t bytes = size_t(N) * size_t(K) * elem;
#ifndef _WIN32
  const bool sparse = est >= 0 && bytes >= g_mmap_min && size_t(N) * size_t(est + 1) * 8 < bytes / 4096;
  if (bytes >= g_mmap_bytes || sparse) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p != MAP_FAILED) {
      zeroed = true;
      return torch::from_blob(p, {N, K}, [bytes](void* q) { munmap(q, bytes); }, torch::dtype(dt));
    }
  }
#endif
  zeroed = false;
  void* p = std::malloc(bytes ? bytes : 1);
  return torch::from_blob(p, {N, K}, [](void* q) { std::free(q); }, torch::dtype(dt));
}

bool g_use_chunked = true;

template <class T>
torch::Tensor run_generic(const torch::Tensor& Z, const torch::Tensor& y, torch::ScalarType dt, double* L, int64_t* kap,
                          double scale, int64_t flat_kmax, int64_t serial_below) {
  const int64_t N = Z.size(0), K = Z.size(1);
  bool zeroed;
  torch::Tensor G = grad_buffer(N, K, dt, sizeof(T), -1, zeroed);
  const T* z = Z.data_ptr<T>(); T* g = G.data_ptr<T>(); const int64_t* yy = y.data_ptr<int64_t>();
  auto body = [&](int64_t b, int64_t e) {
    for (int64_t r = b; r < e; ++r) {
      if (!zeroed) std::memset(g + r * K, 0, size_t(K) * sizeof(T));
      L[r] = fw_row<T>(z + r * K, K, yy[r], g + r * K, scale, flat_kmax, kap[r]);
    }
  };
  if (N * K < serial_below) body(0, N);
  else at::parallel_for(0, N, 1, body);
  return G;
}

torch::Tensor run_f32(const torch::Tensor& Z, const torch::Tensor& y, double* L, int64_t* kap,
                      double scale, int64_t flat_kmax, int64_t serial_below) {
  const int64_t N = Z.size(0), K = Z.size(1);
  if (!g_use_chunked || K >= (int64_t(1) << 31) - 64)
    return run_generic<float>(Z, y, torch::kFloat32, L, kap, scale, flat_kmax, serial_below);
  const float* z = Z.data_ptr<float>(); const int64_t* yy = y.data_ptr<int64_t>();
  // row 0 first: its support size decides how the gradient buffer is allocated
  double tau, thr; int64_t c = 0;
  const int st0 = N > 0 ? solve_f32(z, K, tau, thr, c) : 2;
  bool zeroed;
  torch::Tensor G = grad_buffer(N, K, torch::kFloat32, 4, (st0 == 0 || st0 == 3) ? c : -1, zeroed);
  float* g = G.data_ptr<float>();
  const bool zero = !zeroed;
  if (N > 0) {
    if (st0 == 0 || st0 == 3) { L[0] = write_f32(z, K, yy[0], g, scale, tau, thr, c, zero, st0); kap[0] = c; }
    else { if (zero) std::memset(g, 0, size_t(K) * 4); L[0] = fw_row<float>(z, K, yy[0], g, scale, flat_kmax, kap[0]); }
  }
  auto body = [&](int64_t b, int64_t e) {
    for (int64_t r = b; r < e; ++r) {
      double tau, thr; int64_t c;
      const int st = solve_f32(z + r * K, K, tau, thr, c);
      if (st == 0 || st == 3) { L[r] = write_f32(z + r * K, K, yy[r], g + r * K, scale, tau, thr, c, zero, st); kap[r] = c; }
      else {                                               // generic path, has the robust fallback
        if (zero) std::memset(g + r * K, 0, size_t(K) * 4);
        L[r] = fw_row<float>(z + r * K, K, yy[r], g + r * K, scale, flat_kmax, kap[r]);
      }
    }
  };
  if (N * K < serial_below) body(1, N);
  else at::parallel_for(1, N, 1, body);
  return G;
}

struct Out { torch::Tensor L, G, kap; };

torch::Tensor compute_raw(torch::Tensor Z, torch::Tensor y, double scale, int64_t flat_kmax, int64_t serial_below,
                          double* L, int64_t* kap) {
  if (Z.scalar_type() == torch::kFloat32) return run_f32(Z, y, L, kap, scale, flat_kmax, serial_below);
  if (Z.scalar_type() == torch::kFloat64) return run_generic<double>(Z, y, torch::kFloat64, L, kap, scale, flat_kmax, serial_below);
  TORCH_CHECK(false, "familywise_loss: logits must be float32 or float64");
  return torch::Tensor();
}
void check_args(const torch::Tensor& Z, const torch::Tensor& y) {
  TORCH_CHECK(Z.dim() == 2 && Z.device().is_cpu(), "familywise_loss: logits must be a 2-D CPU tensor");
  TORCH_CHECK(y.dim() == 1 && y.size(0) == Z.size(0) && y.scalar_type() == torch::kLong, "familywise_loss: bad target");
  TORCH_CHECK(Z.size(1) >= 1, "familywise_loss: K must be >= 1");
}
Out compute(torch::Tensor Z, torch::Tensor y, double scale, int64_t flat_kmax, int64_t serial_below) {
  check_args(Z, y);
  Z = Z.contiguous(); y = y.contiguous();
  Out o;
  o.L = torch::empty({Z.size(0)}, torch::kFloat64); o.kap = torch::empty({Z.size(0)}, torch::kLong);
  o.G = compute_raw(Z, y, scale, flat_kmax, serial_below, o.L.data_ptr<double>(), o.kap.data_ptr<int64_t>());
  return o;
}

int64_t g_flat_kmax = std::numeric_limits<int64_t>::max(), g_serial_below = 30000;

struct FWFunction : public torch::autograd::Function<FWFunction> {
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor Z, torch::Tensor y, double scale) {
    check_args(Z, y);
    torch::Tensor Zc = Z.detach().contiguous(), yc = y.contiguous();
    const int64_t N = Zc.size(0);
    static thread_local std::vector<double> L; static thread_local std::vector<int64_t> kap;
    if (int64_t(L.size()) < N) { L.resize(N); kap.resize(N); }
    torch::Tensor G = compute_raw(Zc, yc, scale, g_flat_kmax, g_serial_below, L.data(), kap.data());
    double tot = 0.0;
    for (int64_t r = 0; r < N; ++r) tot += L[r];
    ctx->saved_data["G"] = std::move(G);
    return torch::scalar_tensor(tot * scale, torch::dtype(Z.scalar_type()));
  }
  static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list go) {
    torch::Tensor G = ctx->saved_data["G"].toTensor();
    ctx->saved_data.erase("G");
    const double g = go[0].item<double>();
    if (g != 1.0) G.mul_(g);
    return {G, torch::Tensor(), torch::Tensor()};
  }
};

}  // namespace

// reduction: 0 = sum, 1 = mean
torch::Tensor familywise_loss(torch::Tensor Z, torch::Tensor y, int64_t reduction) {
  const double scale = reduction == 1 ? 1.0 / double(Z.size(0)) : 1.0;
  return FWFunction::apply(Z, y, scale);
}
// (per-sample loss [N] float64, scale * gradient [N,K], kappa_bar [N]) -- for tests
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fw_loss_grad(torch::Tensor Z, torch::Tensor y, double scale,
                                                                    int64_t flat_kmax, int64_t serial_below, bool use_chunked) {
  const bool old = g_use_chunked; g_use_chunked = use_chunked;
  Out o = compute(Z, y, scale, flat_kmax, serial_below);
  g_use_chunked = old;
  return {o.L, o.G, o.kap};
}
void set_params(int64_t flat_kmax, int64_t serial_below, int64_t mmap_bytes, int64_t mmap_min, bool use_chunked,
                int64_t chunk, int64_t cap_factor) {
  g_flat_kmax = flat_kmax; g_serial_below = serial_below; g_mmap_bytes = size_t(mmap_bytes); g_mmap_min = size_t(mmap_min);
  g_use_chunked = use_chunked; g_chunk = chunk; g_cap_factor = cap_factor;
}
void set_dense(int64_t min_k, double frac, double pre, int64_t small_k, int64_t warm_max) { g_dense_min_k = min_k; g_dense_frac = frac; g_pre_frac = pre; g_small_k = small_k; g_warm_max = warm_max; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("familywise_loss", &familywise_loss);
  m.def("fw_loss_grad", &fw_loss_grad);
  m.def("set_params", &set_params);
  m.def("set_dense", &set_dense);
  m.def("simd", []() { return
#if defined(__AVX512F__)
    "avx512";
#elif defined(__AVX2__)
    "avx2";
#else
    "scalar";
#endif
  });
}
