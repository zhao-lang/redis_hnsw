#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Distance metrics an index can use. Every metric returns a similarity: the
/// negated distance, so larger is closer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MetricFuncs {
    /// Negated squared Euclidean distance.
    Euclidean,
    /// Negated count of positions where the vectors differ.
    Hamming,
    /// Negated cosine distance, `1 - cos(angle)`, in [-2, 0].
    Cosine,
}

impl MetricFuncs {
    pub fn func(self) -> MetricFuncT<f32, f32> {
        match self {
            MetricFuncs::Euclidean => euclidean,
            MetricFuncs::Hamming => hamming,
            MetricFuncs::Cosine => cosine,
        }
    }

    /// Name used in commands (`METRIC`) and replies.
    pub fn name(self) -> &'static str {
        match self {
            MetricFuncs::Euclidean => "euclidean",
            MetricFuncs::Hamming => "hamming",
            MetricFuncs::Cosine => "cosine",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "euclidean" => Some(MetricFuncs::Euclidean),
            "hamming" => Some(MetricFuncs::Hamming),
            "cosine" => Some(MetricFuncs::Cosine),
            _ => None,
        }
    }

    /// Stable code for persistence. Never reuse or renumber a code.
    pub fn code(self) -> u64 {
        match self {
            MetricFuncs::Euclidean => 0,
            MetricFuncs::Hamming => 1,
            MetricFuncs::Cosine => 2,
        }
    }

    pub fn from_code(code: u64) -> Option<Self> {
        match code {
            0 => Some(MetricFuncs::Euclidean),
            1 => Some(MetricFuncs::Hamming),
            2 => Some(MetricFuncs::Cosine),
            _ => None,
        }
    }
}

pub type MetricFuncT<T, R> = fn(&[T], &[T], usize) -> R;

pub fn euclidean(v1: &[f32], v2: &[f32], n: usize) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // TODO remove the check on array length with more flexible avx func
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && v1.len().is_multiple_of(32)
        {
            return sim_func_avx_euc(v1, v2, n);
        }
    }
    sim_func_euc(v1, v2, n)
}

/// Negated Hamming distance: the number of positions where `v1` and `v2`
/// differ. Meant for binary or categorical vectors; any two unequal floats
/// count as one difference.
pub fn hamming(v1: &[f32], v2: &[f32], _n: usize) -> f32 {
    -(v1.iter().zip(v2).filter(|(a, b)| a != b).count() as f32)
}

/// Negated cosine distance: `cos(angle) - 1`, from 0 (same direction) to -2
/// (opposite). Magnitude is ignored. A zero vector has no direction, so its
/// distance to anything is 1, as if orthogonal.
pub fn cosine(v1: &[f32], v2: &[f32], _n: usize) -> f32 {
    // Accumulate in f64: squared norms of large vectors lose precision in f32.
    let (mut dot, mut n1, mut n2) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in v1.iter().zip(v2) {
        let (a, b) = (a as f64, b as f64);
        dot += a * b;
        n1 += a * a;
        n2 += b * b;
    }
    if n1 == 0.0 || n2 == 0.0 {
        return -1.0;
    }
    // Rounding can push |cos| just past 1.
    let cos = (dot / (n1.sqrt() * n2.sqrt())).clamp(-1.0, 1.0);
    (cos - 1.0) as f32
}

fn hsum_ps_sse3(v: __m128) -> f32 {
    unsafe {
        let mut shuf: __m128 = _mm_movehdup_ps(v); // broadcast elements 3,1 to 2,0
        let mut sums: __m128 = _mm_add_ps(v, shuf);
        shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
        sums = _mm_add_ss(sums, shuf);
        _mm_cvtss_f32(sums)
    }
}

fn hsum256_ps_avx(v: __m256) -> f32 {
    unsafe {
        let mut vlow: __m128 = _mm256_castps256_ps128(v);
        let vhigh: __m128 = _mm256_extractf128_ps(v, 1); // high 128
        vlow = _mm_add_ps(vlow, vhigh); // add the low 128
        hsum_ps_sse3(vlow) // and inline the sse3 version, which is optimal for AVX
    }
}

// Multiple accumulators and FMA
// since FMA has a latency of 5 cycles but 0.5 CPI
// https://stackoverflow.com/questions/45735679/euclidean-distance-using-intrinsic-instruction
// TODO: extend functionality for vectors of non-multiples of 32 floats
pub fn sim_func_avx_euc(a: &[f32], b: &[f32], n: usize) -> f32 {
    unsafe {
        let mut euc1: __m256 = _mm256_setzero_ps();
        let mut euc2: __m256 = _mm256_setzero_ps();
        let mut euc3: __m256 = _mm256_setzero_ps();
        let mut euc4: __m256 = _mm256_setzero_ps();

        for i in (0..n).step_by(32) {
            let v1: __m256 = _mm256_sub_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i]));
            euc1 = _mm256_fmadd_ps(v1, v1, euc1);

            let v2: __m256 = _mm256_sub_ps(_mm256_loadu_ps(&a[i + 8]), _mm256_loadu_ps(&b[i + 8]));
            euc2 = _mm256_fmadd_ps(v2, v2, euc2);

            let v3: __m256 =
                _mm256_sub_ps(_mm256_loadu_ps(&a[i + 16]), _mm256_loadu_ps(&b[i + 16]));
            euc3 = _mm256_fmadd_ps(v3, v3, euc3);

            let v4: __m256 =
                _mm256_sub_ps(_mm256_loadu_ps(&a[i + 24]), _mm256_loadu_ps(&b[i + 24]));
            euc4 = _mm256_fmadd_ps(v4, v4, euc4);
        }

        let res: f32 = hsum256_ps_avx(_mm256_add_ps(
            _mm256_add_ps(euc1, euc2),
            _mm256_add_ps(euc3, euc4),
        ));
        -res
    }
}

pub fn sim_func_euc(a: &[f32], b: &[f32], _n: usize) -> f32 {
    -a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .fold(0.0, |acc, x| acc + x)
}
