use crate::hnsw::metrics;

#[test]
fn diff_is_zero() {
    let v1 = vec![1.0; 512];
    let v2 = vec![1.0; 512];
    assert!((metrics::sim_func_avx_euc(&v1, &v2, 512) - 0.0).abs() < f32::EPSILON);
    assert!((metrics::sim_func_euc(&v1, &v2, 512) - 0.0).abs() < f32::EPSILON);
}

#[test]
fn diff_is_512() {
    let v1 = vec![0.0; 512];
    let v2 = vec![1.0; 512];
    assert!((metrics::sim_func_avx_euc(&v1, &v2, 512) - -512.0).abs() < f32::EPSILON);
    assert!((metrics::sim_func_euc(&v1, &v2, 512) - -512.0).abs() < f32::EPSILON);
}

#[test]
fn diff_is_512_2_x512() {
    let v1 = vec![0.0; 512];
    let v2 = vec![512.0; 512];
    assert!((metrics::sim_func_avx_euc(&v1, &v2, 512) - -134217728.0).abs() < f32::EPSILON);
    assert!((metrics::sim_func_euc(&v1, &v2, 512) - -134217728.0).abs() < f32::EPSILON);
}

#[test]
fn diff_non_x32() {
    let v1 = vec![0.0; 33];
    let v2 = vec![1.0; 33];
    // assert_eq!(metrics::sim_func_avx_euc(&v1, &v2, 33), -33.0);
    assert!((metrics::sim_func_euc(&v1, &v2, 33) - -33.0).abs() < f32::EPSILON);
}

#[test]
fn hamming_identical_is_zero() {
    let v = vec![1.0, 0.0, 1.0, 1.0];
    assert_eq!(metrics::hamming(&v, &v, 4), 0.0);
}

#[test]
fn hamming_counts_differing_positions() {
    let v1 = vec![1.0, 0.0, 1.0, 1.0, 0.0];
    let v2 = vec![0.0, 0.0, 1.0, 0.0, 1.0];
    assert_eq!(metrics::hamming(&v1, &v2, 5), -3.0);
}

#[test]
fn hamming_ignores_magnitude() {
    let v1 = vec![0.0; 512];
    let v2 = vec![512.0; 512];
    assert_eq!(metrics::hamming(&v1, &v2, 512), -512.0);
}

#[test]
fn metric_names_and_codes_round_trip() {
    for m in [
        metrics::MetricFuncs::Euclidean,
        metrics::MetricFuncs::Hamming,
    ] {
        assert_eq!(metrics::MetricFuncs::parse(m.name()), Some(m));
        assert_eq!(metrics::MetricFuncs::from_code(m.code()), Some(m));
    }
    assert_eq!(
        metrics::MetricFuncs::parse("HAMMING"),
        Some(metrics::MetricFuncs::Hamming)
    );
    assert_eq!(metrics::MetricFuncs::parse("cosine"), None);
}
