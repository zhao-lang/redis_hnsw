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
        metrics::MetricFuncs::Cosine,
    ] {
        assert_eq!(metrics::MetricFuncs::parse(m.name()), Some(m));
        assert_eq!(metrics::MetricFuncs::from_code(m.code()), Some(m));
    }
    assert_eq!(
        metrics::MetricFuncs::parse("HAMMING"),
        Some(metrics::MetricFuncs::Hamming)
    );
    assert_eq!(
        metrics::MetricFuncs::parse("Cosine"),
        Some(metrics::MetricFuncs::Cosine)
    );
    assert_eq!(metrics::MetricFuncs::parse("manhattan"), None);
    assert_eq!(metrics::MetricFuncs::from_code(3), None);
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn cosine_same_direction_is_zero() {
    let v1 = vec![1.0, 2.0, 3.0];
    let v2 = vec![2.0, 4.0, 6.0];
    assert!(close(metrics::cosine(&v1, &v1, 3), 0.0));
    assert!(close(metrics::cosine(&v1, &v2, 3), 0.0));
}

#[test]
fn cosine_orthogonal_and_opposite() {
    assert!(close(metrics::cosine(&[1.0, 0.0], &[0.0, 5.0], 2), -1.0));
    assert!(close(metrics::cosine(&[1.0, 2.0], &[-3.0, -6.0], 2), -2.0));
}

#[test]
fn cosine_45_degrees() {
    let expected = std::f32::consts::FRAC_1_SQRT_2 - 1.0;
    assert!(close(
        metrics::cosine(&[1.0, 0.0], &[1.0, 1.0], 2),
        expected
    ));
}

#[test]
fn cosine_zero_vector_is_orthogonal() {
    assert_eq!(metrics::cosine(&[0.0, 0.0], &[1.0, 2.0], 2), -1.0);
    assert_eq!(metrics::cosine(&[0.0, 0.0], &[0.0, 0.0], 2), -1.0);
}

#[test]
fn cosine_never_positive() {
    let v = vec![0.1_f32; 4096];
    assert!(metrics::cosine(&v, &v, 4096) <= 0.0);
}
