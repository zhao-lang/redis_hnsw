use crate::hnsw::metrics;

#[test]
fn diff_is_zero() {
    let v1 = vec![1.0; 512];
    let v2 = vec![1.0; 512];
    assert_eq!(metrics::sim_func_avx_euc(&v1, &v2, 512), 0.0);
    assert_eq!(metrics::sim_func_euc(&v1, &v2, 512), 0.0);
}

#[test]
fn diff_is_512() {
    let v1 = vec![0.0; 512];
    let v2 = vec![1.0; 512];
    assert_eq!(metrics::sim_func_avx_euc(&v1, &v2, 512), -512.0);
    assert_eq!(metrics::sim_func_euc(&v1, &v2, 512), -512.0);
}

#[test]
fn diff_is_512_2_x512() {
    let v1 = vec![0.0; 512];
    let v2 = vec![512.0; 512];
    assert_eq!(metrics::sim_func_avx_euc(&v1, &v2, 512), -134217728.0);
    assert_eq!(metrics::sim_func_euc(&v1, &v2, 512), -134217728.0);
}

#[test]
fn diff_non_x32() {
    let v1 = vec![0.0; 33];
    let v2 = vec![1.0; 33];
    // assert_eq!(metrics::sim_func_avx_euc(&v1, &v2, 33), -33.0);
    assert_eq!(metrics::sim_func_euc(&v1, &v2, 33), -33.0);
}
