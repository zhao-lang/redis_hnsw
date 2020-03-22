use crate::hnsw::hnsw::*;

#[test]
fn hnsw_test() {
    let mut index = Index::new("foo", 4, 5, 16);
    assert_eq!(&index.name, "foo");
    assert_eq!(index.data_dim, 4);
    assert_eq!(index.m, 5);
    assert_eq!(index.ef_construction, 16);
    assert_eq!(index.node_count, 0);
    assert_eq!(index.max_layer, 0);
    assert_eq!(index.enterpoint, None);

    let mock_fn = |_s: String, _n: Node<f32>| {};
    for i in 0..100 {
        let name = format!("node{}", i);
        let data = vec![i as f32; 4];
        index.add_node(&name, &data, mock_fn).unwrap();
    }
    assert_eq!(index.node_count, 100);
    assert_ne!(index.enterpoint, None);

    let query = vec![10.0; 4];
    let res = index.search_knn(&query, 5).unwrap();
    assert_eq!(res.len(), 5);
    assert_eq!(res[0].sim.into_inner(), 0.0);
    assert_eq!(res[0].name.as_str(), "node10");
    assert_eq!(res[1].sim.into_inner(), -4.0);
    assert_eq!(res[2].sim.into_inner(), -4.0);
    assert_eq!(res[3].sim.into_inner(), -16.0);
    assert_eq!(res[4].sim.into_inner(), -16.0);
}
