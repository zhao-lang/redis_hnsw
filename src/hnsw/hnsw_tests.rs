use crate::hnsw::hnsw::*;
use crate::hnsw::metrics::euclidean;
use std::sync::Arc;

#[test]
fn hnsw_test() {
    // index creation
    let mut index: Index<f32, f32> = Index::new("foo", Box::new(euclidean), 4, 5, 16);
    assert_eq!(&index.name, "foo");
    assert_eq!(index.data_dim, 4);
    assert_eq!(index.m, 5);
    assert_eq!(index.ef_construction, 16);
    assert_eq!(index.node_count, 0);
    assert_eq!(index.max_layer, 0);
    assert_eq!(index.enterpoint, None);

    let mock_fn = |_s: String, _n: Node<f32>| {};

    // add node
    for i in 0..100 {
        let name = format!("node{}", i);
        let data = vec![i as f32; 4];
        index.add_node(&name, &data, mock_fn).unwrap();
    }
    assert_eq!(index.node_count, 100);
    assert_ne!(index.enterpoint, None);

    // search
    let query = vec![10.0; 4];
    let res = index.search_knn(&query, 5).unwrap();
    assert_eq!(res.len(), 5);
    assert_eq!(res[0].sim.into_inner(), 0.0);
    assert_eq!(res[0].name.as_str(), "node10");
    assert_eq!(res[1].sim.into_inner(), -4.0);
    assert_eq!(res[2].sim.into_inner(), -4.0);
    assert_eq!(res[3].sim.into_inner(), -16.0);
    assert_eq!(res[4].sim.into_inner(), -16.0);

    // delete node
    for i in 0..100 {
        let node_name = format!("node{}", i);
        let node = index.nodes.get(&node_name).unwrap().clone();
        index.delete_node(&node_name, mock_fn).unwrap();
        assert_eq!(index.node_count, 100 - i - 1);
        assert_eq!(index.nodes.get(&node_name).is_none(), true);
        for l in &index.layers {
            assert_eq!(l.contains(&node), false);
        }
        for n in index.nodes.values() {
            for l in &n.read().neighbors {
                for nn in l {
                    assert_ne!(*nn, node);
                }
            }
        }
        assert_eq!(Arc::strong_count(&node.0), 1);
    }
}
