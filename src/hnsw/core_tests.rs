use crate::hnsw::core::*;
use crate::hnsw::metrics::euclidean;
use std::sync::Arc;
// use std::{thread, time};

#[test]
fn hnsw_test() {
    let n = 100;
    let data_dim = 4;

    // index creation
    let mut index: Index<f32, f32> = Index::new("foo", Box::new(euclidean), data_dim, 5, 16);
    assert_eq!(&index.name, "foo");
    assert_eq!(index.data_dim, data_dim);
    assert_eq!(index.m, 5);
    assert_eq!(index.ef_construction, 16);
    assert_eq!(index.node_count, 0);
    assert_eq!(index.max_layer, 0);
    assert_eq!(index.enterpoint, None);

    let mock_fn = |_s: String, _n: Node<f32>| {};

    // add node
    for i in 0..n {
        let name = format!("node{}", i);
        let data = vec![i as f32; data_dim];
        index.add_node(&name, &data, mock_fn).unwrap();
    }
    // // sleep for a brief period to make sure all threads are done
    // let ten_millis = time::Duration::from_millis(10);
    // thread::sleep(ten_millis);
    for i in 0..n {
        let node_name = format!("node{}", i);
        let node = index.nodes.get(&node_name).unwrap();
        let sc = Arc::strong_count(&node.0);
        if sc > 1 {
            println!("{:?}", node);
        }
        assert_eq!(sc, 1);
    }
    assert_eq!(index.node_count, n);
    assert_ne!(index.enterpoint, None);

    // search
    let query = vec![10.0; 4];
    let res = index.search_knn(&query, 5).unwrap();
    assert_eq!(res.len(), 5);
    assert!((res[0].sim.into_inner() - 0.0).abs() < f32::EPSILON);
    assert_eq!(res[0].name.as_str(), "node10");
    assert!((res[1].sim.into_inner() - -4.0).abs() < f32::EPSILON);
    assert!((res[2].sim.into_inner() - -4.0).abs() < f32::EPSILON);
    assert!((res[3].sim.into_inner() - -16.0).abs() < f32::EPSILON);
    assert!((res[4].sim.into_inner() - -16.0).abs() < f32::EPSILON);

    // delete node
    for i in 0..n {
        let node_name = format!("node{}", i);
        let node = index.nodes.get(&node_name).unwrap().clone();
        index.delete_node(&node_name, mock_fn).unwrap();
        assert_eq!(index.node_count, n - i - 1);
        assert_eq!(index.nodes.get(&node_name).is_none(), true);
        for l in &index.layers {
            assert_eq!(l.contains(&node.downgrade()), false);
        }
        for n in index.nodes.values() {
            for l in &n.read().neighbors {
                for nn in l {
                    assert_ne!(nn.upgrade(), node);
                }
            }
        }
        // // sleep for a brief period to make sure all threads are done
        // let ten_millis = time::Duration::from_millis(10);
        // thread::sleep(ten_millis);
        let sc = Arc::strong_count(&node.0);
        if sc > 1 {
            println!("Delete {:?}", node);
        }
        assert_eq!(sc, 1);
    }
}
