use crate::hnsw::core::*;
use crate::hnsw::metrics::{MetricFuncs, cosine, euclidean, hamming};
use std::sync::Arc;
// use std::{thread, time};

#[test]
fn hnsw_test() {
    let n = 100;
    let data_dim = 4;

    // index creation
    let mut index: Index<f32, f32> = Index::new(
        "foo",
        Box::new(euclidean),
        MetricFuncs::Euclidean,
        data_dim,
        5,
        16,
    );
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
        assert!(!index.nodes.contains_key(&node_name));
        for l in &index.layers {
            assert!(!l.contains(&node.downgrade()));
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

#[test]
fn error_string_is_plain_message() {
    assert_eq!(HNSWError::from("static msg").error_string(), "static msg");
    assert_eq!(
        HNSWError::from(String::from("owned msg")).error_string(),
        "owned msg"
    );
}

/// Deterministic pseudo-random data so failures are reproducible.
fn lcg_data(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed;
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f32) / ((1u64 << 24) as f32)
    };
    (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
}

fn brute_force_knn(data: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut d: Vec<(f32, usize)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (-euclidean(v, q, q.len()), i))
        .collect();
    d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    d.into_iter().take(k).map(|(_, i)| i).collect()
}

fn recall(index: &Index<f32, f32>, data: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> f64 {
    let mut hits = 0;
    for q in queries {
        let truth = brute_force_knn(data, q, k);
        let got: Vec<String> = index
            .search_knn(q, k)
            .unwrap()
            .into_iter()
            .map(|r| r.name)
            .collect();
        hits += truth
            .iter()
            .filter(|i| got.contains(&format!("n{}", i)))
            .count();
    }
    hits as f64 / (queries.len() * k) as f64
}

#[test]
fn hnsw_recall_vs_brute_force() {
    let (n, dim, k) = (2000, 16, 10);
    let data = lcg_data(n, dim, 42);
    let queries = lcg_data(100, dim, 7);
    let mut index: Index<f32, f32> =
        Index::new("r", Box::new(euclidean), MetricFuncs::Euclidean, dim, 8, 64);
    for (i, v) in data.iter().enumerate() {
        index.add_node(&format!("n{}", i), v, |_, _| {}).unwrap();
    }
    for node in index.nodes.values() {
        let r = node.read();
        for (lc, l) in r.neighbors.iter().enumerate() {
            let cap = if lc == 0 { index.m_max_0 } else { index.m_max };
            assert!(
                l.len() <= cap,
                "layer {} has {} > {} links",
                lc,
                l.len(),
                cap
            );
        }
    }
    let r = recall(&index, &data, &queries, k);
    println!("recall@{} = {:.3}", k, r);
    assert!(r >= 0.95, "recall@{} = {:.3}", k, r);

    // delete half the nodes; the rest must stay searchable with good recall
    let mut rng_keep = Vec::new();
    for i in 0..n {
        if i % 2 == 0 {
            index.delete_node(&format!("n{}", i), |_, _| {}).unwrap();
        } else {
            rng_keep.push(i);
        }
    }
    for node in index.nodes.values() {
        let r = node.read();
        for (lc, l) in r.neighbors.iter().enumerate() {
            let cap = if lc == 0 { index.m_max_0 } else { index.m_max };
            assert!(
                l.len() <= cap,
                "after delete: layer {} has {} > {} links",
                lc,
                l.len(),
                cap
            );
        }
    }
    let kept: Vec<Vec<f32>> = rng_keep.iter().map(|&i| data[i].clone()).collect();
    let mut hits = 0;
    for q in &queries {
        let truth: Vec<usize> = brute_force_knn(&kept, q, k)
            .into_iter()
            .map(|j| rng_keep[j])
            .collect();
        let got: Vec<String> = index
            .search_knn(q, k)
            .unwrap()
            .into_iter()
            .map(|r| r.name)
            .collect();
        hits += truth
            .iter()
            .filter(|i| got.contains(&format!("n{}", i)))
            .count();
    }
    let r = hits as f64 / (queries.len() * k) as f64;
    println!("recall@{} after deleting half = {:.3}", k, r);
    assert!(r >= 0.9, "recall@{} after delete = {:.3}", k, r);
}

#[test]
fn search_returns_k_even_when_k_exceeds_ef() {
    let data = lcg_data(300, 4, 3);
    let mut index: Index<f32, f32> =
        Index::new("k", Box::new(euclidean), MetricFuncs::Euclidean, 4, 5, 10);
    for (i, v) in data.iter().enumerate() {
        index.add_node(&format!("n{}", i), v, |_, _| {}).unwrap();
    }
    assert_eq!(index.search_knn(&data[0], 50).unwrap().len(), 50);
}

#[test]
fn hamming_index_finds_nearest_bit_vectors() {
    let dim = 32;
    // Binary vectors from the bits of distinct seeds.
    let data: Vec<Vec<f32>> = (0..500u64)
        .map(|i| {
            let bits = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32;
            (0..dim).map(|b| ((bits >> b) & 1) as f32).collect()
        })
        .collect();
    let mut index: Index<f32, f32> =
        Index::new("h", Box::new(hamming), MetricFuncs::Hamming, dim, 8, 64);
    for (i, v) in data.iter().enumerate() {
        index.add_node(&format!("n{}", i), v, |_, _| {}).unwrap();
    }

    // An indexed vector is its own nearest neighbor at distance 0.
    let hits = index.search_knn(&data[7], 1).unwrap();
    assert_eq!(hits[0].name, "n7");

    // Flip one bit: the original is still nearest, at distance 1.
    let mut q = data[7].clone();
    q[3] = 1.0 - q[3];
    let hits = index.search_knn(&q, 1).unwrap();
    assert_eq!(hits[0].name, "n7");
    assert_eq!(hamming(&q, &data[7], dim), -1.0);
}

#[test]
fn cosine_index_ignores_magnitude() {
    let data = lcg_data(500, 8, 11);
    let mut index: Index<f32, f32> =
        Index::new("c", Box::new(cosine), MetricFuncs::Cosine, 8, 8, 64);
    for (i, v) in data.iter().enumerate() {
        index.add_node(&format!("n{}", i), v, |_, _| {}).unwrap();
    }

    // A scaled copy of an indexed vector points the same way, so that vector
    // is nearest at distance ~0.
    let q: Vec<f32> = data[42].iter().map(|x| x * 100.0).collect();
    let hits = index.search_knn(&q, 1).unwrap();
    assert_eq!(hits[0].name, "n42");
    assert!(cosine(&q, &data[42], 8).abs() < 1e-6);
}
