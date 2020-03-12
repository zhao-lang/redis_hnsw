use super::metrics;

use std::fmt;

#[derive(Debug)]
pub enum HNSWRedisMode {
    Source,
    Storage,
}

impl fmt::Display for HNSWRedisMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format!("{:?}", self).to_uppercase())
    }
}

fn euclidean(v1: &Vec<f32>, v2: &Vec<f32>, n: usize) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return metrics::sim_func_avx_euc(v1, v2, n);
        }
    }
    metrics::sim_func_euc(v1, v2, n)
}

// #[derive(Debug)]
pub struct Index {
    pub name: String,
    pub mfunc_: Box<fn(&Vec<f32>, &Vec<f32>, usize) -> f32>, // metric function
    pub data_dim_: usize,                                    // dimensionality of the data

    pub m_: usize,               // out vertexts per node
    pub m_max_: usize,           // max number of vertexes per node
    pub m_max_0_: usize,         // max number of vertexes at layer 0
    pub ef_construction_: usize, // size of dynamic candidate list
}

impl Index {
    pub fn new(name: String) -> Index {
        Index {
            name: name,
            mfunc_: Box::new(euclidean),
            data_dim_: 0,
            m_: 0,
            m_max_: 0,
            m_max_0_: 0,
            ef_construction_: 0,
        }
    }
}
