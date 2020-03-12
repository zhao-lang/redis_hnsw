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

// #[derive(Debug)]
pub struct Index {
    pub name: String,
    pub mfunc_: Box<metrics::MetricFuncT>, // metric function
    pub data_dim_: usize,                  // dimensionality of the data

    pub m_: usize,               // out vertexts per node
    pub m_max_: usize,           // max number of vertexes per node
    pub m_max_0_: usize,         // max number of vertexes at layer 0
    pub ef_construction_: usize, // size of dynamic candidate list
}

impl Index {
    pub fn new(name: String) -> Index {
        Index {
            name: name,
            mfunc_: Box::new(metrics::euclidean),
            data_dim_: 0,
            m_: 0,
            m_max_: 0,
            m_max_0_: 0,
            ef_construction_: 0,
        }
    }
}
