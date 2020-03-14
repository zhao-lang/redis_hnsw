use super::metrics;

use rand::prelude::*;
use redis_module::RedisError;
use std::collections::HashMap;
use std::sync::Mutex;
use std::{clone, fmt};

#[derive(Copy, Clone, Debug)]
pub enum HNSWRedisMode {
    Source,
    Storage,
}

impl fmt::Display for HNSWRedisMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format!("{:?}", self).to_uppercase())
    }
}

pub struct Node<'a> {
    pub name_: String,
    pub data_: Vec<f32>,
    pub neighbors_: Vec<&'a Node<'a>>,
}

impl<'a> Node<'a> {
    fn new(name: &str, data: Vec<f32>) -> Self {
        Node {
            name_: name.to_owned(),
            data_: data,
            neighbors_: Vec::new(),
        }
    }
}

pub struct Index<'a> {
    pub name_: String,                          // index name
    pub mode_: HNSWRedisMode,                   // redis mode
    pub mfunc_: Box<metrics::MetricFuncT>,      // metric function
    pub mfunc_kind_: metrics::MetricFuncs,      // kind of the metric function
    pub data_dim_: usize,                       // dimensionality of the data
    pub m_: usize,                              // out vertexts per node
    pub m_max_: usize,                          // max number of vertexes per node
    pub m_max_0_: usize,                        // max number of vertexes at layer 0
    pub ef_construction_: usize,                // size of dynamic candidate list
    pub level_mult_: f64,                       // level generation factor
    pub node_count_: Mutex<usize>,              // count of nodes
    pub max_layer_: Mutex<usize>,               // idx of top layer
    pub nodes_: HashMap<String, Box<Node<'a>>>, // hashmap of nodes
    pub enterpoint_: Option<String>,            // string key to the enterpoint node
    rng_: StdRng,                               // rng for level generation
}

impl<'a> Index<'a> {
    pub fn new(
        name: &str,
        mode: HNSWRedisMode,
        data_dim: usize,
        m: usize,
        ef_construction: usize,
    ) -> Self {
        Index {
            name_: name.to_string(),
            mode_: mode,
            mfunc_: Box::new(metrics::euclidean),
            mfunc_kind_: metrics::MetricFuncs::Euclidean,
            data_dim_: data_dim,
            m_: m,
            m_max_: m,
            m_max_0_: m * 2,
            ef_construction_: ef_construction,
            level_mult_: 1.0 / (1.0 * m as f64).ln(),
            node_count_: Mutex::new(0),
            max_layer_: Mutex::new(0),
            nodes_: HashMap::new(),
            enterpoint_: None,
            rng_: StdRng::from_entropy(),
        }
    }
}

impl<'a> fmt::Display for Index<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "name: {}\n\
             mode: {}\n\
             metric: {:?}\n\
             data_dim: {}\n\
             M: {}\n\
             ef_construction: {}\n\
             node_count: {:?}\n\
             max_layer: {:?}\n\
             enterpoint: {:?}\n",
            self.name_,
            self.mode_,
            self.mfunc_kind_,
            self.data_dim_,
            self.m_,
            self.ef_construction_,
            *self.node_count_.lock().unwrap(),
            *self.max_layer_.lock().unwrap(),
            self.enterpoint_,
        )
    }
}

impl<'a> clone::Clone for Index<'a> {
    fn clone(&self) -> Self {
        Index {
            name_: self.name_.clone(),
            mode_: self.mode_,
            mfunc_: self.mfunc_.clone(),
            mfunc_kind_: self.mfunc_kind_,
            data_dim_: self.data_dim_,
            m_: self.m_,
            m_max_: self.m_max_,
            m_max_0_: self.m_max_0_,
            ef_construction_: self.ef_construction_,
            level_mult_: self.level_mult_,
            node_count_: Mutex::new(*self.node_count_.lock().unwrap()),
            max_layer_: Mutex::new(*self.max_layer_.lock().unwrap()),
            nodes_: HashMap::new(), // blank map for now, rehydrate nodes from node set in redis
            enterpoint_: self.enterpoint_.clone(),
            rng_: self.rng_.clone(),
        }
    }
}

impl<'a> Index<'a> {
    pub fn add_node(&mut self, name: &str, data: Vec<f32>) -> Result<(), RedisError> {
        if *self.node_count_.lock().unwrap() == 0 {
            let node = Node::new(name, data);

            self.nodes_.insert(name.to_owned(), Box::new(node));
            *self.node_count_.lock().unwrap() += 1;
            self.enterpoint_ = Some(name.to_owned());
        } else {
            if self.nodes_.get(name).is_none() {
                return Err(format!("Node: {:?} already exists", name).into());
            }
        }

        Ok(())
    }
}
