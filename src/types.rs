use redis_module::native_types::RedisType;
use redis_module::raw;
use std::convert::From;
use std::fmt;
use std::os::raw::c_void;

use super::hnsw::{Index, MetricFuncs, Node};

pub struct IndexRedis {
    pub name: String,               // index name
    pub mfunc_kind: MetricFuncs,    // kind of the metric function
    pub data_dim: usize,            // dimensionality of the data
    pub m: usize,                   // out vertexs per node
    pub m_max: usize,               // max number of vertexes per node
    pub m_max_0: usize,             // max number of vertexes at layer 0
    pub ef_construction: usize,     // size of dynamic candidate list
    pub level_mult: f64,            // level generation factor
    pub node_count: usize,          // count of nodes
    pub max_layer: usize,           // idx of top layer
    pub layers: Vec<Vec<String>>,   // distinct nodes in each layer
    pub nodes: Vec<String>,         // set of node names
    pub enterpoint: Option<String>, // string key to the enterpoint node
}

impl From<&Index> for IndexRedis {
    fn from(index: &Index) -> Self {
        IndexRedis {
            name: index.name.clone(),
            mfunc_kind: index.mfunc_kind,
            data_dim: index.data_dim,
            m: index.m,
            m_max: index.m_max,
            m_max_0: index.m_max_0,
            ef_construction: index.ef_construction,
            level_mult: index.level_mult,
            node_count: index.node_count,
            max_layer: index.max_layer,
            layers: index
                .layers
                .iter()
                .map(|l| {
                    l.into_iter()
                        .map(|n| n.read().name.clone())
                        .collect::<Vec<String>>()
                })
                .collect(),
            nodes: index
                .nodes
                .keys()
                .map(|k| k.clone())
                .collect::<Vec<String>>(),
            enterpoint: match &index.enterpoint {
                Some(ep) => Some(ep.read().name.clone()),
                None => None,
            },
        }
    }
}

impl fmt::Debug for IndexRedis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "name: {}, \
             metric: {:?}, \
             data_dim: {}, \
             M: {}, \
             ef_construction: {}, \
             level_mult: {}, \
             node_count: {}, \
             max_layer: {}, \
             enterpoint: {}",
            self.name,
            self.mfunc_kind,
            self.data_dim,
            self.m,
            self.ef_construction,
            self.level_mult,
            self.node_count,
            self.max_layer,
            match &self.enterpoint {
                Some(ep) => ep.as_str(),
                None => "None",
            },
        )
    }
}

pub static HNSW_INDEX_REDIS_TYPE: RedisType = RedisType::new(
    "hnswindex",
    0,
    raw::RedisModuleTypeMethods {
        version: raw::REDISMODULE_TYPE_METHOD_VERSION as u64,
        rdb_load: None,
        rdb_save: None,
        aof_rewrite: None,
        free: Some(free_index),

        mem_usage: None,
        digest: None,

        aux_load: None,
        aux_save: None,
        aux_save_triggers: 0,
    },
);

unsafe extern "C" fn free_index(value: *mut c_void) {
    Box::from_raw(value as *mut IndexRedis);
}

#[derive(Debug)]
pub struct NodeRedis {
    pub data: Vec<f32>,
    pub neighbors: Vec<Vec<String>>, // vector of neighbor node names
}

impl From<&Node<f32>> for NodeRedis {
    fn from(node: &Node<f32>) -> Self {
        let r = node.read();
        NodeRedis {
            data: r.data.to_owned(),
            neighbors: r
                .neighbors
                .to_owned()
                .into_iter()
                .map(|l| {
                    l.into_iter()
                        .map(|n| n.read().name.clone())
                        .collect::<Vec<String>>()
                })
                .collect(),
        }
    }
}

pub static HNSW_NODE_REDIS_TYPE: RedisType = RedisType::new(
    "hnswnodet",
    0,
    raw::RedisModuleTypeMethods {
        version: raw::REDISMODULE_TYPE_METHOD_VERSION as u64,
        rdb_load: None,
        rdb_save: None,
        aof_rewrite: None,
        free: Some(free_node),

        mem_usage: None,
        digest: None,

        aux_load: None,
        aux_save: None,
        aux_save_triggers: 0,
    },
);

unsafe extern "C" fn free_node(value: *mut c_void) {
    Box::from_raw(value as *mut NodeRedis);
}
