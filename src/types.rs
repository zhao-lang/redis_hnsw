use redis_module::native_types::RedisType;
use redis_module::raw;
use std::convert::From;
use std::os::raw::c_void;

use super::hnsw::{Index, Node};

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
    Box::from_raw(value as *mut Index);
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
                        .map(|n| n.read().name.to_owned())
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
