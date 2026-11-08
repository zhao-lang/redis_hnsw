use redis_module::native_types::RedisType;
use redis_module::{RedisValue, raw};

use std::collections::HashMap;
use std::convert::From;
use std::os::raw::c_void;
use std::{fmt, ptr};

use super::hnsw::{Float, Index, Node, SearchResult, metrics};

static INDEX_VERSION: i32 = 0;
static NODE_VERSION: i32 = 0;

impl From<IndexRedis> for Index<f32, f32> {
    fn from(index: IndexRedis) -> Self {
        Index {
            name: index.name.clone(),
            mfunc: match index.mfunc_kind.as_str() {
                "Euclidean" => Box::new(metrics::euclidean),
                _ => Box::new(metrics::euclidean),
            },
            mfunc_kind: match index.mfunc_kind.as_str() {
                "Euclidean" => metrics::MetricFuncs::Euclidean,
                _ => metrics::MetricFuncs::Euclidean,
            },
            data_dim: index.data_dim,
            m: index.m,
            m_max: index.m_max,
            m_max_0: index.m_max_0,
            ef_construction: index.ef_construction,
            level_mult: index.level_mult,
            node_count: index.node_count,
            max_layer: index.max_layer,
            // the next 3 need to be populated from redis
            layers: Vec::new(),
            nodes: HashMap::new(),
            enterpoint: None,
        }
    }
}

#[derive(Default, Clone)]
pub struct IndexRedis {
    pub name: String,               // index name
    pub mfunc_kind: String,         // kind of the metric function
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

impl<T: Float, R: Float> From<Index<T, R>> for IndexRedis {
    fn from(index: Index<T, R>) -> Self {
        IndexRedis {
            name: index.name.clone(),
            mfunc_kind: format!("{:?}", index.mfunc_kind),
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
                    l.iter()
                        .map(|n| n.upgrade().read().name.clone())
                        .collect::<Vec<String>>()
                })
                .collect(),
            nodes: index.nodes.keys().cloned().collect::<Vec<String>>(),
            enterpoint: index
                .enterpoint
                .as_ref()
                .map(|ep| ep.upgrade().read().name.clone()),
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
                None => "null",
            },
        )
    }
}

impl From<IndexRedis> for RedisValue {
    fn from(index: IndexRedis) -> Self {
        let reply: Vec<RedisValue> = vec![
            "name".into(),
            index.name.as_str().into(),
            "metric".into(),
            index.mfunc_kind.as_str().into(),
            "data_dim".into(),
            index.data_dim.into(),
            "m".into(),
            index.m.into(),
            "ef_construction".into(),
            index.ef_construction.into(),
            "level_mult".into(),
            index.level_mult.into(),
            "node_count".into(),
            index.node_count.into(),
            "max_layer".into(),
            index.max_layer.into(),
            "enterpoint".into(),
            index.enterpoint.into(),
        ];
        reply.into()
    }
}

pub static HNSW_INDEX_REDIS_TYPE: RedisType = RedisType::new(
    "hnswindex",
    INDEX_VERSION,
    raw::RedisModuleTypeMethods {
        version: raw::REDISMODULE_TYPE_METHOD_VERSION as u64,
        rdb_load: Some(load_index),
        rdb_save: Some(save_index),
        aof_rewrite: None,
        free: Some(free_index),

        mem_usage: None,
        digest: None,

        aux_load: None,
        aux_save: None,
        aux_save2: None,
        aux_save_triggers: 0,

        free_effort: None,
        unlink: None,
        copy: None,
        defrag: None,

        copy2: None,
        free_effort2: None,
        mem_usage2: None,
        unlink2: None,
    },
);

unsafe extern "C" fn free_index(value: *mut c_void) {
    drop(unsafe { Box::from_raw(value as *mut IndexRedis) });
}

// RDB layout must stay byte-for-byte compatible with existing dumps; bump
// INDEX_VERSION if it ever changes.
unsafe extern "C" fn load_index(rdb: *mut raw::RedisModuleIO, version: i32) -> *mut c_void {
    if version != INDEX_VERSION {
        return ptr::null_mut();
    }
    match read_index(rdb) {
        Some(index) => Box::into_raw(Box::new(index)) as *mut c_void,
        None => ptr::null_mut(),
    }
}

fn read_index(rdb: *mut raw::RedisModuleIO) -> Option<IndexRedis> {
    let mut index = IndexRedis {
        name: load_str(rdb)?,
        mfunc_kind: load_str(rdb)?,
        data_dim: load_usize(rdb)?,
        m: load_usize(rdb)?,
        m_max: load_usize(rdb)?,
        m_max_0: load_usize(rdb)?,
        ef_construction: load_usize(rdb)?,
        level_mult: raw::load_double(rdb).ok()?,
        node_count: load_usize(rdb)?,
        max_layer: load_usize(rdb)?,
        ..Default::default()
    };

    let num_layers = load_usize(rdb)?;
    index.layers = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        index.layers.push(load_str_vec(rdb)?);
    }

    index.nodes = load_str_vec(rdb)?;

    let ep = load_str(rdb)?;
    index.enterpoint = if ep == "null" { None } else { Some(ep) };

    Some(index)
}

unsafe extern "C" fn save_index(rdb: *mut raw::RedisModuleIO, value: *mut c_void) {
    // Borrow only: Redis still owns the value after saving.
    let index = unsafe { &*(value as *const IndexRedis) };

    raw::save_string(rdb, &index.name);
    raw::save_string(rdb, &index.mfunc_kind);
    raw::save_unsigned(rdb, index.data_dim as u64);
    raw::save_unsigned(rdb, index.m as u64);
    raw::save_unsigned(rdb, index.m_max as u64);
    raw::save_unsigned(rdb, index.m_max_0 as u64);
    raw::save_unsigned(rdb, index.ef_construction as u64);
    raw::save_double(rdb, index.level_mult);
    raw::save_unsigned(rdb, index.node_count as u64);
    raw::save_unsigned(rdb, index.max_layer as u64);

    raw::save_unsigned(rdb, index.layers.len() as u64);
    for layer in &index.layers {
        save_str_vec(rdb, layer);
    }

    save_str_vec(rdb, &index.nodes);

    raw::save_string(rdb, index.enterpoint.as_deref().unwrap_or("null"));
}

fn load_usize(rdb: *mut raw::RedisModuleIO) -> Option<usize> {
    raw::load_unsigned(rdb).ok()?.try_into().ok()
}

fn load_str(rdb: *mut raw::RedisModuleIO) -> Option<String> {
    let s = raw::load_string(rdb).ok()?;
    s.try_as_str().ok().map(str::to_owned)
}

/// A length-prefixed list of strings.
fn load_str_vec(rdb: *mut raw::RedisModuleIO) -> Option<Vec<String>> {
    let n = load_usize(rdb)?;
    (0..n).map(|_| load_str(rdb)).collect()
}

fn save_str_vec(rdb: *mut raw::RedisModuleIO, v: &[String]) {
    raw::save_unsigned(rdb, v.len() as u64);
    for s in v {
        raw::save_string(rdb, s);
    }
}

#[derive(Default)]
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
                .iter()
                .map(|l| {
                    l.iter()
                        .map(|n| n.upgrade().read().name.clone())
                        .collect::<Vec<String>>()
                })
                .collect(),
        }
    }
}

impl fmt::Debug for NodeRedis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "data: {:?}, \
             neighbors: {:?}",
            self.data, self.neighbors,
        )
    }
}

impl From<&NodeRedis> for RedisValue {
    fn from(n: &NodeRedis) -> Self {
        let mut reply: Vec<RedisValue> = Vec::new();

        reply.push("data".into());
        reply.push(
            n.data
                .iter()
                .map(|x| *x as f64)
                .collect::<Vec<f64>>()
                .into(),
        );

        reply.push("neighbors".into());
        reply.push(
            n.neighbors
                .iter()
                .map(|layer| {
                    layer
                        .iter()
                        .map(|node| node.into())
                        .collect::<Vec<RedisValue>>()
                        .into()
                })
                .collect::<Vec<RedisValue>>()
                .into(),
        );

        reply.into()
    }
}

pub static HNSW_NODE_REDIS_TYPE: RedisType = RedisType::new(
    "hnswnodet",
    NODE_VERSION,
    raw::RedisModuleTypeMethods {
        version: raw::REDISMODULE_TYPE_METHOD_VERSION as u64,
        rdb_load: Some(load_node),
        rdb_save: Some(save_node),
        aof_rewrite: None,
        free: Some(free_node),

        mem_usage: None,
        digest: None,

        aux_load: None,
        aux_save: None,
        aux_save2: None,
        aux_save_triggers: 0,

        free_effort: None,
        unlink: None,
        copy: None,
        defrag: None,

        copy2: None,
        free_effort2: None,
        mem_usage2: None,
        unlink2: None,
    },
);

unsafe extern "C" fn free_node(value: *mut c_void) {
    drop(unsafe { Box::from_raw(value as *mut NodeRedis) });
}

unsafe extern "C" fn load_node(rdb: *mut raw::RedisModuleIO, version: i32) -> *mut c_void {
    if version != NODE_VERSION {
        return ptr::null_mut();
    }
    match read_node(rdb) {
        Some(node) => Box::into_raw(Box::new(node)) as *mut c_void,
        None => ptr::null_mut(),
    }
}

fn read_node(rdb: *mut raw::RedisModuleIO) -> Option<NodeRedis> {
    let num_datum = load_usize(rdb)?;
    let data = (0..num_datum)
        .map(|_| raw::load_float(rdb).ok())
        .collect::<Option<Vec<f32>>>()?;

    let num_layers = load_usize(rdb)?;
    let neighbors = (0..num_layers)
        .map(|_| load_str_vec(rdb))
        .collect::<Option<Vec<Vec<String>>>>()?;

    Some(NodeRedis { data, neighbors })
}

unsafe extern "C" fn save_node(rdb: *mut raw::RedisModuleIO, value: *mut c_void) {
    // Borrow only: Redis still owns the value after saving.
    let node = unsafe { &*(value as *const NodeRedis) };

    raw::save_unsigned(rdb, node.data.len() as u64);
    for &datum in &node.data {
        raw::save_float(rdb, datum);
    }

    raw::save_unsigned(rdb, node.neighbors.len() as u64);
    for layer in &node.neighbors {
        save_str_vec(rdb, layer);
    }
}

#[derive(Default)]
pub struct SearchResultRedis {
    pub sim: f64,
    pub name: String,
}

impl From<&SearchResult<f32, f32>> for SearchResultRedis {
    fn from(res: &SearchResult<f32, f32>) -> Self {
        SearchResultRedis {
            sim: res.sim.into_inner() as f64,
            name: res.name.clone(),
        }
    }
}

impl From<SearchResultRedis> for RedisValue {
    fn from(sr: SearchResultRedis) -> Self {
        let reply: Vec<RedisValue> = vec![
            "similarity".into(),
            sr.sim.into(),
            "name".into(),
            sr.name.as_str().into(),
        ];
        reply.into()
    }
}
