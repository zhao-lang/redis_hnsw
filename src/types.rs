use redis_module::native_types::RedisType;
use redis_module::{raw, RedisString, RedisValue};
use std::convert::From;
use std::os::raw::c_void;
use std::{fmt, ptr};

use super::hnsw::{Index, Node, SearchResult};

static INDEX_VERSION: i32 = 0;
static NODE_VERSION: i32 = 0;

#[derive(Default)]
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

impl From<&Index<f32, f32>> for IndexRedis {
    fn from(index: &Index<f32, f32>) -> Self {
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
                None => "null",
            },
        )
    }
}

impl IndexRedis {
    pub fn as_redisvalue(&self) -> RedisValue {
        let mut reply: Vec<RedisValue> = Vec::new();

        reply.push("name".into());
        reply.push(self.name.as_str().into());

        reply.push("metric".into());
        reply.push(self.mfunc_kind.as_str().into());

        reply.push("data_dim".into());
        reply.push(self.data_dim.into());

        reply.push("m".into());
        reply.push(self.m.into());

        reply.push("ef_construction".into());
        reply.push(self.ef_construction.into());

        reply.push("level_mult".into());
        reply.push(self.level_mult.into());

        reply.push("node_count".into());
        reply.push(self.node_count.into());

        reply.push("max_layer".into());
        reply.push(self.max_layer.into());

        reply.push("enterpoint".into());
        reply.push(self.enterpoint.clone().into());

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
        aux_save_triggers: 0,
    },
);

unsafe extern "C" fn free_index(value: *mut c_void) {
    Box::from_raw(value as *mut IndexRedis);
}

unsafe extern "C" fn load_index(rdb: *mut raw::RedisModuleIO, version: i32) -> *mut c_void {
    if version != INDEX_VERSION {
        return ptr::null_mut() as *mut c_void;
    }

    let mut index = Box::new(IndexRedis::default());

    let name = raw::RedisModule_LoadString.unwrap()(rdb);
    index.name = redis_module::RedisString::from_ptr(name)
        .unwrap()
        .to_owned();

    let mfunc_kind = raw::RedisModule_LoadString.unwrap()(rdb);
    index.mfunc_kind = redis_module::RedisString::from_ptr(mfunc_kind)
        .unwrap()
        .to_owned();

    index.data_dim = raw::RedisModule_LoadUnsigned.unwrap()(rdb) as usize;
    index.m = raw::RedisModule_LoadUnsigned.unwrap()(rdb) as usize;
    index.m_max = raw::RedisModule_LoadUnsigned.unwrap()(rdb) as usize;
    index.m_max_0 = raw::RedisModule_LoadUnsigned.unwrap()(rdb) as usize;
    index.ef_construction = raw::RedisModule_LoadUnsigned.unwrap()(rdb) as usize;
    index.level_mult = raw::RedisModule_LoadDouble.unwrap()(rdb);
    index.node_count = raw::RedisModule_LoadUnsigned.unwrap()(rdb) as usize;
    index.max_layer = raw::RedisModule_LoadUnsigned.unwrap()(rdb) as usize;

    let num_layers = raw::RedisModule_LoadUnsigned.unwrap()(rdb) as usize;
    index.layers = Vec::with_capacity(num_layers);
    for l in 0..num_layers {
        let num_nodes = raw::RedisModule_LoadUnsigned.unwrap()(rdb) as usize;
        index.layers.push(Vec::with_capacity(num_nodes));
        for _n in 0..num_nodes {
            let node_name = raw::RedisModule_LoadString.unwrap()(rdb);
            index.layers[l].push(
                redis_module::RedisString::from_ptr(node_name)
                    .unwrap()
                    .to_owned(),
            );
        }
    }

    let num_nodes = raw::RedisModule_LoadUnsigned.unwrap()(rdb) as usize;
    index.nodes = Vec::with_capacity(num_nodes);
    for _n in 0..num_nodes {
        let node_name = raw::RedisModule_LoadString.unwrap()(rdb);
        index.nodes.push(
            redis_module::RedisString::from_ptr(node_name)
                .unwrap()
                .to_owned(),
        );
    }

    let ep = raw::RedisModule_LoadString.unwrap()(rdb);
    let ep = redis_module::RedisString::from_ptr(ep).unwrap().to_owned();
    index.enterpoint = match ep.as_str() {
        "null" => None,
        _ => Some(ep),
    };

    let index: *mut c_void = Box::into_raw(index) as *mut c_void;
    index
}

unsafe extern "C" fn save_index(rdb: *mut raw::RedisModuleIO, value: *mut c_void) {
    let index = Box::from_raw(value as *mut IndexRedis);

    let ctx = ptr::null_mut();

    let name = RedisString::create(ctx, &index.name);
    raw::RedisModule_SaveString.unwrap()(rdb, name.inner);

    let mfunc_kind = RedisString::create(ctx, &index.mfunc_kind);
    raw::RedisModule_SaveString.unwrap()(rdb, mfunc_kind.inner);

    raw::RedisModule_SaveUnsigned.unwrap()(rdb, index.data_dim as u64);
    raw::RedisModule_SaveUnsigned.unwrap()(rdb, index.m as u64);
    raw::RedisModule_SaveUnsigned.unwrap()(rdb, index.m_max as u64);
    raw::RedisModule_SaveUnsigned.unwrap()(rdb, index.m_max_0 as u64);
    raw::RedisModule_SaveUnsigned.unwrap()(rdb, index.ef_construction as u64);
    raw::RedisModule_SaveDouble.unwrap()(rdb, index.level_mult);
    raw::RedisModule_SaveUnsigned.unwrap()(rdb, index.node_count as u64);
    raw::RedisModule_SaveUnsigned.unwrap()(rdb, index.max_layer as u64);

    raw::RedisModule_SaveUnsigned.unwrap()(rdb, index.layers.len() as u64);
    for layer in index.layers {
        raw::RedisModule_SaveUnsigned.unwrap()(rdb, layer.len() as u64);
        for n in layer {
            let s = RedisString::create(ctx, &n);
            raw::RedisModule_SaveString.unwrap()(rdb, s.inner);
        }
    }

    raw::RedisModule_SaveUnsigned.unwrap()(rdb, index.nodes.len() as u64);
    for n in index.nodes {
        let s = RedisString::create(ctx, &n);
        raw::RedisModule_SaveString.unwrap()(rdb, s.inner);
    }

    let ep = if index.enterpoint.is_some() {
        RedisString::create(ctx, &index.enterpoint.unwrap())
    } else {
        RedisString::create(ctx, "null")
    };
    raw::RedisModule_SaveString.unwrap()(rdb, ep.inner);
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

impl NodeRedis {
    pub fn as_redisvalue(&self) -> RedisValue {
        let mut reply: Vec<RedisValue> = Vec::new();

        reply.push("data".into());
        reply.push(
            self.data
                .iter()
                .map(|x| *x as f64)
                .collect::<Vec<f64>>()
                .into(),
        );

        reply.push("neighbors".into());
        reply.push(
            self.neighbors
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
        aux_save_triggers: 0,
    },
);

unsafe extern "C" fn free_node(value: *mut c_void) {
    Box::from_raw(value as *mut NodeRedis);
}

unsafe extern "C" fn load_node(rdb: *mut raw::RedisModuleIO, version: i32) -> *mut c_void {
    if version != NODE_VERSION {
        return ptr::null_mut() as *mut c_void;
    }

    let mut node = Box::new(NodeRedis::default());

    let num_datum = raw::RedisModule_LoadUnsigned.unwrap()(rdb) as usize;
    node.data = Vec::with_capacity(num_datum);
    for _d in 0..num_datum {
        let datum = raw::RedisModule_LoadFloat.unwrap()(rdb);
        node.data.push(datum);
    }

    let num_layers = raw::RedisModule_LoadUnsigned.unwrap()(rdb) as usize;
    node.neighbors = Vec::with_capacity(num_layers);
    for l in 0..num_layers {
        let num_nodes = raw::RedisModule_LoadUnsigned.unwrap()(rdb) as usize;
        node.neighbors.push(Vec::new());
        for _n in 0..num_nodes {
            let node_name = raw::RedisModule_LoadString.unwrap()(rdb);
            node.neighbors[l].push(
                redis_module::RedisString::from_ptr(node_name)
                    .unwrap()
                    .to_owned(),
            );
        }
    }

    let p: *mut c_void = Box::into_raw(node) as *mut c_void;
    p
}

unsafe extern "C" fn save_node(rdb: *mut raw::RedisModuleIO, value: *mut c_void) {
    let ctx = ptr::null_mut();

    let node = Box::from_raw(value as *mut NodeRedis);

    raw::RedisModule_SaveUnsigned.unwrap()(rdb, node.data.len() as u64);
    for datum in node.data {
        raw::RedisModule_SaveFloat.unwrap()(rdb, datum);
    }

    raw::RedisModule_SaveUnsigned.unwrap()(rdb, node.neighbors.len() as u64);
    for l in node.neighbors {
        raw::RedisModule_SaveUnsigned.unwrap()(rdb, l.len() as u64);
        for n in l {
            let s = RedisString::create(ctx, &n);
            raw::RedisModule_SaveString.unwrap()(rdb, s.inner);
        }
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

impl SearchResultRedis {
    pub fn as_redisvalue(&self) -> RedisValue {
        let mut reply: Vec<RedisValue> = Vec::new();

        reply.push("similarity".into());
        reply.push(self.sim.into());

        reply.push("name".into());
        reply.push(self.name.as_str().into());

        reply.into()
    }
}
