use redis_module::native_types::RedisType;
use redis_module::{RedisString, RedisValue, raw};

use std::ffi::CString;
use std::os::raw::{c_int, c_void};
use std::ptr;

use super::hnsw::SearchResult;

/// Encoding version of the node type.
///
/// v0 stored the graph's neighbor lists inside every node key, plus a separate
/// index key. v1 stores only the vector and the parameters of the index it
/// belongs to: the graph is a shard-local structure derived from the keys, so
/// keys can move between cluster shards freely.
const NODE_VERSION: i32 = 1;

/// The parameters that define an index. Every shard keeps its own copy, and
/// every node carries one, so a shard that receives a node (slot migration,
/// replication) can build the index even if `HNSW.NEW` never reached it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexParams {
    pub dim: usize,
    pub m: usize,
    pub ef_construction: usize,
}

/// The value stored at a vector key.
#[derive(Clone, Debug)]
pub struct NodeRedis {
    pub index: String,
    pub params: IndexParams,
    pub data: Vec<f32>,
}

pub static HNSW_NODE_REDIS_TYPE: RedisType = RedisType::new(
    "hnswnodet",
    NODE_VERSION,
    raw::RedisModuleTypeMethods {
        version: raw::REDISMODULE_TYPE_METHOD_VERSION as u64,
        rdb_load: Some(load_node),
        rdb_save: Some(save_node),
        aof_rewrite: Some(rewrite_node),
        free: Some(free_node),

        mem_usage: Some(mem_usage_node),
        digest: None,

        // Index definitions travel in the RDB before the keys, so they exist
        // when the keys are loaded. This covers indexes with no local nodes.
        aux_load: Some(aux_load),
        aux_save: Some(aux_save),
        aux_save2: None,
        aux_save_triggers: raw::Aux::Before as c_int,

        free_effort: None,
        unlink: None,
        copy: None,
        defrag: None,

        copy2: None,
        free_effort2: None,
        mem_usage2: None,
        // Called whenever a node leaves the keyspace: DEL, expiry, eviction,
        // overwrite, and MIGRATE on the source shard.
        unlink2: Some(unlink_node),
    },
);

unsafe extern "C" fn free_node(value: *mut c_void) {
    drop(unsafe { Box::from_raw(value as *mut NodeRedis) });
}

unsafe extern "C" fn mem_usage_node(value: *const c_void) -> usize {
    let node = unsafe { &*(value as *const NodeRedis) };
    std::mem::size_of::<NodeRedis>() + node.index.len() + node.data.len() * 4
}

unsafe extern "C" fn unlink_node(ctx: *mut raw::RedisModuleKeyOptCtx, value: *const c_void) {
    let node = unsafe { &*(value as *const NodeRedis) };
    let key = unsafe { raw::RedisModule_GetKeyNameFromOptCtx.unwrap()(ctx) };
    let key = RedisString::string_as_slice(key as *mut _);
    if let Ok(key) = std::str::from_utf8(key) {
        crate::unindex_node(&node.index, key);
    }
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
    let index = load_str(rdb)?;
    let params = load_params(rdb)?;
    let num_datum = load_usize(rdb)?;
    let data = (0..num_datum)
        .map(|_| raw::load_float(rdb).ok())
        .collect::<Option<Vec<f32>>>()?;
    Some(NodeRedis {
        index,
        params,
        data,
    })
}

unsafe extern "C" fn save_node(rdb: *mut raw::RedisModuleIO, value: *mut c_void) {
    // Borrow only: Redis still owns the value after saving.
    let node = unsafe { &*(value as *const NodeRedis) };

    raw::save_string(rdb, &node.index);
    save_params(rdb, &node.params);
    raw::save_unsigned(rdb, node.data.len() as u64);
    for &datum in &node.data {
        raw::save_float(rdb, datum);
    }
}

/// AOF rewrite: recreate the index (idempotent) and the node.
unsafe extern "C" fn rewrite_node(
    aof: *mut raw::RedisModuleIO,
    key: *mut raw::RedisModuleString,
    value: *mut c_void,
) {
    let node = unsafe { &*(value as *const NodeRedis) };
    let key = String::from_utf8_lossy(RedisString::string_as_slice(key)).into_owned();
    let p = &node.params;
    let mut new_args = vec![
        node.index.clone(),
        "DIM".into(),
        p.dim.to_string(),
        "M".into(),
        p.m.to_string(),
        "EFCON".into(),
        p.ef_construction.to_string(),
    ];
    emit_aof(aof, "HNSW.NEW", &mut new_args);

    let mut add_args = vec![
        node.index.clone(),
        key,
        "DATA".into(),
        node.data.len().to_string(),
    ];
    add_args.extend(node.data.iter().map(|d| d.to_string()));
    emit_aof(aof, "HNSW.NODE.ADD", &mut add_args);
}

fn emit_aof(aof: *mut raw::RedisModuleIO, cmd: &str, args: &mut [String]) {
    let strings: Vec<RedisString> = args
        .iter()
        .map(|a| RedisString::create(None, a.as_str()))
        .collect();
    let mut argv: Vec<*mut raw::RedisModuleString> = strings.iter().map(|s| s.inner).collect();
    let cmd = CString::new(cmd).unwrap();
    // "v": an array of RedisModuleString pointers followed by its length.
    unsafe {
        raw::RedisModule_EmitAOF.unwrap()(
            aof,
            cmd.as_ptr(),
            c"v".as_ptr(),
            argv.as_mut_ptr(),
            argv.len(),
        );
    }
}

unsafe extern "C" fn aux_save(rdb: *mut raw::RedisModuleIO, when: c_int) {
    if when != raw::Aux::Before as c_int {
        return;
    }
    let defs = crate::index_definitions();
    raw::save_unsigned(rdb, defs.len() as u64);
    for (name, params) in &defs {
        raw::save_string(rdb, name);
        save_params(rdb, params);
    }
}

unsafe extern "C" fn aux_load(rdb: *mut raw::RedisModuleIO, encver: c_int, when: c_int) -> c_int {
    if encver != NODE_VERSION {
        return raw::Status::Err as c_int;
    }
    if when != raw::Aux::Before as c_int {
        return raw::Status::Ok as c_int;
    }
    let read = || -> Option<Vec<(String, IndexParams)>> {
        let n = load_usize(rdb)?;
        (0..n)
            .map(|_| Some((load_str(rdb)?, load_params(rdb)?)))
            .collect()
    };
    match read() {
        Some(defs) => {
            crate::define_indices(defs);
            raw::Status::Ok as c_int
        }
        None => raw::Status::Err as c_int,
    }
}

fn save_params(rdb: *mut raw::RedisModuleIO, p: &IndexParams) {
    raw::save_unsigned(rdb, p.dim as u64);
    raw::save_unsigned(rdb, p.m as u64);
    raw::save_unsigned(rdb, p.ef_construction as u64);
}

fn load_params(rdb: *mut raw::RedisModuleIO) -> Option<IndexParams> {
    Some(IndexParams {
        dim: load_usize(rdb)?,
        m: load_usize(rdb)?,
        ef_construction: load_usize(rdb)?,
    })
}

fn load_usize(rdb: *mut raw::RedisModuleIO) -> Option<usize> {
    raw::load_unsigned(rdb).ok()?.try_into().ok()
}

fn load_str(rdb: *mut raw::RedisModuleIO) -> Option<String> {
    let s = raw::load_string(rdb).ok()?;
    s.try_as_str().ok().map(str::to_owned)
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
