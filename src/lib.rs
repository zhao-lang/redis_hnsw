#![allow(clippy::not_unsafe_ptr_arg_deref)]

//! HNSW for Redis, cluster-ready.
//!
//! Vectors are ordinary keys of the `hnswnodet` type, named by the client and
//! routed by Redis Cluster like any other key. Each shard keeps an in-memory
//! HNSW graph per index over the vectors it currently holds. The graph is
//! derived state: it follows the keyspace through keyspace events (writes,
//! RESTORE from slot migration, RDB load) and the type's unlink callback (DEL,
//! expiry, eviction, overwrite, MIGRATE away), and it is never stored in keys.
//!
//! `HNSW.SEARCH` sent to any node searches the whole cluster: the node
//! coordinates a fan-out to the other shards (see `coordinator`). With `LOCAL`
//! it searches only the receiving node's graph, which is what the coordinator
//! sends to each shard, and what a client-side coordinator can use instead
//! (see `client/hnsw_cluster.py`). Index definitions (`HNSW.NEW` / `HNSW.DEL`)
//! are per shard and must be sent to every primary.

mod coordinator;
#[cfg(test)]
mod coordinator_tests;
mod hnsw;
mod types;

use coordinator::Hit;
use hnsw::Index;
use redis_module::{
    Context, ContextFlags, NotifyEvent, RedisError, RedisResult, RedisString, RedisValue, Status,
    ThreadSafeContext, logging, raw, redis_module,
};
use redismodule_cmd::{ArgType, Collection, Command, command, rediscmd_doc};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::{LazyLock, Mutex, MutexGuard, TryLockError};
use types::*;

fn err(msg: impl Into<String>) -> RedisError {
    RedisError::String(msg.into())
}

type IndexT = Index<f32, f32>;

/// Shard-local indices by name.
///
/// Only the Redis main thread touches this (and a forked RDB child, which has
/// its own copy). Keyspace callbacks can fire while a command runs, so the code
/// never holds the lock across a keyspace write or key open; `indices()` turns
/// any re-entry into an error rather than a deadlock.
static INDICES: LazyLock<Mutex<HashMap<String, IndexT>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn indices() -> Result<MutexGuard<'static, HashMap<String, IndexT>>, RedisError> {
    match INDICES.try_lock() {
        Ok(guard) => Ok(guard),
        Err(TryLockError::Poisoned(p)) => Ok(p.into_inner()),
        Err(TryLockError::WouldBlock) => Err(err("hnsw: re-entrant access to the index registry")),
    }
}

fn new_graph(name: &str, p: &IndexParams) -> IndexT {
    Index::new(
        name,
        Box::new(hnsw::metrics::euclidean),
        p.dim,
        p.m,
        p.ef_construction,
    )
}

fn params_of(index: &IndexT) -> IndexParams {
    IndexParams {
        dim: index.data_dim,
        m: index.m,
        ef_construction: index.ef_construction,
    }
}

/// Definitions of all local indices, for RDB aux save.
pub(crate) fn index_definitions() -> Vec<(String, IndexParams)> {
    match indices() {
        Ok(indices) => indices
            .iter()
            .map(|(name, index)| (name.clone(), params_of(index)))
            .collect(),
        Err(e) => {
            logging::log_warning(format!("{e}: index definitions not saved"));
            Vec::new()
        }
    }
}

/// Define indices loaded from RDB aux data, keeping any that already exist.
pub(crate) fn define_indices(defs: Vec<(String, IndexParams)>) {
    let Ok(mut indices) = indices() else { return };
    for (name, params) in defs {
        indices
            .entry(name.clone())
            .or_insert_with(|| new_graph(&name, &params));
    }
}

/// Remove `key` from the graph of `index`. Called when a node leaves the
/// keyspace.
pub(crate) fn unindex_node(index: &str, key: &str) {
    let mut indices = match indices() {
        Ok(i) => i,
        Err(e) => return logging::log_warning(format!("{e}: {key} not removed from {index}")),
    };
    if let Some(graph) = indices.get_mut(index)
        && graph.nodes.contains_key(key)
    {
        let _ = graph.delete_node(key, |_, _| {});
    }
}

/// Make the graphs agree with the current value of `key`.
fn reconcile(ctx: &Context, key: &str) {
    // Read before locking: opening a key can expire it, which re-enters
    // through the unlink callback.
    let value = ctx
        .open_key(&ctx.create_string(key))
        .get_value::<NodeRedis>(&HNSW_NODE_REDIS_TYPE)
        .ok()
        .flatten()
        .cloned();

    let mut indices = match indices() {
        Ok(i) => i,
        Err(e) => return ctx.log_warning(&format!("{e}: {key} not reconciled")),
    };

    // Drop the key from any graph it no longer belongs to. The unlink callback
    // normally does this already; this is the fallback.
    for (name, graph) in indices.iter_mut() {
        let stale = match (&value, graph.nodes.get(key)) {
            (_, None) => false,
            (Some(v), Some(n)) => v.index != *name || n.read().data != v.data,
            (None, Some(_)) => true,
        };
        if stale {
            let _ = graph.delete_node(key, |_, _| {});
        }
    }

    let Some(node) = value else { return };
    let graph = match indices.entry(node.index.clone()) {
        Entry::Occupied(o) => o.into_mut(),
        Entry::Vacant(v) => {
            // A node arrived (slot migration, replication) before HNSW.NEW
            // reached this shard: its parameters define the index.
            ctx.log_notice(&format!("defining index {} from {}", node.index, key));
            v.insert(new_graph(&node.index, &node.params))
        }
    };
    if params_of(graph) != node.params {
        return ctx.log_warning(&format!(
            "{} was written with parameters {:?} but index {} has {:?}; not indexed",
            key,
            node.params,
            node.index,
            params_of(graph)
        ));
    }
    if !graph.nodes.contains_key(key)
        && let Err(e) = graph.add_node(key, &node.data, |_, _| {})
    {
        ctx.log_warning(&format!("{key} not indexed: {}", e.error_string()));
    }
}

fn on_key_event(ctx: &Context, _class: NotifyEvent, _event: &str, key: &[u8]) {
    if let Ok(key) = std::str::from_utf8(key) {
        reconcile(ctx, key);
    }
}

/// FLUSHALL/FLUSHDB and the start of a load replace the whole dataset, so the
/// graphs and definitions start over. A load brings its definitions back from
/// the RDB aux data before any key.
extern "C" fn on_reset(
    _ctx: *mut raw::RedisModuleCtx,
    eid: raw::RedisModuleEvent,
    subevent: u64,
    _data: *mut std::os::raw::c_void,
) {
    let reset = match eid.id {
        raw::REDISMODULE_EVENT_FLUSHDB => subevent == raw::REDISMODULE_SUBEVENT_FLUSHDB_START,
        raw::REDISMODULE_EVENT_LOADING => matches!(
            subevent,
            raw::REDISMODULE_SUBEVENT_LOADING_RDB_START
                | raw::REDISMODULE_SUBEVENT_LOADING_AOF_START
                | raw::REDISMODULE_SUBEVENT_LOADING_REPL_START
        ),
        _ => false,
    };
    if reset && let Ok(mut indices) = indices() {
        indices.clear();
    }
}

fn init(ctx: &Context, _args: &[RedisString]) -> Status {
    for id in [
        raw::REDISMODULE_EVENT_FLUSHDB,
        raw::REDISMODULE_EVENT_LOADING,
    ] {
        let event = raw::RedisModuleEvent { id, dataver: 1 };
        let res = unsafe {
            raw::RedisModule_SubscribeToServerEvent.unwrap()(ctx.ctx, event, Some(on_reset))
        };
        if res != raw::REDISMODULE_OK as i32 {
            ctx.log_warning("failed to subscribe to server events");
            return Status::Err;
        }
    }
    Status::Ok
}

thread_local! {
    #[rediscmd_doc(clean)]
    static NEW_INDEX_CMD: Command = command!{
        name: "hnsw.new",
        desc: "Create an HNSW index on this shard. Idempotent when the parameters match. In a cluster, send it to every primary.",
        args: [
            ["name", "Name of the index.", ArgType::Arg, String, Collection::Unit, None],
            ["dim", "Dimensionality of the data.", ArgType::Kwarg, u64, Collection::Unit, None],
            [
                "m",
                "Parameter for the number of neighbors to select for each node.",
                ArgType::Kwarg, u64, Collection::Unit, Some(Box::new(5_u64))
            ],
            [
                "efcon",
                "Parameter for the size of the dynamic candidate list.",
                ArgType::Kwarg, u64, Collection::Unit, Some(Box::new(200_u64))
            ],
        ],
    };

    #[rediscmd_doc]
    static GET_INDEX_CMD: Command = command!{
        name: "hnsw.get",
        desc: "Retrieve an HNSW index as seen by this shard (node_count is local).",
        args: [
            ["name", "Name of the index.", ArgType::Arg, String, Collection::Unit, None],
        ],
    };

    #[rediscmd_doc]
    static DEL_INDEX_CMD: Command = command!{
        name: "hnsw.del",
        desc: "Delete an HNSW index and its nodes on this shard. In a cluster, send it to every primary.",
        args: [
            ["name", "Name of the index.", ArgType::Arg, String, Collection::Unit, None],
        ],
    };

    #[rediscmd_doc]
    static ADD_NODE_CMD: Command = command!{
        name: "hnsw.node.add",
        desc: "Add a node to the index, or replace it. The node is stored at `key`.",
        args: [
            ["index", "name of the index", ArgType::Arg, String, Collection::Unit, None],
            ["key", "key that stores the node; it decides the cluster slot", ArgType::Arg, String, Collection::Unit, None],
            [
                "data",
                "Dimensionality followed by a space separated vector of data. Total entries must match `DIM` of index",
                ArgType::Kwarg, f64, Collection::Vec, None
            ],
        ],
    };

    #[rediscmd_doc]
    static GET_NODE_CMD: Command = command!{
        name: "hnsw.node.get",
        desc: "Retrieve a node from the index.",
        args: [
            ["index", "name of the index", ArgType::Arg, String, Collection::Unit, None],
            ["key", "key that stores the node", ArgType::Arg, String, Collection::Unit, None],
        ],
    };

    #[rediscmd_doc]
    static DEL_NODE_CMD: Command = command!{
        name: "hnsw.node.del",
        desc: "Delete a node from the index. Plain DEL on the key works too.",
        args: [
            ["index", "name of the index", ArgType::Arg, String, Collection::Unit, None],
            ["key", "key that stores the node", ArgType::Arg, String, Collection::Unit, None],
        ],
    };

    #[rediscmd_doc]
    static SEARCH_CMD: Command = command!{
        name: "hnsw.search",
        desc: "Search the index for the K nearest elements to the query. In a cluster the receiving node searches every shard; add LOCAL to search only the receiving node's part.",
        args: [
            ["index", "name of the index", ArgType::Arg, String, Collection::Unit, None],
            [
                "k",
                "number of nearest neighbors to return",
                ArgType::Kwarg, u64, Collection::Unit, Some(Box::new(5_u64))
            ],
            [
                "query",
                "Dimensionality followed by a space separated vector of data. Total entries must match `DIM` of index",
                ArgType::Kwarg, f64, Collection::Vec, None
            ],
        ],
    };
}

fn no_index(name: &str) -> RedisError {
    err(format!("Index: {} does not exist", name))
}

fn no_node(index: &str, key: &str) -> RedisError {
    err(format!("Node: {} does not exist in Index: {}", key, index))
}

fn new_index(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    let mut parsed = NEW_INDEX_CMD.with(|cmd| cmd.parse_args(args))?;

    let name = parsed.remove("name").unwrap().as_string()?;
    let params = IndexParams {
        dim: parsed.remove("dim").unwrap().as_u64()? as usize,
        m: parsed.remove("m").unwrap().as_u64()? as usize,
        ef_construction: parsed.remove("efcon").unwrap().as_u64()? as usize,
    };
    if params.dim == 0 {
        return Err(RedisError::Str("DIM must be at least 1"));
    }
    // mL = 1/ln(M) is undefined for M < 2
    if params.m < 2 {
        return Err(RedisError::Str("M must be at least 2"));
    }

    let mut indices = indices()?;
    match indices.entry(name.clone()) {
        Entry::Occupied(o) if params_of(o.get()) != params => {
            return Err(err(format!(
                "Index: {} already exists with different parameters",
                name
            )));
        }
        Entry::Occupied(_) => {}
        Entry::Vacant(v) => {
            v.insert(new_graph(&name, &params));
        }
    }
    drop(indices);

    ctx.replicate_verbatim();
    Ok("OK".into())
}

fn get_index(_ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    let mut parsed = GET_INDEX_CMD.with(|cmd| cmd.parse_args(args))?;
    let name = parsed.remove("name").unwrap().as_string()?;

    let indices = indices()?;
    let index = indices.get(&name).ok_or_else(|| no_index(&name))?;
    let enterpoint: Option<String> = index
        .enterpoint
        .as_ref()
        .map(|ep| ep.upgrade().read().name.clone());
    let reply: Vec<RedisValue> = vec![
        "name".into(),
        index.name.as_str().into(),
        "metric".into(),
        format!("{:?}", index.mfunc_kind).into(),
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
        enterpoint.into(),
    ];
    Ok(reply.into())
}

fn delete_index(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    let mut parsed = DEL_INDEX_CMD.with(|cmd| cmd.parse_args(args))?;
    let name = parsed.remove("name").unwrap().as_string()?;

    // Drop the graph first and release the lock: deleting the keys below
    // re-enters through the unlink callback, which then finds nothing to do.
    let graph = indices()?.remove(&name).ok_or_else(|| no_index(&name))?;

    for key_name in graph.nodes.keys() {
        let key_name = ctx.create_string(key_name.as_str());
        let key = ctx.open_key_writable(&key_name);
        let ours = matches!(
            key.get_value::<NodeRedis>(&HNSW_NODE_REDIS_TYPE),
            Ok(Some(node)) if node.index == name
        );
        if ours {
            key.delete()?;
            ctx.notify_keyspace_event(NotifyEvent::GENERIC, "del", &key_name);
        }
    }

    ctx.replicate_verbatim();
    Ok(1_usize.into())
}

fn add_node(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    let mut parsed = ADD_NODE_CMD.with(|cmd| cmd.parse_args(args))?;

    let index = parsed.remove("index").unwrap().as_string()?;
    let key_name = parsed.remove("key").unwrap().as_string()?;
    let data = parsed
        .remove("data")
        .unwrap()
        .as_f64vec()?
        .iter()
        .map(|d| *d as f32)
        .collect::<Vec<f32>>();

    let params = {
        let indices = indices()?;
        let graph = indices.get(&index).ok_or_else(|| no_index(&index))?;
        params_of(graph)
    };
    if data.len() != params.dim {
        return Err(err(format!(
            "data dimension: {} does not match Index",
            data.len()
        )));
    }

    let key_name = ctx.create_string(key_name.as_str());
    let key = ctx.open_key_writable(&key_name);
    // Errors on a key of another type instead of overwriting it.
    key.get_value::<NodeRedis>(&HNSW_NODE_REDIS_TYPE)?;
    key.set_value(
        &HNSW_NODE_REDIS_TYPE,
        NodeRedis {
            index,
            params,
            data,
        },
    )?;
    // Indexing happens in the keyspace event handler, the same path that
    // picks up keys arriving through RESTORE or RDB load.
    ctx.notify_keyspace_event(NotifyEvent::MODULE, "hnsw.node.add", &key_name);

    ctx.replicate_verbatim();
    Ok("OK".into())
}

fn get_node(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    let mut parsed = GET_NODE_CMD.with(|cmd| cmd.parse_args(args))?;
    let index = parsed.remove("index").unwrap().as_string()?;
    let key_name = parsed.remove("key").unwrap().as_string()?;

    let key = ctx.open_key(&ctx.create_string(key_name.as_str()));
    let node = match key.get_value::<NodeRedis>(&HNSW_NODE_REDIS_TYPE)? {
        Some(node) if node.index == index => node.clone(),
        _ => return Err(no_node(&index, &key_name)),
    };

    // Links in this shard's graph, per layer.
    let neighbors: Vec<RedisValue> = indices()?
        .get(&index)
        .and_then(|g| g.nodes.get(&key_name))
        .map(|n| {
            n.read()
                .neighbors
                .iter()
                .map(|layer| {
                    layer
                        .iter()
                        .map(|w| w.upgrade().read().name.as_str().into())
                        .collect::<Vec<RedisValue>>()
                        .into()
                })
                .collect()
        })
        .unwrap_or_default();

    let reply: Vec<RedisValue> = vec![
        "index".into(),
        node.index.as_str().into(),
        "data".into(),
        node.data
            .iter()
            .map(|x| *x as f64)
            .collect::<Vec<f64>>()
            .into(),
        "neighbors".into(),
        neighbors.into(),
    ];
    Ok(reply.into())
}

fn delete_node(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    let mut parsed = DEL_NODE_CMD.with(|cmd| cmd.parse_args(args))?;
    let index = parsed.remove("index").unwrap().as_string()?;
    let key_name = parsed.remove("key").unwrap().as_string()?;

    let key_rs = ctx.create_string(key_name.as_str());
    let key = ctx.open_key_writable(&key_rs);
    match key.get_value::<NodeRedis>(&HNSW_NODE_REDIS_TYPE)? {
        Some(node) if node.index == index => {}
        _ => return Err(no_node(&index, &key_name)),
    }
    // The unlink callback removes it from the graph.
    key.delete()?;
    ctx.notify_keyspace_event(NotifyEvent::GENERIC, "del", &key_rs);

    ctx.replicate_verbatim();
    Ok(1_usize.into())
}

fn search_knn(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    // LOCAL is a bare flag, which the argument parser has no type for. Only
    // look past the index name (argv[1]), which may itself be "local".
    let local_at = args
        .iter()
        .skip(2)
        .position(|a| a.as_slice().eq_ignore_ascii_case(b"LOCAL"))
        .map(|i| i + 2);
    let mut args = args;
    let local = local_at.map(|i| args.remove(i)).is_some();
    let raw_args: Vec<Vec<u8>> = args.iter().map(|a| a.as_slice().to_vec()).collect();

    let mut parsed = SEARCH_CMD.with(|cmd| cmd.parse_args(args))?;

    let index = parsed.remove("index").unwrap().as_string()?;
    let k = parsed.remove("k").unwrap().as_u64()? as usize;
    let data = parsed
        .remove("query")
        .unwrap()
        .as_f64vec()?
        .iter()
        .map(|d| *d as f32)
        .collect::<Vec<f32>>();

    // The graph is only touched on the main thread, so search it here.
    let hits: Vec<Hit> = {
        let indices = indices()?;
        let graph = indices.get(&index).ok_or_else(|| no_index(&index))?;
        graph
            .search_knn(&data, k)
            .map_err(|e| err(e.error_string()))?
            .iter()
            .map(|r| Hit {
                sim: r.sim.into_inner() as f64,
                name: r.name.clone(),
            })
            .collect()
    };

    let flags = ctx.get_flags();
    if local || !flags.contains(ContextFlags::CLUSTER) {
        return Ok(coordinator::reply(&hits));
    }
    let remotes = coordinator::remote_primaries(ctx)?;
    if remotes.is_empty() {
        return Ok(coordinator::reply(&hits));
    }
    if flags.contains(ContextFlags::DENY_BLOCKING) {
        return Err(RedisError::Str(
            "a cluster-wide HNSW.SEARCH cannot run inside MULTI or a script; add LOCAL",
        ));
    }

    let auth = coordinator::read_auth(ctx);
    let mut shard_args = raw_args;
    shard_args.push(b"LOCAL".to_vec());

    let blocked = ctx.block_client();
    std::thread::spawn(move || {
        let (auth, shard_args) = (&auth, &shard_args);
        let results: Vec<Result<Vec<Hit>, String>> = std::thread::scope(|s| {
            let handles: Vec<_> = remotes
                .iter()
                .map(|&addr| s.spawn(move || coordinator::search_remote(addr, auth, shard_args)))
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err("shard search panicked".into()))
                })
                .collect()
        });
        let reply = results
            .into_iter()
            .collect::<Result<Vec<_>, String>>()
            .map(|lists| {
                coordinator::reply(&coordinator::merge(lists.into_iter().chain([hits]), k))
            })
            .map_err(|e| err(format!("HNSW.SEARCH failed on shard {e}")));
        ThreadSafeContext::with_blocked_client(blocked).reply(reply);
    });
    Ok(RedisValue::NoReply)
}

// The Redis allocator only exists inside a running Redis server, so unit tests
// fall back to the system allocator.
#[cfg(not(test))]
type Alloc = redis_module::alloc::RedisAlloc;
#[cfg(not(test))]
const ALLOC: Alloc = redis_module::alloc::RedisAlloc;
#[cfg(test)]
type Alloc = std::alloc::System;
#[cfg(test)]
const ALLOC: Alloc = std::alloc::System;

redis_module! {
    name: "hnsw",
    version: 1,
    allocator: (Alloc, ALLOC),
    data_types: [
        HNSW_NODE_REDIS_TYPE,
    ],
    init: init,
    // Index-level commands are keyless: they act on the shard they are sent
    // to. Node commands declare the node's key (argv[2]) so Redis Cluster
    // routes them to the shard that owns it.
    commands: [
        ["hnsw.new", new_index, "write deny-oom", 0, 0, 0, ""],
        ["hnsw.get", get_index, "readonly", 0, 0, 0, ""],
        ["hnsw.del", delete_index, "write", 0, 0, 0, ""],
        ["hnsw.search", search_knn, "readonly", 0, 0, 0, ""],
        ["hnsw.node.add", add_node, "write deny-oom", 2, 2, 1, ""],
        ["hnsw.node.get", get_node, "readonly", 2, 2, 1, ""],
        ["hnsw.node.del", delete_node, "write", 2, 2, 1, ""],
    ],
    // Keys arriving by any route: our own writes (MODULE), RESTORE / RENAME
    // (GENERIC) and RDB load (LOADED).
    event_handlers: [
        [@GENERIC @MODULE @LOADED: on_key_event],
    ],
}
