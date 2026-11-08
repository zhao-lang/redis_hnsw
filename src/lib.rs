#![allow(clippy::not_unsafe_ptr_arg_deref)]

mod hnsw;
mod types;

use hnsw::{Index, Node};
use redis_module::{
    Context, RedisError, RedisResult, RedisString, RedisValue, key::RedisKey,
    key::RedisKeyWritable, redis_module,
};
use redismodule_cmd::{ArgType, Collection, Command, command, rediscmd_doc};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock, RwLock};
use types::*;

static PREFIX: &str = "hnsw";

fn err(msg: impl Into<String>) -> RedisError {
    RedisError::String(msg.into())
}

fn open_key(ctx: &Context, name: &str) -> RedisKey {
    ctx.open_key(&ctx.create_string(name))
}

fn open_key_writable(ctx: &Context, name: &str) -> RedisKeyWritable {
    ctx.open_key_writable(&ctx.create_string(name))
}

type IndexArc = Arc<RwLock<IndexT>>;
type IndexT = Index<f32, f32>;

static INDICES: LazyLock<RwLock<HashMap<String, IndexArc>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

thread_local! {
    #[rediscmd_doc(clean)]
    static NEW_INDEX_CMD: Command = command!{
        name: "hnsw.new",
        desc: "Create a new HNSW index.",
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
        desc: "Retrieve an HNSW index.",
        args: [
            ["name", "Name of the index.", ArgType::Arg, String, Collection::Unit, None],
        ],
    };

    #[rediscmd_doc]
    static DEL_INDEX_CMD: Command = command!{
        name: "hnsw.del",
        desc: "Delete an HNSW index.",
        args: [
            ["name", "Name of the index.", ArgType::Arg, String, Collection::Unit, None],
        ],
    };

    #[rediscmd_doc]
    static ADD_NODE_CMD: Command = command!{
        name: "hnsw.node.add",
        desc: "Add a node to the index.",
        args: [
            ["index", "name of the index", ArgType::Arg, String, Collection::Unit, None],
            ["node", "name of the node", ArgType::Arg, String, Collection::Unit, None],
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
            ["node", "name of the node", ArgType::Arg, String, Collection::Unit, None],
        ],
    };

    #[rediscmd_doc]
    static DEL_NODE_CMD: Command = command!{
        name: "hnsw.node.del",
        desc: "Delete a node from the index.",
        args: [
            ["index", "name of the index", ArgType::Arg, String, Collection::Unit, None],
            ["node", "name of the node", ArgType::Arg, String, Collection::Unit, None],
        ],
    };

    #[rediscmd_doc]
    static SEARCH_CMD: Command = command!{
        name: "hnsw.search",
        desc: "Search the index for the K nearest elements to the query.",
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

fn new_index(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    ctx.auto_memory();

    let mut parsed = NEW_INDEX_CMD.with(|cmd| cmd.parse_args(args))?;

    let name_suffix = parsed.remove("name").unwrap().as_string()?;
    let index_name = format!("{}.{}", PREFIX, name_suffix);
    let data_dim = parsed.remove("dim").unwrap().as_u64()? as usize;
    let m = parsed.remove("m").unwrap().as_u64()? as usize;
    let ef_construction = parsed.remove("efcon").unwrap().as_u64()? as usize;
    // mL = 1/ln(M) is undefined for M < 2
    if m < 2 {
        return Err(RedisError::Str("M must be at least 2"));
    }

    // write to redis
    let key = open_key_writable(ctx, &index_name);
    match key.get_value::<IndexRedis>(&HNSW_INDEX_REDIS_TYPE)? {
        Some(_) => {
            return Err(RedisError::String(format!(
                "Index: {} already exists",
                index_name
            )));
        }
        None => {
            // create index
            let index = Index::new(
                &index_name,
                Box::new(hnsw::metrics::euclidean),
                data_dim,
                m,
                ef_construction,
            );
            ctx.log_debug(format!("{:?}", index).as_str());
            key.set_value::<IndexRedis>(&HNSW_INDEX_REDIS_TYPE, index.clone().into())?;
            // Add index to global hashmap
            INDICES
                .write()
                .unwrap()
                .insert(index_name, Arc::new(RwLock::new(index)));
        }
    }

    Ok("OK".into())
}

fn get_index(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    ctx.auto_memory();

    let mut parsed = GET_INDEX_CMD.with(|cmd| cmd.parse_args(args))?;

    let name_suffix = parsed.remove("name").unwrap().as_string()?;
    let index_name = format!("{}.{}", PREFIX, name_suffix);

    let index = load_index(ctx, &index_name)?;
    let index = index.try_read().map_err(|e| err(e.to_string()))?;
    ctx.log_debug(format!("Index: {:?}", index).as_str());
    ctx.log_debug(format!("Layers: {:?}", index.layers.len()).as_str());
    ctx.log_debug(format!("Nodes: {:?}", index.nodes.len()).as_str());

    let index_redis: IndexRedis = index.clone().into();

    Ok(index_redis.into())
}

fn delete_index(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    ctx.auto_memory();

    let mut parsed = DEL_INDEX_CMD.with(|cmd| cmd.parse_args(args))?;

    let name_suffix = parsed.remove("name").unwrap().as_string()?;
    let index_name = format!("{}.{}", PREFIX, name_suffix);

    // get index from global hashmap
    load_index(ctx, &index_name)?;
    let mut indices = INDICES.write().unwrap();
    let index = indices
        .remove(&index_name)
        .ok_or_else(|| err(format!("Index: {} does not exist", name_suffix)))?;
    let index = index.try_read().map_err(|e| err(e.to_string()))?;

    for node_name in index.nodes.keys() {
        delete_node_redis(ctx, node_name)?;
    }

    // get index from redis
    ctx.log_debug(format!("deleting index: {}", index_name).as_str());
    let rkey = open_key_writable(ctx, &index_name);

    match rkey.get_value::<IndexRedis>(&HNSW_INDEX_REDIS_TYPE)? {
        Some(_) => rkey.delete()?,
        None => {
            return Err(RedisError::String(format!(
                "Index: {} does not exist",
                name_suffix
            )));
        }
    };

    Ok(1_usize.into())
}

fn load_index(ctx: &Context, index_name: &str) -> Result<IndexArc, RedisError> {
    let mut indices = INDICES.write().unwrap();
    // check if index is in global hashmap
    let index = match indices.entry(index_name.to_string()) {
        Entry::Occupied(o) => o.into_mut(),
        // if index isn't present, load it from redis
        Entry::Vacant(v) => {
            // get index from redis
            ctx.log_debug(format!("get key: {}", index_name).as_str());
            let rkey = open_key(ctx, index_name);

            let index_redis = rkey
                .get_value::<IndexRedis>(&HNSW_INDEX_REDIS_TYPE)?
                .ok_or_else(|| err(format!("Index: {} does not exist", index_name)))?;

            let index = make_index(ctx, index_redis)?;
            v.insert(Arc::new(RwLock::new(index)))
        }
    };

    Ok(index.clone())
}

fn make_index(ctx: &Context, ir: &IndexRedis) -> Result<IndexT, RedisError> {
    let mut index: IndexT = ir.clone().into();

    index.nodes = HashMap::with_capacity(ir.node_count);
    for node_name in &ir.nodes {
        let key = open_key(ctx, node_name);

        let nr = key
            .get_value::<NodeRedis>(&HNSW_NODE_REDIS_TYPE)?
            .ok_or_else(|| err(format!("Node: {} does not exist", node_name)))?;

        let node = Node::new(node_name, &nr.data, index.m_max_0);
        index.nodes.insert(node_name.to_owned(), node);
    }

    // reconstruct nodes
    for node_name in &ir.nodes {
        let target = index.nodes.get(node_name).unwrap();

        let key = open_key(ctx, node_name);

        let nr = key
            .get_value::<NodeRedis>(&HNSW_NODE_REDIS_TYPE)?
            .ok_or_else(|| err(format!("Node: {} does not exist", node_name)))?;
        for layer in &nr.neighbors {
            let mut node_layer = Vec::with_capacity(layer.len());
            for neighbor in layer {
                let nn = index
                    .nodes
                    .get(neighbor)
                    .ok_or_else(|| err(format!("Node: {} does not exist", neighbor)))?;
                node_layer.push(nn.downgrade());
            }
            target.write().neighbors.push(node_layer);
        }
    }

    // reconstruct layers
    for layer in &ir.layers {
        let mut node_layer = HashSet::with_capacity(layer.len());
        for node_name in layer {
            let node = index
                .nodes
                .get(node_name)
                .ok_or_else(|| err(format!("Node: {} does not exist", node_name)))?;
            node_layer.insert(node.downgrade());
        }
        index.layers.push(node_layer);
    }

    // set enterpoint
    index.enterpoint = match &ir.enterpoint {
        Some(node_name) => {
            let node = index
                .nodes
                .get(node_name)
                .ok_or_else(|| err(format!("Node: {} does not exist", node_name)))?;
            Some(node.downgrade())
        }
        None => None,
    };

    index.rebuild_in_neighbors();

    Ok(index)
}

fn update_index(ctx: &Context, index_name: &str, index: &IndexT) -> Result<(), RedisError> {
    let key = open_key_writable(ctx, index_name);
    match key.get_value::<IndexRedis>(&HNSW_INDEX_REDIS_TYPE)? {
        Some(_) => {
            ctx.log_debug(format!("update index: {}", index_name).as_str());
            key.set_value::<IndexRedis>(&HNSW_INDEX_REDIS_TYPE, index.clone().into())?;
        }
        None => {
            return Err(RedisError::String(format!(
                "Index: {} does not exist",
                index_name
            )));
        }
    }
    Ok(())
}

fn add_node(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    ctx.auto_memory();

    let mut parsed = ADD_NODE_CMD.with(|cmd| cmd.parse_args(args))?;

    let index_suffix = parsed.remove("index").unwrap().as_string()?;
    let node_suffix = parsed.remove("node").unwrap().as_string()?;

    let index_name = format!("{}.{}", PREFIX, index_suffix);
    let node_name = format!("{}.{}.{}", PREFIX, index_suffix, node_suffix);

    let dataf64 = parsed.remove("data").unwrap().as_f64vec()?;
    let data = dataf64.iter().map(|d| *d as f32).collect::<Vec<f32>>();

    let index = load_index(ctx, &index_name)?;
    let mut index = index.try_write().map_err(|e| err(e.to_string()))?;

    let up = |name: String, node: Node<f32>| {
        write_node(ctx, &name, (&node).into()).unwrap();
    };

    ctx.log_debug(format!("Adding node: {} to Index: {}", node_name, index_name).as_str());
    index
        .add_node(&node_name, &data, up)
        .map_err(|e| err(e.error_string()))?;

    // write node to redis
    let node = index.nodes.get(&node_name).unwrap();
    write_node(ctx, &node_name, node.into())?;

    // update index in redis
    update_index(ctx, &index_name, &index)?;

    Ok("OK".into())
}

fn delete_node(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    ctx.auto_memory();

    let mut parsed = DEL_NODE_CMD.with(|cmd| cmd.parse_args(args))?;

    let index_suffix = parsed.remove("index").unwrap().as_string()?;
    let node_suffix = parsed.remove("node").unwrap().as_string()?;

    let index_name = format!("{}.{}", PREFIX, index_suffix);
    let node_name = format!("{}.{}.{}", PREFIX, index_suffix, node_suffix);

    let index = load_index(ctx, &index_name)?;
    let mut index = index.try_write().map_err(|e| err(e.to_string()))?;

    let node = index
        .nodes
        .get(&node_name)
        .ok_or_else(|| err(format!("Node: {} does not exist", node_name)))?;
    if Arc::strong_count(&node.0) > 1 {
        return Err(err(format!(
            "{} is being accessed, unable to delete. Try again later",
            node_name
        )));
    }

    let up = |name: String, node: Node<f32>| {
        write_node(ctx, &name, (&node).into()).unwrap();
    };

    index
        .delete_node(&node_name, up)
        .map_err(|e| err(e.error_string()))?;

    delete_node_redis(ctx, &node_name)?;

    // update index in redis
    update_index(ctx, &index_name, &index)?;

    Ok(1_usize.into())
}

fn delete_node_redis(ctx: &Context, node_name: &str) -> Result<(), RedisError> {
    ctx.log_debug(format!("del key: {}", node_name).as_str());
    let rkey = open_key_writable(ctx, node_name);
    match rkey.get_value::<NodeRedis>(&HNSW_NODE_REDIS_TYPE)? {
        Some(_) => rkey.delete()?,
        None => {
            return Err(RedisError::String(format!(
                "Node: {} does not exist",
                node_name
            )));
        }
    };

    Ok(())
}

fn get_node(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    ctx.auto_memory();

    let mut parsed = GET_NODE_CMD.with(|cmd| cmd.parse_args(args))?;

    let index_suffix = parsed.remove("index").unwrap().as_string()?;
    let node_suffix = parsed.remove("node").unwrap().as_string()?;

    let node_name = format!("{}.{}.{}", PREFIX, index_suffix, node_suffix);

    ctx.log_debug(format!("get key: {}", node_name).as_str());

    let key = open_key(ctx, &node_name);

    let value = key
        .get_value::<NodeRedis>(&HNSW_NODE_REDIS_TYPE)?
        .ok_or_else(|| err(format!("Node: {} does not exist", node_name)))?;

    Ok(value.into())
}

fn write_node(ctx: &Context, key: &str, node: NodeRedis) -> RedisResult {
    ctx.log_debug(format!("set key: {}", key).as_str());
    let rkey = open_key_writable(ctx, key);

    match rkey.get_value::<NodeRedis>(&HNSW_NODE_REDIS_TYPE)? {
        Some(value) => {
            value.data = node.data;
            value.neighbors = node.neighbors;
        }
        None => {
            rkey.set_value(&HNSW_NODE_REDIS_TYPE, node)?;
        }
    }
    Ok(key.into())
}

fn search_knn(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    ctx.auto_memory();

    let mut parsed = SEARCH_CMD.with(|cmd| cmd.parse_args(args))?;

    let index_suffix = parsed.remove("index").unwrap().as_string()?;
    let k = parsed.remove("k").unwrap().as_u64()? as usize;
    let dataf64 = parsed.remove("query").unwrap().as_f64vec()?;
    let data = dataf64.iter().map(|d| *d as f32).collect::<Vec<f32>>();

    let index_name = format!("{}.{}", PREFIX, index_suffix);
    let index = load_index(ctx, &index_name)?;
    let index = index.try_read().map_err(|e| err(e.to_string()))?;

    ctx.log_debug(format!("Searching for {} nearest nodes in Index: {}", k, index_name).as_str());

    match index.search_knn(&data, k) {
        Ok(res) => {
            let mut reply: Vec<RedisValue> = Vec::new();
            reply.push(res.len().into());
            for r in &res {
                let sr: SearchResultRedis = r.into();
                reply.push(sr.into());
            }
            Ok(reply.into())
        }
        Err(e) => Err(err(e.error_string())),
    }
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
        HNSW_INDEX_REDIS_TYPE,
        HNSW_NODE_REDIS_TYPE,
    ],
    commands: [
        ["hnsw.new", new_index, "write", 0, 0, 0, ""],
        ["hnsw.get", get_index, "readonly", 0, 0, 0, ""],
        ["hnsw.del", delete_index, "write", 0, 0, 0, ""],
        ["hnsw.search", search_knn, "readonly", 0, 0, 0, ""],
        ["hnsw.node.add", add_node, "write", 0, 0, 0, ""],
        ["hnsw.node.get", get_node, "readonly", 0, 0, 0, ""],
        ["hnsw.node.del", delete_node, "write", 0, 0, 0, ""],
    ],
}
