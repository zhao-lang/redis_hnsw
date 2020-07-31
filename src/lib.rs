mod hnsw;
mod types;

#[macro_use]
extern crate redis_module;

#[macro_use]
extern crate redismodule_cmd;

#[macro_use]
extern crate lazy_static;

extern crate num;
extern crate ordered_float;
extern crate owning_ref;

use hnsw::{Index, Node};
use redis_module::{
    Context, RedisError, RedisResult, RedisValue,
};
use redismodule_cmd::{Command, ArgType, Collection};
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;
use std::sync::{Arc, RwLock};
use types::*;

static PREFIX: &str = "hnsw";

type IndexArc = Arc<RwLock<IndexT>>;
type IndexT = Index<f32, f32>;

lazy_static! {
    static ref INDICES: Arc<RwLock<HashMap<String, IndexArc>>> =
        Arc::new(RwLock::new(HashMap::new()));
}

thread_local! {
    static NEW_INDEX_CMD: Command = command!{
        name: "hnsw.new",
        args: [
            ["name", ArgType::Arg, String, Collection::Unit, None],
            ["dim", ArgType::Kwarg, u64, Collection::Unit, None],
            ["m", ArgType::Kwarg, u64, Collection::Unit, Some(Box::new(5_u64))],
            ["efcon", ArgType::Kwarg, u64, Collection::Unit, Some(Box::new(200_u64))],
        ],
    };

    static GET_INDEX_CMD: Command = command!{
        name: "hnsw.get",
        args: [
            ["name", ArgType::Arg, String, Collection::Unit, None],
        ],
    };

    static DEL_INDEX_CMD: Command = command!{
        name: "hnsw.del",
        args: [
            ["name", ArgType::Arg, String, Collection::Unit, None],
        ],
    };

    static ADD_NODE_CMD: Command = command!{
        name: "hnsw.node.add",
        args: [
            ["index", ArgType::Arg, String, Collection::Unit, None],
            ["node", ArgType::Arg, String, Collection::Unit, None],
            ["data", ArgType::Kwarg, f64, Collection::Vec, None],
        ],
    };

    static GET_NODE_CMD: Command = command!{
        name: "hnsw.node.get",
        args: [
            ["index", ArgType::Arg, String, Collection::Unit, None],
            ["node", ArgType::Arg, String, Collection::Unit, None],
        ],
    };

    static DEL_NODE_CMD: Command = command!{
        name: "hnsw.node.del",
        args: [
            ["index", ArgType::Arg, String, Collection::Unit, None],
            ["node", ArgType::Arg, String, Collection::Unit, None],
        ],
    };

    static SEARCH_CMD: Command = command!{
        name: "hnsw.search",
        args: [
            ["index", ArgType::Arg, String, Collection::Unit, None],
            ["k", ArgType::Kwarg, u64, Collection::Unit, Some(Box::new(5_u64))],
            ["query", ArgType::Kwarg, f64, Collection::Vec, None],
        ],
    };
}


fn new_index(ctx: &Context, args: Vec<String>) -> RedisResult {
    ctx.auto_memory();

    let mut parsed = NEW_INDEX_CMD.with(|cmd| {
        cmd.parse_args(args)
    })?;

    let name_suffix = parsed.remove("name").unwrap().as_string()?;
    let index_name = format!("{}.{}", PREFIX, name_suffix);
    let data_dim = parsed.remove("dim").unwrap().as_u64()? as usize;
    let m = parsed.remove("m").unwrap().as_u64()? as usize;
    let ef_construction = parsed.remove("efcon").unwrap().as_u64()? as usize;

    // write to redis
    let key = ctx.open_key_writable(&index_name);
    match key.get_value::<IndexRedis>(&HNSW_INDEX_REDIS_TYPE)? {
        Some(_) => {
            return Err(RedisError::String(format!(
                "Index: {} already exists",
                &index_name
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
            key.set_value::<IndexRedis>(&HNSW_INDEX_REDIS_TYPE, (&index).into())?;
            // Add index to global hashmap
            INDICES
                .write()
                .unwrap()
                .insert(index_name, Arc::new(RwLock::new(index)));
        }
    }

    Ok("OK".into())
}

fn get_index(ctx: &Context, args: Vec<String>) -> RedisResult {
    ctx.auto_memory();

    let mut parsed = GET_INDEX_CMD.with(|cmd| {
        cmd.parse_args(args)
    })?;

    let name_suffix = parsed.remove("name").unwrap().as_string()?;
    let index_name = format!("{}.{}", PREFIX, name_suffix);

    let index = load_index(ctx, &index_name)?;
    let index = match index.try_read() {
        Ok(index) => index,
        Err(e) => return Err(e.to_string().into())
    };

    ctx.log_debug(format!("Index: {:?}", index).as_str());
    ctx.log_debug(format!("Layers: {:?}", index.layers.len()).as_str());
    ctx.log_debug(format!("Nodes: {:?}", index.nodes.len()).as_str());

    let index_redis: IndexRedis = (&*index).into();

    Ok(index_redis.as_redisvalue())
}

fn delete_index(ctx: &Context, args: Vec<String>) -> RedisResult {
    ctx.auto_memory();

    let mut parsed = DEL_INDEX_CMD.with(|cmd| {
        cmd.parse_args(args)
    })?;

    let name_suffix = parsed.remove("name").unwrap().as_string()?;
    let index_name = format!("{}.{}", PREFIX, name_suffix);

    // get index from redis
    ctx.log_debug(format!("deleting index: {}", &index_name).as_str());
    let rkey = ctx.open_key_writable(&index_name);

    match rkey.get_value::<IndexRedis>(&HNSW_INDEX_REDIS_TYPE)? {
        Some(_) => rkey.delete()?,
        None => {
            return Err(RedisError::String(format!(
                "Index: {} does not exist",
                name_suffix
            )));
        }
    };

    // get index from global hashmap
    let mut indices = INDICES.write().unwrap();
    indices
        .remove(&index_name)
        .ok_or_else(|| format!("Index: {} does not exist", name_suffix))?;

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
            ctx.log_debug(format!("get key: {}", &index_name).as_str());
            let rkey = ctx.open_key(&index_name);

            let index_redis = match rkey.get_value::<IndexRedis>(&HNSW_INDEX_REDIS_TYPE)? {
                Some(index) => index,
                None => return Err(format!("Index: {} does not exist", index_name).into()),
            };
            let index = make_index(ctx, index_redis)?;
            v.insert(Arc::new(RwLock::new(index)))
        }
    };

    Ok(index.clone())
}

fn make_index(ctx: &Context, ir: &IndexRedis) -> Result<IndexT, RedisError> {
    let mut index: IndexT = ir.into();

    index.nodes = HashMap::with_capacity(ir.node_count);
    for node_name in &ir.nodes {
        let key = ctx.open_key(&node_name);

        let nr = match key.get_value::<NodeRedis>(&HNSW_NODE_REDIS_TYPE)? {
            Some(n) => n,
            None => return Err(format!("Node: {} does not exist", node_name).into()),
        };
        let node = Node::new(node_name, &nr.data, index.m_max_0);
        index.nodes.insert(node_name.to_owned(), node);
    }

    // reconstruct nodes
    for node_name in &ir.nodes {
        let target = index.nodes.get(node_name).unwrap();

        let key = ctx.open_key(&node_name);

        let nr = match key.get_value::<NodeRedis>(&HNSW_NODE_REDIS_TYPE)? {
            Some(n) => n,
            None => return Err(format!("Node: {} does not exist", node_name).into()),
        };
        for layer in &nr.neighbors {
            let mut node_layer = Vec::with_capacity(layer.len());
            for neighbor in layer {
                let nn = match index.nodes.get(neighbor) {
                    Some(node) => node,
                    None => return Err(format!("Node: {} does not exist", node_name).into()),
                };
                node_layer.push(nn.downgrade());
            }
            target.write().neighbors.push(node_layer);
        }
    }

    // reconstruct layers
    for layer in &ir.layers {
        let mut node_layer = HashSet::with_capacity(layer.len());
        for node_name in layer {
            let node = match index.nodes.get(node_name) {
                Some(n) => n,
                None => return Err(format!("Node: {} does not exist", node_name).into()),
            };
            node_layer.insert(node.downgrade());
        }
        index.layers.push(node_layer);
    }

    // set enterpoint
    index.enterpoint = match &ir.enterpoint {
        Some(node_name) => {
            let node = match index.nodes.get(node_name) {
                Some(n) => n,
                None => return Err(format!("Node: {} does not exist", node_name).into()),
            };
            Some(node.downgrade())
        }
        None => None,
    };

    Ok(index)
}

fn update_index(
    ctx: &Context,
    index_name: &str,
    index: &IndexT,
) -> Result<(), RedisError> {
    let key = ctx.open_key_writable(index_name);
    match key.get_value::<IndexRedis>(&HNSW_INDEX_REDIS_TYPE)? {
        Some(_) => {
            ctx.log_debug(format!("update index: {}", index_name).as_str());
            key.set_value::<IndexRedis>(&HNSW_INDEX_REDIS_TYPE, (&*index).into())?;
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

fn add_node(ctx: &Context, args: Vec<String>) -> RedisResult {
    ctx.auto_memory();

    let mut parsed = ADD_NODE_CMD.with(|cmd| {
        cmd.parse_args(args)
    })?;

    let index_suffix = parsed.remove("index").unwrap().as_string()?;
    let node_suffix = parsed.remove("node").unwrap().as_string()?;

    let index_name = format!("{}.{}", PREFIX, index_suffix);
    let node_name = format!("{}.{}.{}", PREFIX, index_suffix, node_suffix);

    let dataf64 = parsed.remove("data").unwrap().as_f64vec()?;
    let data = dataf64.iter().map(|d| *d as f32).collect::<Vec<f32>>();

    let index = load_index(ctx, &index_name)?;
    let mut index = match index.try_write() {
        Ok(index) => index,
        Err(e) => return Err(e.to_string().into())
    };

    let up = |name: String, node: Node<f32>| {
        write_node(ctx, &name, (&node).into()).unwrap();
    };

    ctx.log_debug(format!("Adding node: {} to Index: {}", &node_name, &index_name).as_str());
    if let Err(e) = index.add_node(&node_name, &data, up) {
        return Err(e.error_string().into())
    }

    // write node to redis
    let node = index.nodes.get(&node_name).unwrap();
    write_node(ctx, &node_name, node.into())?;

    // update index in redis
    update_index(ctx, &index_name, &index)?;

    Ok("OK".into())
}

fn delete_node(ctx: &Context, args: Vec<String>) -> RedisResult {
    ctx.auto_memory();

    let mut parsed = DEL_NODE_CMD.with(|cmd| {
        cmd.parse_args(args)
    })?;

    let index_suffix = parsed.remove("index").unwrap().as_string()?;
    let node_suffix = parsed.remove("node").unwrap().as_string()?;

    let index_name = format!("{}.{}", PREFIX, index_suffix);
    let node_name = format!("{}.{}.{}", PREFIX, index_suffix, node_suffix);

    let index = load_index(ctx, &index_name)?;
    let mut index = match index.try_write() {
        Ok(index) => index,
        Err(e) => return Err(e.to_string().into())
    };
    
    // TODO return error if node has more than 1 strong_count
    let node = index.nodes.get(&node_name).unwrap();
    if Arc::strong_count(&node.0) > 1 {
        return Err(format!(
            "{} is being accessed, unable to delete. Try again later",
            &node_name
        )
        .into());
    }

    let up = |name: String, node: Node<f32>| {
        write_node(ctx, &name, (&node).into()).unwrap();
    };
    
    if let Err(e) = index.delete_node(&node_name, up) {
        return Err(e.error_string().into())
    }

    ctx.log_debug(format!("del key: {}", &node_name).as_str());
    let rkey = ctx.open_key_writable(&node_name);
    match rkey.get_value::<NodeRedis>(&HNSW_NODE_REDIS_TYPE)? {
        Some(_) => rkey.delete()?,
        None => {
            return Err(RedisError::String(format!(
                "Node: {} does not exist",
                &node_name
            )));
        }
    };

    // update index in redis
    update_index(ctx, &index_name, &index)?;

    Ok(1_usize.into())
}

fn get_node(ctx: &Context, args: Vec<String>) -> RedisResult {
    ctx.auto_memory();

    let mut parsed = GET_NODE_CMD.with(|cmd| {
        cmd.parse_args(args)
    })?;

    let index_suffix = parsed.remove("index").unwrap().as_string()?;
    let node_suffix = parsed.remove("node").unwrap().as_string()?;

    let node_name = format!("{}.{}.{}", PREFIX, index_suffix, node_suffix);

    ctx.log_debug(format!("get key: {}", node_name).as_str());

    let key = ctx.open_key(&node_name);

    let value = match key.get_value::<NodeRedis>(&HNSW_NODE_REDIS_TYPE)? {
        Some(node) => node.as_redisvalue(),
        None => {
            return Err(RedisError::String(format!(
                "Node: {} does not exist",
                &node_name
            )));
        }
    };

    Ok(value)
}

fn write_node<'a>(ctx: &'a Context, key: &str, node: NodeRedis) -> RedisResult {
    ctx.log_debug(format!("set key: {}", key).as_str());
    let rkey = ctx.open_key_writable(key);

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

fn search_knn(ctx: &Context, args: Vec<String>) -> RedisResult {
    ctx.auto_memory();

    let mut parsed = SEARCH_CMD.with(|cmd| {
        cmd.parse_args(args)
    })?;

    let index_suffix = parsed.remove("index").unwrap().as_string()?;
    let k = parsed.remove("k").unwrap().as_u64()? as usize;
    let dataf64 = parsed.remove("query").unwrap().as_f64vec()?;
    let data = dataf64.iter().map(|d| *d as f32).collect::<Vec<f32>>();

    let index_name = format!("{}.{}", PREFIX, index_suffix);
    let index = load_index(ctx, &index_name)?;
    let index = match index.try_read() {
        Ok(index) => index,
        Err(e) => return Err(e.to_string().into())
    };

    ctx.log_debug(
        format!(
            "Searching for {} nearest nodes in Index: {}",
            k, &index_name
        )
        .as_str(),
    );

    match index.search_knn(&data, k) {
        Ok(res) => {
            {
                let mut reply: Vec<RedisValue> = Vec::new();
                reply.push(res.len().into());
                for r in &res {
                    let sr: SearchResultRedis = r.into();
                    reply.push(sr.as_redisvalue());
                }
                Ok(reply.into())
            }
        }
        Err(e) => Err(e.error_string().into()),
    }
}

redis_module! {
    name: "hnsw",
    version: 1,
    data_types: [
        HNSW_INDEX_REDIS_TYPE,
        HNSW_NODE_REDIS_TYPE,
    ],
    commands: [
        ["hnsw.new", new_index, "write", 0, 0, 0],
        ["hnsw.get", get_index, "readonly", 0, 0, 0],
        ["hnsw.del", delete_index, "write", 0, 0, 0],
        ["hnsw.search", search_knn, "readonly", 0, 0, 0],
        ["hnsw.node.add", add_node, "write", 0, 0, 0],
        ["hnsw.node.get", get_node, "readonly", 0, 0, 0],
        ["hnsw.node.del", delete_node, "write", 0, 0, 0],
    ],
}
