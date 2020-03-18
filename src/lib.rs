mod hnsw;
mod types;

#[macro_use]
extern crate redis_module;

#[macro_use]
extern crate lazy_static;

extern crate ordered_float;
extern crate owning_ref;

use hnsw::Index;
use redis_module::{parse_float, parse_unsigned_integer, Context, RedisError, RedisResult};
use std::collections::HashMap;
use std::convert::TryInto;
use std::sync::{Arc, RwLock};
use types::*;

static PREFIX: &str = "hnsw";

lazy_static! {
    static ref INDICES: Arc<RwLock<HashMap<String, Arc<RwLock<Index>>>>> =
        Arc::new(RwLock::new(HashMap::new()));
}

fn new_index(ctx: &Context, args: Vec<String>) -> RedisResult {
    if args.len() < 2 {
        return Err(RedisError::WrongArity);
    }

    ctx.auto_memory();

    let index_name = format!("{}.{}", PREFIX, &args[1]);

    let mut data_dim = 512;
    if args.len() > 2 {
        data_dim = parse_unsigned_integer(&args[2])?
            .try_into()
            .unwrap_or(data_dim);
    }
    let mut m = 5;
    if args.len() > 3 {
        m = parse_unsigned_integer(&args[3])?.try_into().unwrap_or(m);
    }
    let mut ef_construction = 200;
    if args.len() > 4 {
        ef_construction = parse_unsigned_integer(&args[4])?
            .try_into()
            .unwrap_or(ef_construction);
    }

    // create index
    let index = Index::new(&index_name, data_dim, m, ef_construction);
    ctx.log_debug(format!("{:?}", index).as_str());

    // write to redis
    let key = ctx.open_key_writable(&index_name);
    match key.get_value::<Index>(&HNSW_INDEX_REDIS_TYPE)? {
        Some(_) => {
            return Err(RedisError::String(format!(
                "Index: {} already exists",
                &index_name
            )));
        }
        None => {
            key.set_value(&HNSW_INDEX_REDIS_TYPE, index.clone())?;
        }
    }

    // Add index to global hashmap
    INDICES
        .write()
        .unwrap()
        .insert(index_name.clone(), Arc::new(RwLock::new(index)));

    Ok(index_name.into())
}

fn get_index(ctx: &Context, args: Vec<String>) -> RedisResult {
    if args.len() < 2 {
        return Err(RedisError::WrongArity);
    }

    let index_name = format!("{}.{}", PREFIX, &args[1]);

    // get index from global hashmap
    let indices = INDICES.read().unwrap();
    let index = indices
        .get(&index_name)
        .ok_or_else(|| format!("Index: {} does not exist", &args[1]))?
        .read()
        .unwrap();
    ctx.log_debug(format!("{:?}", index).as_str());

    // get index from redis
    ctx.log_debug(format!("get key: {}", &index.name).as_str());
    let rkey = ctx.open_key(&index.name);

    let output: String = match rkey.get_value::<Index>(&HNSW_INDEX_REDIS_TYPE)? {
        Some(value) => format!("{:?}", value).as_str().into(),
        None => String::from(""),
    };

    Ok(output.into())
}

fn add_node(ctx: &Context, args: Vec<String>) -> RedisResult {
    if args.len() < 4 {
        return Err(RedisError::WrongArity);
    }

    ctx.auto_memory();

    let index_name = format!("{}.{}", PREFIX, &args[1]);
    let node_name = format!("{}.{}.{}", PREFIX, &args[1], &args[2]);

    let dataf64 = &args[3..]
        .into_iter()
        .map(|s| parse_float(s))
        .collect::<Result<Vec<f64>, RedisError>>()?;
    let data = dataf64.into_iter().map(|d| *d as f32).collect::<Vec<f32>>();

    // update index in global hashmap
    let indices = INDICES.read().unwrap();
    let mut index = indices
        .get(index_name.as_str())
        .ok_or_else(|| format!("Index: {} does not exist", index_name))?
        .write()
        .unwrap();

    ctx.log_debug(format!("Adding node: {} to Index: {}", &node_name, &index_name).as_str());
    match index.add_node(&node_name, &data, update_node) {
        Err(e) => return Err(e.error_string().into()),
        _ => (),
    }

    // write node to redis
    let node = index.nodes.get(&node_name).unwrap();
    write_node(ctx, &node_name, node.into())?;

    // update index in redis
    let key = ctx.open_key_writable(&index_name);
    match key.get_value::<Index>(&HNSW_INDEX_REDIS_TYPE)? {
        Some(_) => {
            ctx.log_debug(format!("update index: {}", &index_name).as_str());
            key.set_value(&HNSW_INDEX_REDIS_TYPE, index.clone())?;
        }
        None => {
            return Err(RedisError::String(format!(
                "Index: {} does not exist",
                &index_name
            )));
        }
    }

    let index_nodes = format!("{}.{}", index_name, "nodeset");
    ctx.call("SADD", &[&index_nodes, &node_name])?;

    Ok(node_name.into())
}

// TODO handle deletion of enterpoint
fn delete_node(ctx: &Context, args: Vec<String>) -> RedisResult {
    if args.len() < 3 {
        return Err(RedisError::WrongArity);
    }

    let index_name = format!("{}.{}", PREFIX, &args[1]);
    let indices = INDICES.read().unwrap();
    let mut index = indices
        .get(index_name.as_str())
        .ok_or_else(|| format!("Index: {} does not exist", index_name))?
        .write()
        .unwrap();

    let node_name = format!("{}.{}.{}", PREFIX, &args[1], &args[2]);

    match index.delete_node(&node_name, update_node) {
        Err(e) => return Err(e.error_string().into()),
        _ => (),
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

    // update index
    let key = ctx.open_key_writable(&index_name);
    match key.get_value::<Index>(&HNSW_INDEX_REDIS_TYPE)? {
        Some(_) => {
            ctx.log_debug(format!("update index: {}", &index_name).as_str());
            key.set_value(&HNSW_INDEX_REDIS_TYPE, index.clone())?;
        }
        None => {
            return Err(RedisError::String(format!(
                "Index: {} does not exist",
                &index_name
            )));
        }
    }

    // update nodeset

    Ok(1_usize.into())
}

fn search_knn(ctx: &Context, args: Vec<String>) -> RedisResult {
    if args.len() < 4 {
        return Err(RedisError::WrongArity);
    }

    let index_name = format!("{}.{}", PREFIX, &args[1]);
    let indices = INDICES.read().unwrap();
    let index = indices
        .get(index_name.as_str())
        .ok_or_else(|| format!("Index: {} does not exist", index_name))?
        .write()
        .unwrap();
    let k = parse_unsigned_integer(&args[2])? as usize;
    let dataf64 = &args[3..]
        .into_iter()
        .map(|s| parse_float(s))
        .collect::<Result<Vec<f64>, RedisError>>()?;
    let data = dataf64.into_iter().map(|d| *d as f32).collect::<Vec<f32>>();

    ctx.log_debug(
        format!(
            "Searching for {} nearest nodes in Index: {}",
            k, &index_name
        )
        .as_str(),
    );

    match index.search_knn(&data, k) {
        Ok(res) => return Ok(format!("{:?}", res).into()),
        Err(e) => return Err(e.error_string().into()),
    };
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

fn update_node(name: String, node: hnsw::Node<f32>) {
    let ctx = Context::get_thread_safe_context();
    ctx.lock();
    write_node(&ctx, &name, (&node).into()).unwrap();
    ctx.unlock();
}

fn get_node(ctx: &Context, args: Vec<String>) -> RedisResult {
    if args.len() < 3 {
        return Err(RedisError::WrongArity);
    }

    let node_name = format!("{}.{}.{}", PREFIX, &args[1], &args[2]);

    ctx.log_debug(format!("get key: {}", node_name).as_str());

    let key = ctx.open_key(&node_name);

    let value = match key.get_value::<NodeRedis>(&HNSW_NODE_REDIS_TYPE)? {
        Some(value) => format!("{:?}", value).as_str().into(),
        None => ().into(),
    };

    Ok(value)
}

redis_module! {
    name: "hnsw",
    version: 1,
    data_types: [
        HNSW_INDEX_REDIS_TYPE,
        HNSW_NODE_REDIS_TYPE,
    ],
    commands: [
        ["hnsw.new", new_index, "write"],
        ["hnsw.get", get_index, "readonly"],
        ["hnsw.search", search_knn, "readonly"],
        ["hnsw.node.add", add_node, "write"],
        ["hnsw.node.get", get_node, "readonly"],
        ["hnsw.node.del", delete_node, "write"],
    ],
}
