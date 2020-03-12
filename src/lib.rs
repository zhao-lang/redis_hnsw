mod hnsw;
mod types;

#[macro_use]
extern crate redis_module;

#[macro_use]
extern crate lazy_static;

use hnsw::{HNSWRedisMode, Index};
use redis_module::{Context, RedisError, RedisResult};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

static PREFIX: &str = "hnsw";

type IndexArcT = Arc<RwLock<Index>>;
type IndexesArcT = Arc<RwLock<HashMap<String, IndexArcT>>>;

lazy_static! {
    static ref INDEXES: IndexesArcT = Arc::new(RwLock::new(HashMap::new()));
}

fn new_index(ctx: &Context, args: Vec<String>) -> RedisResult {
    if args.len() < 3 {
        return Err(RedisError::WrongArity);
    }

    let index_name = format!("{}.{}", PREFIX, &args[1]);
    let index_mode = match args[2].to_uppercase().as_str() {
        "SOURCE" => HNSWRedisMode::Source,
        "STORAGE" => HNSWRedisMode::Storage,
        _ => {
            return Err(RedisError::Str(
                "Invalid HNSW Redis Mode, expected \"SOURCE\" or \"STORAGE\"",
            ))
        }
    };
    let index_nodes = format!("{}.{}", index_name, "nodeset");

    ctx.auto_memory();
    ctx.call(
        "HSET",
        &[&index_name, "redis_mode", index_mode.to_string().as_str()],
    )?;
    ctx.call("SADD", &[&index_nodes, "zero_entry"])?;

    ctx.log_debug(format!("{} is using redis as: {:?}", index_name, index_mode).as_str());

    let index = Index::new(index_name.clone());
    ctx.log_debug(format!("{:?}", index.name).as_str());

    // Add index to global hashmap
    INDEXES
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

    let indexes = INDEXES.read().unwrap();
    let index = indexes
        .get(index_name.as_str())
        .ok_or_else(|| format!("Index: {} does not exists", args[1]))?
        .read()
        .unwrap();
    ctx.log_debug(format!("{:?}", index.name).as_str());

    Ok(index_name.into())
}

redis_module! {
    name: "hnsw",
    version: 1,
    data_types: [],
    commands: [
        ["hnsw.new", new_index, ""],
        ["hnsw.get", get_index, ""],
        ["hnsw.node.set", types::hnsw_node_set, ""],
        ["hnsw.node.get", types::hnsw_node_get, ""],
    ],
}
