use redis_module::native_types::RedisType;
use redis_module::{raw, Context, NextArg, RedisResult};
use std::os::raw::c_void;

#[derive(Debug)]
struct HNSWNode {
    meta: String,
    data: Vec<f32>,
    neighbors: Vec<*const HNSWNode>,
}

static HNSW_NODE_REDIS_TYPE: RedisType = RedisType::new(
    "hnswnodet",
    1,
    raw::RedisModuleTypeMethods {
        version: raw::REDISMODULE_TYPE_METHOD_VERSION as u64,
        rdb_load: None,
        rdb_save: None,
        aof_rewrite: None,
        free: Some(free),

        mem_usage: None,
        digest: None,

        aux_load: None,
        aux_save: None,
        aux_save_triggers: 0,
    },
);

unsafe extern "C" fn free(value: *mut c_void) {
    Box::from_raw(value as *mut HNSWNode);
}

pub fn hnsw_node_set(ctx: &Context, args: Vec<String>) -> RedisResult {
    let mut args = args.into_iter().skip(1);
    let key = args.next_string()?;
    
    ctx.log_debug(format!("set key: {}", key).as_str());

    let rkey = ctx.open_key_writable(&key);

    match rkey.get_value::<HNSWNode>(&HNSW_NODE_REDIS_TYPE)? {
        Some(_) => {
            // ctx.log_debug(format!("data: {:?}", value.data).as_str());
            Ok(key.into())
        }
        None => {
            let value = HNSWNode {
                meta: String::from("metadata here"),
                data: vec![0.0; 4],
                neighbors: Vec::new(),
            };

            rkey.set_value(&HNSW_NODE_REDIS_TYPE, value)?;

            Ok(key.into())
        }
    }
}


pub fn hnsw_node_get(ctx: &Context, args: Vec<String>) -> RedisResult {
    let mut args = args.into_iter().skip(1);
    let key = args.next_string()?;

    ctx.log_debug(format!("get key: {}", key).as_str());

    let rkey = ctx.open_key(&key);

    let value = match rkey.get_value::<HNSWNode>(&HNSW_NODE_REDIS_TYPE)? {
        Some(value) => {
            format!("{:?}", value.data).as_str().into()
        }
        None => ().into()
    };

    Ok(value)
}