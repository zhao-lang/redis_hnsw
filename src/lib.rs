mod hnsw;
mod types;

#[macro_use]
extern crate redis_module;

use redis_module::{parse_integer, Context, RedisError, RedisResult};
use hnsw::HNSWRedisMode;

static PREFIX: &'static str = "hnsw";

fn example(_: &Context, args: Vec<String>) -> RedisResult {
    if args.len() < 2 {
        return Err(RedisError::WrongArity);
    }

    let nums = args.into_iter().skip(1).map(|s| parse_integer(&s)).collect::<Result<Vec<i64>, RedisError>>()?;

    let product = nums.iter().product();

    let mut response = Vec::from(nums);
    response.push(product);

    Ok(response.into())
}

fn new_index(ctx: &Context, args: Vec<String>) -> RedisResult {
    if args.len() < 3 {
        return Err(RedisError::WrongArity);
    }

    let index_name = format!("{}.{}", PREFIX, &args[1]);
    let index_mode = match args[2].to_uppercase().as_str() {
        "SOURCE" => HNSWRedisMode::Source,
        "STORAGE" => HNSWRedisMode::Storage,
        _ => return Err(RedisError::Str("Invalid HNSW Redis Mode, expected \"SOURCE\" or \"STORAGE\"")),
    };
    let index_nodes = format!("{}.{}", index_name, "nodeset");

    ctx.auto_memory();
    ctx.call("HSET", &[&index_name, "redis_mode", index_mode.to_string().as_str()])?;
    ctx.call("SADD", &[&index_nodes, "zero_entry"])?;

    ctx.log_debug(format!("{} is using redis as: {:?}", index_name, index_mode).as_str());

    Ok(index_name.into())
}

redis_module! {
    name: "hnsw",
    version: 1,
    data_types: [],
    commands: [
        ["hello.mul", example, ""],
        ["hnsw.new", new_index, ""],
        ["hnsw.node.set", types::hnsw_node_set, ""],
        ["hnsw.node.get", types::hnsw_node_get, ""],
    ],
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
