pub mod hnsw;

#[macro_use]
extern crate redis_module;

use redis_module::{parse_integer, Context, RedisError, RedisResult};

fn hello_mul(_: &Context, args: Vec<String>) -> RedisResult {
    if args.len() < 2 {
        return Err(RedisError::WrongArity);
    }

    let nums = args.into_iter().skip(1).map(|s| parse_integer(&s)).collect::<Result<Vec<i64>, RedisError>>()?;

    let product = nums.iter().product();

    let mut response = Vec::from(nums);
    response.push(product);

    Ok(response.into())
}

redis_module! {
    name: "hnsw",
    version: 1,
    data_types: [],
    commands: [
        ["hnsw.new", hello_mul, ""],
    ],
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
