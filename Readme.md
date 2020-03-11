# HNSW for Redis

`redis_hnsw` is a Hierarchical Navigable Small World (HNSW) implementation for Redis.

## Getting Started

Build the module - `cargo build`
Load the module - `redis-server --loadmodule ./target/<build_mode>/libredis_hnsw.<dylib|so>`

### Redis commands
Creating a new index - `hnsw.new <index_name> <source|storage>`
