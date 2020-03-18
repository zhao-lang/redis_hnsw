# HNSW for Redis

`redis_hnsw` is a Hierarchical Navigable Small World (HNSW) implementation for Redis.

## Getting Started

Build the module - `cargo build`
Load the module - `redis-server --loadmodule ./target/<build_mode>/libredis_hnsw.<dylib|so>`

### Redis commands
Creating a new index - `hnsw.new <index_name> <data_dim_> <M> <ef_construction>`
Add nodes - `hnsw.node.add <index_name> <node_name> <...data>`
Delete nodes - `hnsw.node.del <index_name> <node_name>`
Search KNN - `hnsw.search <index_name> <k> <..data>`
