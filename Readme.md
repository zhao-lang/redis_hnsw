[![licence](https://img.shields.io/github/license/zhao-lang/redis_hnsw.svg)](https://github.com/zhao-lang/redis_hnsw/blob/main/LICENSE)
[![release](https://img.shields.io/github/v/release/zhao-lang/redis_hnsw.svg)](https://github.com/zhao-lang/redis_hnsw/releases/latest)
[![rust](https://github.com/zhao-lang/redis_hnsw/workflows/Rust/badge.svg)](https://github.com/zhao-lang/redis_hnsw/actions?query=workflow%3ARust)

# HNSW for Redis
<a id="markdown-hnsw-for-redis" name="hnsw-for-redis"></a>

`redis_hnsw` is a Hierarchical Navigable Small World (HNSW) implementation for Redis. Based on the paper [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320). Currently only supports Euclidean distance, Hamming distance forthcoming.

<!-- TOC -->
## Table of Contents

- [Getting Started](#getting-started)
    - [Redis commands](#redis-commands)
- [Redis Cluster](#redis-cluster)
- [Command Reference](#command-reference)
    - [HNSW.NEW](#hnswnew)
    - [HNSW.GET](#hnswget)
    - [HNSW.DEL](#hnswdel)
    - [HNSW.NODE.ADD](#hnswnodeadd)
    - [HNSW.NODE.GET](#hnswnodeget)
    - [HNSW.NODE.DEL](#hnswnodedel)
    - [HNSW.SEARCH](#hnswsearch)

<!-- /TOC -->

## Getting Started
<a id="markdown-getting-started" name="getting-started"></a>

Requires stable Rust 1.88 or newer, plus libclang for building the Redis bindings (`sudo apt-get install libclang-dev` on Debian/Ubuntu).

Build the module - `cargo build`

Load the module - `redis-server --loadmodule ./target/<build_mode>/libredis_hnsw.<dylib|so>`

### Redis commands
<a id="markdown-redis-commands" name="redis-commands"></a>
Creating a new index - `hnsw.new {index_name} [DIM {data_dim}] [M {m}] [EFCON {ef_construction}]`

Add nodes - `hnsw.node.add {index_name} {key} [DATA {dim} {...data}]`

Delete nodes - `hnsw.node.del {index_name} {key}` (or plain `DEL {key}`)

Search KNN - `hnsw.search {index_name} [K {k}] [QUERY {dim} {...data}] [LOCAL]`

Each node is an ordinary Redis key that you name. The HNSW graph is kept in memory and is not stored in keys. The module keeps it in step with the keyspace, so `DEL`, `EXPIRE`, eviction, `RENAME` and overwrites all update the index. The graph is rebuilt from the vectors when the server loads an RDB or AOF.

> **Upgrading from 0.2:** the on-disk format changed. 0.2 stored index and graph keys, and 0.3 cannot load them. Re-add your vectors after upgrading.

## Redis Cluster
<a id="markdown-redis-cluster" name="redis-cluster"></a>

Every shard keeps its own HNSW graph over the vectors it stores. A cluster-wide search sends the query to every primary and merges the per-shard top-k results.

* **Node commands are routed by key.** `HNSW.NODE.ADD/GET/DEL` declare `{key}` as their key, so the cluster sends them to the shard that owns it. Vectors spread across shards like any other keys. Use a hash tag in the key (`doc:{42}:vec`) to keep a vector next to related data.
* **Index commands are per shard.** `HNSW.NEW`, `HNSW.DEL` and `HNSW.GET` are keyless and act only on the node that receives them. Send `HNSW.NEW` and `HNSW.DEL` to every primary. `HNSW.NEW` is idempotent when the parameters match, so you can re-send it after adding a shard. Each node also carries its index's parameters, so a shard that receives vectors through slot migration defines the index by itself.
* **Search is coordinated by the node that receives it.** Send `HNSW.SEARCH` to any node: a primary, or a replica after `READONLY`. That node searches its own graph, then sends `HNSW.SEARCH ... LOCAL` in parallel to the primary of every other shard. It merges the per-shard top-k results, counts a key that appears on two shards mid-migration only once, and replies with the global top K. The client stays unblocked server-side while the shards answer, so the node keeps serving other commands.
  * The coordinator connects to the other nodes on their client port. If `masterauth` is set, it authenticates with `masteruser`/`masterauth`. TLS is not supported.
  * Each shard gets a 5s timeout. A shard that is down or slow fails the search with an error naming that shard. Give your client a longer socket timeout than that, so it doesn't time out first and retry the search.
  * Inside `MULTI` or a script, the command can't block. Add `LOCAL` there.
* **`LOCAL` searches one shard.** It searches only the receiving node's graph. You can use it to fan out from the client instead of the server.

[doc/CLUSTER.md](doc/CLUSTER.md) explains step by step how the in-module coordinator works, and compares it with fanning out from the client.
* **Slot migration, replication and persistence need nothing extra.** `MIGRATE` removes nodes from the source graph and adds them to the target graph. Replicas build their own graphs from the replication stream. Index definitions are saved in the RDB, so they survive restarts even when a shard holds no nodes.

[`client/hnsw_cluster.py`](client/hnsw_cluster.py) is a reference client built on redis-py. It broadcasts index commands and routes node commands. It searches through the server-side coordinator by default, or with `fanout="client"` it merges `LOCAL` searches itself:

```python
from redis.cluster import RedisCluster
from hnsw_cluster import HnswCluster

hc = HnswCluster(RedisCluster(host="127.0.0.1", port=7000))
hc.create_index("docs", dim=3)
hc.add("docs", "doc:1:vec", [0.1, 0.2, 0.3])
hc.search("docs", [0.1, 0.2, 0.3], k=5)   # [("doc:1:vec", -0.0), ...]
hc.search("docs", [0.1, 0.2, 0.3], k=5, fanout="client")
```

To run the end-to-end test against a local 3-primary, 3-replica cluster (set `PASSWORD` and `REDIS_PASSWORD` to run it with authentication):

```bash
cargo build && scripts/cluster.sh start 3 1 && python client/test_cluster.py; scripts/cluster.sh stop
```

**Limitations**

* `COPY` and `MOVE` of node keys are not supported. The graph is per server, not per database.
* Recall depends on each shard's own graph. Each shard searches with `ef = max(EFCON, K)`.
* `FLUSHALL`/`FLUSHDB` drop index definitions as well as nodes.


## Command Reference
<a id="markdown-command-reference" name="command-reference"></a>

### HNSW.NEW
<a id="markdown-hnsw.new" name="hnsw.new"></a>
#### Format
```
HNSW.NEW {index} [DIM {data_dim}] [M {m}] [EFCON {ef_construction}]
```
#### Description
Creates an HNSW index on the receiving node. Repeating it with the same parameters is a no-op. In a cluster, send it to every primary.
#### Example
```
HNSW.NEW foo DIM 128 M 5 EFCON 200
```
#### Parameters
* **index**: required, name of the new index.
* **DIM**: required, dimensionality of the data.
* **M**: optional, algorithm parameter for the number of neighbors to select for each node.
* **EFCON**: optional, algorithm parameter for the size of the dynamic candidate list.
#### Complexity
O(1)
#### Returns
OK or an error

### HNSW.GET
<a id="markdown-hnsw.get" name="hnsw.get"></a>
#### Format
```
HNSW.GET {index}
```
#### Description
Retrieves an HNSW index as seen by the receiving node. `node_count` counts that node's vectors only.
#### Example
```
HNSW.GET foo
```
#### Parameters
* **index**: required, name of the index.
#### Complexity
O(1)
#### Returns
**Array Reply** key-value pairs of index attributes

### HNSW.DEL
<a id="markdown-hnsw.del" name="hnsw.del"></a>
#### Format
```
HNSW.DEL {index}
```
#### Description
Deletes an HNSW index and all of its node keys on the receiving node. In a cluster, send it to every primary.
#### Example
```
HNSW.DEL foo
```
#### Parameters
* **index**: required, name of the index.
#### Complexity
O(n) where n is the number of local nodes in the index
#### Returns
OK or an error

### HNSW.NODE.ADD
<a id="markdown-hnsw.node.add" name="hnsw.node.add"></a>
#### Format
```
HNSW.NODE.ADD {index} {key} [DATA {dim} {...data}]
```
#### Description
Stores a vector at `key` and adds it to the index. If `key` already holds a vector, the vector is replaced. If `key` holds another type, the command returns an error.
#### Example
```
HNSW.NODE.ADD foo bar DATA 4 1.0 1.0 1.0 1.0
```
#### Parameters
* **index**: required, name of the index
* **key**: required, the key that stores the node. It also determines the cluster slot.
* **DATA**: required, dimensionality followed by a space separated vector of data. Total entries must match `DIM` of index
#### Complexity
O(log(n)) where n is the number of nodes in the index
#### Returns
OK or an error

### HNSW.NODE.GET
<a id="markdown-hnsw.node.get" name="hnsw.node.get"></a>
#### Format
```
HNSW.NODE.GET {index} {key}
```
#### Description
Retrieves an element from the index: its data and its links in the local graph
#### Example
```
HNSW.NODE.GET foo bar
```
#### Parameters
* **index**: required, name of the index
* **key**: required, the key that stores the node
#### Complexity
O(1)
#### Returns
**Array Reply** key-value pairs of node attributes

### HNSW.NODE.DEL
<a id="markdown-hnsw.node.del" name="hnsw.node.del"></a>
#### Format
```
HNSW.NODE.DEL {index} {key}
```
#### Description
Deletes the node key and removes it from the index. A plain `DEL {key}` does the same.
#### Example
```
HNSW.NODE.DEL foo bar
```
#### Parameters
* **index**: required, name of the index
* **key**: required, the key that stores the node
#### Complexity
O(log(n)) where n is the number of nodes in the index
#### Returns
OK or an error

### HNSW.SEARCH
<a id="markdown-hnsw.search" name="hnsw.search"></a>
#### Format
```
HNSW.SEARCH {index} [K {k}] [QUERY {dim} {...data}] [LOCAL]
```
#### Description
Searches the index for the K nearest elements to the query. In a cluster, the receiving node searches every shard and merges the results (see [Redis Cluster](#redis-cluster)). With `LOCAL`, it searches only the receiving node's graph.
#### Example
```
HNSW.SEARCH foo K 5 QUERY 4 0.0 0.0 0.0 0.0
```
#### Parameters
* **index**: required, name of the index
* **K**: required, number of nearest neighbors to return
* **QUERY**: required, dimensionality followed by space separated vector of query data. Total entries must match `DIM` of index
* **LOCAL**: optional, search only the receiving node's part of the index
#### Complexity
O(log(n)) where n is the number of nodes in the index
#### Returns
**Array Reply** where the first element is the number of results, followed by key-value pairs of similiarity and returned node key.
