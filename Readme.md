[![licence](https://img.shields.io/github/license/zhao-lang/redis_hnsw.svg)](https://github.com/zhao-lang/redis_hnsw/blob/master/LICENSE)
[![release](https://img.shields.io/github/v/release/zhao-lang/redis_hnsw.svg)](https://github.com/zhao-lang/redis_hnsw/releases/latest)
[![rust](https://github.com/zhao-lang/redis_hnsw/workflows/Rust/badge.svg)](https://github.com/zhao-lang/redis_hnsw/actions?query=workflow%3ARust)

# HNSW for Redis
<a id="markdown-hnsw-for-redis" name="hnsw-for-redis"></a>

`redis_hnsw` is a Hierarchical Navigable Small World (HNSW) implementation for Redis. Based on the paper [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320). Currently only supports Euclidean distance, Hamming distance forthcoming.

<!-- TOC -->
## Table of Contents

- [Getting Started](#getting-started)
    - [Redis commands](#redis-commands)
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

> :warning: **requires nightly rust**

Build the module - `cargo build`

Load the module - `redis-server --loadmodule ./target/<build_mode>/libredis_hnsw.<dylib|so>`

### Redis commands
<a id="markdown-redis-commands" name="redis-commands"></a>
Creating a new index - `hnsw.new {index_name} [DIM {data_dim}] [M {m}] [EFCON {ef_construction}]`

Add nodes - `hnsw.node.add {index_name} {node_name} [DATA {dim} {...data}]`

Delete nodes - `hnsw.node.del {index_name} {node_name}`

Search KNN - `hnsw.search {index_name} [K {k}] [DATA {dim} {...data}]`


## Command Reference
<a id="markdown-command-reference" name="command-reference"></a>

### HNSW.NEW
<a id="markdown-hnsw.new" name="hnsw.new"></a>
#### Format
```
HNSW.NEW {index} [DIM {data_dim}] [M {m}] [EFCON {ef_construction}]
```
#### Description
Creates an HNSW index 
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
Retrieves an HNSW index 
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
Deletes an HNSW index 
#### Example
```
HNSW.DEL foo
```
#### Parameters
* **index**: required, name of the index.
#### Complexity
O(1)
#### Returns
OK or an error

### HNSW.NODE.ADD
<a id="markdown-hnsw.node.add" name="hnsw.node.add"></a>
#### Format
```
HNSW.NODE.ADD {index} {node} [DATA {dim} {...data}]
```
#### Description
Adds an element to the index 
#### Example
```
HNSW.NODE.ADD foo bar DATA 4 1.0 1.0 1.0 1.0
```
#### Parameters
* **index**: required, name of the index
* **node**: required, name of the new node
* **DATA**: required, dimensionality followed by a space separated vector of data. Total entries must match `DIM` of index
#### Complexity
O(log(n)) where n is the number of nodes in the index
#### Returns
OK or an error

### HNSW.NODE.GET
<a id="markdown-hnsw.node.get" name="hnsw.node.get"></a>
#### Format
```
HNSW.NODE.GET {index} {node}
```
#### Description
Retrieves an element from the index 
#### Example
```
HNSW.NODE.GET foo bar
```
#### Parameters
* **index**: required, name of the index
* **node**: required, name of the node
#### Complexity
O(1)
#### Returns
**Array Reply** key-value pairs of node attributes

### HNSW.NODE.DEL
<a id="markdown-hnsw.node.del" name="hnsw.node.del"></a>
#### Format
```
HNSW.NODE.DEL {index} {node}
```
#### Description
Removes an element from the index 
#### Example
```
HNSW.NODE.DEL foo bar
```
#### Parameters
* **index**: required, name of the index
* **node**: required, name of the node
#### Complexity
O(log(n)) where n is the number of nodes in the index
#### Returns
OK or an error

### HNSW.SEARCH
<a id="markdown-hnsw.search" name="hnsw.search"></a>
#### Format
```
HNSW.SEARCH {index} [K {k}] [QUERY {dim} {...data}]
```
#### Description
Search the index for the K nearest elements to the query
#### Example
```
HNSW.SEARCH foo K 5 QUERY 4 0.0 0.0 0.0 0.0
```
#### Parameters
* **index**: required, name of the index
* **K**: required, number of nearest neighbors to return
* **DATA**: required, dimensionality followed by space separated vector of query data. Total entries must match `DIM` of index
#### Complexity
O(log(n)) where n is the number of nodes in the index
#### Returns
**Array Reply** where the first element is the number of results, followed by key-value pairs of similiarity and returned node key.
