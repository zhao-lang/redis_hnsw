# HNSW for Redis

`redis_hnsw` is a Hierarchical Navigable Small World (HNSW) implementation for Redis. Based on the paper [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320). Currently only supports Euclidean distance, Hamming distance forthcoming.

## Getting Started

Build the module - `cargo build`

Load the module - `redis-server --loadmodule ./target/<build_mode>/libredis_hnsw.<dylib|so>`

### Redis commands
Creating a new index - `hnsw.new {index_name} [DIM {data_dim}] [M {m}] [EFCON {ef_construction}]`

Add nodes - `hnsw.node.add {index_name} {node_name} [DATA {dim} {...data}]`

Delete nodes - `hnsw.node.del {index_name} {node_name}`

Search KNN - `hnsw.search {index_name} [K {k}] [DATA {dim} {...data}]`


## Command Reference

### HNSW.NEW
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
