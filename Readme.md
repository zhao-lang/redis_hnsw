# HNSW for Redis

`redis_hnsw` is a Hierarchical Navigable Small World (HNSW) implementation for Redis. Based on the paper [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320). Currently only supports Euclidean distance, Hamming distance forthcoming.

## Getting Started

Build the module - `cargo build`

Load the module - `redis-server --loadmodule ./target/<build_mode>/libredis_hnsw.<dylib|so>`

### Redis commands
Creating a new index - `hnsw.new <index_name> <data_dim_> <M> <ef_construction>`

Add nodes - `hnsw.node.add <index_name> <node_name> <...data>`

Delete nodes - `hnsw.node.del <index_name> <node_name>`

Search KNN - `hnsw.search <index_name> <k> <..data>`


## Command Reference

### HNSW.NEW
#### Format
```
HNSW.NEW {index} [DATA_DIM] [M] [EF_CONSTRUCTION]
```
#### Description
Creates an HNSW index 
#### Example
```
HNSW.NEW foo 128 5 200
```
#### Parameters
* **index**: name of the new index.
* **DATA_DIM**: dimensionality of the data.
* **M**: algorithm parameter for the number of neighbors to select for each node.
* **EF_CONSTRUCTION**: algorithm parameter for the size of the dynamic candidate list.
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
* **index**: name of the index.
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
* **index**: name of the index.
#### Complexity
O(1)
#### Returns
OK or an error

### HNSW.NODE.ADD
#### Format
```
HNSW.NODE.ADD {index} {node} {...DATA}
```
#### Description
Adds an element to the index 
#### Example
```
HNSW.NODE.ADD foo bar 1.0 1.0 1.0 1.0
```
#### Parameters
* **index**: name of the index
* **node**: name of the new node
* **DATA**: space separated vector of data. Total entries must match `DATA_DIM` of index
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
* **index**: name of the index
* **node**: name of the node
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
* **index**: name of the index
* **node**: name of the node
#### Complexity
O(log(n)) where n is the number of nodes in the index
#### Returns
OK or an error

### HNSW.SEARCH
#### Format
```
HNSW.SEARCH {index} {K} {...DATA}
```
#### Description
Search the index for the K nearest elements to the query
#### Example
```
HNSW.SEARCH foo 5 0.0 0.0 0.0 0.0
```
#### Parameters
* **index**: name of the index
* **K**: number of nearest neighbors to return
* **DATA**: space separated vector of query data. Total entries must match `DATA_DIM` of index
#### Complexity
O(log(n)) where n is the number of nodes in the index
#### Returns
**Array Reply** where the first element is the number of results, followed by key-value pairs of similiarity and returned node key.
