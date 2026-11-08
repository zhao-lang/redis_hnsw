"""Client-side coordinator for redis_hnsw on Redis Cluster.

Each shard holds an HNSW graph over the vectors stored on it. This client:

* sends index-level commands (HNSW.NEW / HNSW.DEL) to every primary,
* lets the cluster route node commands by key, like any other key command,
* fans HNSW.SEARCH out to every primary and merges the per-shard top-k.

Requires redis-py >= 5 (``pip install redis``).

    from redis.cluster import RedisCluster
from redis.exceptions import AskError, MovedError
    from hnsw_cluster import HnswCluster

    hc = HnswCluster(RedisCluster(host="127.0.0.1", port=7000))
    hc.create_index("docs", dim=3)
    hc.add("docs", "doc:1:vec", [0.1, 0.2, 0.3])
    hc.search("docs", [0.1, 0.2, 0.3], k=5)   # [("doc:1:vec", -0.0), ...]
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from redis.cluster import RedisCluster


class HnswCluster:
    def __init__(self, rc: RedisCluster, max_workers: int = 16):
        self.rc = rc
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def _on_primaries(self, *args):
        """Run a keyless command on every primary, in parallel."""
        primaries = self.rc.get_primaries()
        futures = [
            self._pool.submit(self.rc.execute_command, *args, target_nodes=node)
            for node in primaries
        ]
        return [f.result() for f in futures]

    def _on_key(self, key: str, *args):
        """Run a command on the shard that owns `key`.

        Routed here rather than by RedisCluster.execute_command, whose command
        policy lookup rejects module commands named with more than one dot
        (``HNSW.NODE.ADD``).
        """
        for _ in range(5):
            node = self.rc.get_node_from_key(key)
            try:
                return node.redis_connection.execute_command(*args)
            except MovedError:
                self.rc.nodes_manager.initialize()
            except AskError as e:
                # Slot is mid-migration and the key is already on the target.
                target = self.rc.get_node(host=e.host, port=e.port)
                if target is None:
                    self.rc.nodes_manager.initialize()
                    continue
                pipe = target.redis_connection.pipeline(transaction=False)
                pipe.execute_command("ASKING")
                pipe.execute_command(*args)
                return pipe.execute()[1]
        raise RuntimeError(f"could not route command for key {key!r}")

    # -- index ----------------------------------------------------------

    def create_index(self, name: str, dim: int, m: int = 5, ef_construction: int = 200):
        """Define the index on every primary. Safe to repeat, e.g. after adding a shard."""
        self._on_primaries("HNSW.NEW", name, "DIM", dim, "M", m, "EFCON", ef_construction)

    def delete_index(self, name: str):
        """Drop the index and all of its nodes on every primary."""
        self._on_primaries("HNSW.DEL", name)

    def node_counts(self, name: str) -> dict[str, int]:
        """Nodes per primary, keyed by "host:port"."""
        counts = {}
        for node in self.rc.get_primaries():
            reply = self.rc.execute_command("HNSW.GET", name, target_nodes=node)
            info = dict(zip(reply[::2], reply[1::2]))
            counts[node.name] = int(info[_key(info, "node_count")])
        return counts

    # -- nodes (routed by key) ------------------------------------------

    def add(self, index: str, key: str, vector: Sequence[float]):
        """Store `vector` at `key` and index it. Replaces any previous vector."""
        self._on_key(key, "HNSW.NODE.ADD", index, key, "DATA", len(vector), *vector)

    def get(self, index: str, key: str) -> list[float]:
        reply = self._on_key(key, "HNSW.NODE.GET", index, key)
        info = dict(zip(reply[::2], reply[1::2]))
        return [float(x) for x in info[_key(info, "data")]]

    def delete(self, index: str, key: str):
        self._on_key(key, "HNSW.NODE.DEL", index, key)

    # -- search ---------------------------------------------------------

    def search(self, index: str, query: Sequence[float], k: int = 5) -> list[tuple[str, float]]:
        """K nearest neighbours across the cluster, as (key, similarity) pairs.

        Each shard returns its own top-k, so the global top-k is among their
        union. A key seen twice (mid-migration) is counted once.
        """
        replies = self._on_primaries("HNSW.SEARCH", index, "K", k, "QUERY", len(query), *query)
        best: dict[str, float] = {}
        for reply in replies:
            for item in reply[1:]:
                fields = dict(zip(item[::2], item[1::2]))
                name = _str(fields[_key(fields, "name")])
                sim = float(fields[_key(fields, "similarity")])
                if sim > best.get(name, float("-inf")):
                    best[name] = sim
        return sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:k]


def _str(v) -> str:
    return v.decode() if isinstance(v, bytes) else v


def _key(d: dict, name: str):
    """Field lookup that works with and without decode_responses."""
    return name if name in d else name.encode()
