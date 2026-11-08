"""End-to-end test of redis_hnsw on a local Redis Cluster.

    cargo build
    scripts/cluster.sh start 3 1
    python client/test_cluster.py
    scripts/cluster.sh stop

Set REDIS_PASSWORD (for both commands) to run against a cluster that requires
a password; the module's coordinator then authenticates with masterauth.
"""

from __future__ import annotations

import os
import random
import socket
import subprocess
import threading
import time

from redis import Redis
from redis.cluster import RedisCluster
from redis.exceptions import ResponseError

from hnsw_cluster import HnswCluster, _hits as hc_hits

DIM = 8
N = 400
K = 10
INDEX = "test"
PASSWORD = os.environ.get("REDIS_PASSWORD") or None
CLI = ["redis-cli"] + (["-a", PASSWORD, "--no-auth-warning"] if PASSWORD else [])


def R(port: int) -> Redis:
    return Redis(port=port, password=PASSWORD)


def RC() -> RedisCluster:
    return RedisCluster(host="127.0.0.1", port=7000, password=PASSWORD)


def brute_force(vectors: dict[str, list[float]], q: list[float], k: int) -> list[str]:
    dist = {key: sum((a - b) ** 2 for a, b in zip(v, q)) for key, v in vectors.items()}
    return sorted(dist, key=dist.get)[:k]


def recall(hc: HnswCluster, vectors, queries, fanout: str = "server") -> float:
    hits = 0
    for q in queries:
        got = {key for key, _ in hc.search(INDEX, q, K, fanout=fanout)}
        hits += len(got & set(brute_force(vectors, q, K)))
    return hits / (K * len(queries))


def wait_for(cond, what: str, timeout: float = 10.0):
    deadline = time.time() + timeout
    while not cond():
        if time.time() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.1)


def shard_consistent(rc: RedisCluster, hc: HnswCluster) -> bool:
    """Every primary's graph holds exactly the vector keys it stores."""
    counts = hc.node_counts(INDEX)
    for node in rc.get_primaries():
        if counts[node.name] != rc.execute_command("DBSIZE", target_nodes=node):
            return False
    return True


def node_id(port: int) -> str:
    return R(port).execute_command("CLUSTER", "MYID").decode()


def main():
    rng = random.Random(42)
    rc = RC()
    hc = HnswCluster(rc)
    rc.flushall(target_nodes=RedisCluster.PRIMARIES)

    hc.create_index(INDEX, dim=DIM, m=8, ef_construction=100)
    hc.create_index(INDEX, dim=DIM, m=8, ef_construction=100)  # idempotent

    vectors = {f"vec:{i}": [rng.random() for _ in range(DIM)] for i in range(N)}
    for key, v in vectors.items():
        hc.add(INDEX, key, v)
    queries = [[rng.random() for _ in range(DIM)] for _ in range(30)]

    counts = hc.node_counts(INDEX)
    print("nodes per shard:", counts)
    assert sum(counts.values()) == N
    assert all(c > 0 for c in counts.values()), "keys should spread over shards"
    assert shard_consistent(rc, hc)

    for fanout in ("server", "client"):
        r = recall(hc, vectors, queries, fanout)
        print(f"recall@{K} ({fanout} fan-out): {r:.3f}")
        assert r >= 0.95

    # in-module coordinator: any primary gives the same answer as merging
    # LOCAL searches on the client
    q = queries[0]
    args = ("HNSW.SEARCH", INDEX, "K", K, "QUERY", DIM, *q)
    expected = hc.search(INDEX, q, K, fanout="client")
    for node in rc.get_primaries():
        got = hc_hits(rc.execute_command(*args, target_nodes=node))
        assert got == expected, (node.name, got, expected)
        # LOCAL only returns keys stored on that node
        local = hc_hits(rc.execute_command(*args, "LOCAL", target_nodes=node))
        assert local and all(rc.get_node_from_key(k).name == node.name for k, _ in local)
    # cannot block inside MULTI
    pipe = R(7000).pipeline(transaction=True)
    pipe.execute_command(*args)
    res = pipe.execute(raise_on_error=False)[0]
    assert isinstance(res, ResponseError) and "MULTI" in str(res), res

    # exact match is found and ranked first
    key = "vec:7"
    assert hc.search(INDEX, vectors[key], 1)[0][0] == key
    assert all(abs(a - b) < 1e-6 for a, b in zip(hc.get(INDEX, key), vectors[key]))  # f32

    # removal through every route: HNSW.NODE.DEL, DEL, expiry, overwrite by SET
    hc.delete(INDEX, "vec:1")
    rc.delete("vec:2")
    rc.pexpire("vec:3", 1)
    rc.set("vec:4", "not a vector")
    time.sleep(0.2)
    rc.get("vec:3")  # trigger lazy expiry
    for k in ("vec:1", "vec:2", "vec:3", "vec:4"):
        assert k not in dict(hc.search(INDEX, vectors[k], K)), k
        del vectors[k]
    rc.delete("vec:4")
    wait_for(lambda: shard_consistent(rc, hc), "consistency after deletes")

    # overwrite with a new vector moves it in the graph
    vectors["vec:5"] = [5.0] * DIM
    hc.add(INDEX, "vec:5", vectors["vec:5"])
    assert hc.search(INDEX, [5.0] * DIM, 1)[0][0] == "vec:5"

    # RENAME keeps it indexed under the new name (same slot via hash tag)
    v = [rng.random() for _ in range(DIM)]
    hc.add(INDEX, "{r}:old", v)
    rc.rename("{r}:old", "{r}:new")
    vectors["{r}:new"] = v
    top = hc.search(INDEX, v, 1)[0][0]
    assert top == "{r}:new", top
    wait_for(lambda: shard_consistent(rc, hc), "consistency after rename")

    # slot migration: move 3000 slots from the first primary to the second
    before = hc.node_counts(INDEX)
    subprocess.run(
        [*CLI, "--cluster", "reshard", "127.0.0.1:7000",
         "--cluster-from", node_id(7000), "--cluster-to", node_id(7001),
         "--cluster-slots", "3000", "--cluster-yes"],
        check=True, capture_output=True,
    )
    rc = RC()
    hc = HnswCluster(rc)
    after = hc.node_counts(INDEX)
    print("after reshard:", after)
    assert after != before
    assert sum(after.values()) == len(vectors)
    assert shard_consistent(rc, hc)
    r = recall(hc, vectors, queries)
    print(f"recall@{K} after reshard: {r:.3f}")
    assert recall(hc, vectors, queries, "client") == r
    assert r >= 0.95

    # replicas build their own graph from the replication stream
    for primary in rc.get_primaries():
        replica_port = None
        for line in R(primary.port).info("replication").items():
            if line[0].startswith("slave") and isinstance(line[1], dict):
                replica_port = line[1]["port"]
        assert replica_port, f"no replica for {primary.name}"
        replica = R(replica_port)
        expected = hc.node_counts(INDEX)[primary.name]

        def replica_count():
            try:
                info = replica.execute_command("HNSW.GET", INDEX)
            except ResponseError:  # still in its initial sync
                return -1
            return int(dict(zip(info[::2], info[1::2]))[b"node_count"])

        wait_for(lambda: replica_count() == expected, f"replica {replica_port}")

        # a replica coordinates too, covering its primary's shard itself
        pipe = replica.pipeline(transaction=False)
        pipe.execute_command("READONLY")
        pipe.execute_command(*args)
        assert hc_hits(pipe.execute()[1]) == hc.search(INDEX, q, K, fanout="client")

    # restart from RDB: definitions (including an empty index) and graphs survive
    hc.create_index("empty", dim=2)
    for node in rc.get_primaries():
        rc.execute_command("DEBUG", "RELOAD", target_nodes=node)
    assert hc.node_counts("empty") == {n.name: 0 for n in rc.get_primaries()}
    assert sum(hc.node_counts(INDEX).values()) == len(vectors)
    assert recall(hc, vectors, queries) >= 0.95

    # FLUSHALL drops everything; HNSW.DEL removes nodes on every shard
    hc.delete_index(INDEX)
    assert sum(rc.dbsize(target_nodes=n) for n in rc.get_primaries()) == 0

    # a shard that does not answer fails the search instead of hanging it
    hc.create_index("down", dim=2)
    sleeping = Redis(port=7002, password=PASSWORD, socket_timeout=30)
    sleeper = threading.Thread(target=lambda: sleeping.execute_command("DEBUG", "SLEEP", 7))
    sleeper.start()

    def asleep():
        # raw socket: redis-py would retry a timed-out PING with backoff
        with socket.create_connection(("127.0.0.1", 7002), timeout=0.2) as sock:
            sock.sendall(b"PING\r\n")
            try:
                sock.recv(64)
                return False
            except TimeoutError:
                return True

    wait_for(asleep, "shard to stall")
    start = time.time()
    try:
        # longer than the module's 5s shard timeout, so redis-py does not
        # time out first and retry the search
        client = Redis(port=7000, password=PASSWORD, socket_timeout=30)
        client.execute_command("HNSW.SEARCH", "down", "QUERY", 2, 0, 0)
        raise AssertionError("search should fail while a shard sleeps")
    except ResponseError as e:
        assert "failed on shard" in str(e), e
    print(f"stalled shard fails the search after {time.time() - start:.1f}s")
    sleeper.join()
    hc.delete_index("down")

    print("OK")


if __name__ == "__main__":
    main()
