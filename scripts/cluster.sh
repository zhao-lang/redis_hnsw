#!/usr/bin/env bash
# Start or stop a local Redis Cluster with the module loaded.
#
#   scripts/cluster.sh start [primaries] [replicas-per-primary]   (default 3 1)
#   scripts/cluster.sh stop
#
# Nodes listen on ports 7000+, data lives in $CLUSTER_DIR (default
# ./target/cluster). Build the module first: `cargo build`. Set PASSWORD to
# require a password (requirepass + masterauth) on every node.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${CLUSTER_DIR:-$ROOT/target/cluster}"
MODULE="${MODULE:-$ROOT/target/debug/libredis_hnsw.so}"
BASE_PORT="${BASE_PORT:-7000}"
PASSWORD="${PASSWORD:-}"
CLI=(redis-cli)
AUTH_ARGS=()
if [ -n "$PASSWORD" ]; then
    CLI+=(-a "$PASSWORD" --no-auth-warning)
    AUTH_ARGS=(--requirepass "$PASSWORD" --masterauth "$PASSWORD")
fi

stop() {
    for pidfile in "$DIR"/*/redis.pid; do
        [ -f "$pidfile" ] || continue
        kill "$(cat "$pidfile")" 2>/dev/null || true
    done
    sleep 0.5
    rm -rf "$DIR"
}

start() {
    local primaries="${1:-3}" replicas="${2:-1}"
    local total=$((primaries * (replicas + 1)))
    stop
    mkdir -p "$DIR"
    local addrs=()
    for ((i = 0; i < total; i++)); do
        local port=$((BASE_PORT + i))
        mkdir -p "$DIR/$port"
        redis-server --port "$port" --dir "$DIR/$port" --daemonize yes \
            --pidfile "$DIR/$port/redis.pid" --logfile "$DIR/$port/redis.log" \
            --cluster-enabled yes --cluster-config-file nodes.conf \
            --appendonly no --save "" --enable-debug-command yes "${AUTH_ARGS[@]}" --loadmodule "$MODULE"
        addrs+=("127.0.0.1:$port")
    done
    for addr in "${addrs[@]}"; do
        until "${CLI[@]}" -p "${addr#*:}" ping >/dev/null 2>&1; do sleep 0.1; done
    done
    "${CLI[@]}" --cluster create "${addrs[@]}" --cluster-replicas "$replicas" --cluster-yes >/dev/null
    until "${CLI[@]}" -p "$BASE_PORT" cluster info | grep -q 'cluster_state:ok'; do sleep 0.2; done
    # wait for every replica's initial sync
    for addr in "${addrs[@]}"; do
        local info
        info="$("${CLI[@]}" -p "${addr#*:}" info replication)"
        if grep -q 'role:slave' <<<"$info"; then
            until "${CLI[@]}" -p "${addr#*:}" info replication | grep -q 'master_link_status:up'; do
                sleep 0.2
            done
        fi
    done
    echo "cluster up: ${addrs[*]}"
}

case "${1:-}" in
    start) shift; start "$@" ;;
    stop) stop ;;
    *) echo "usage: $0 start [primaries] [replicas] | stop" >&2; exit 1 ;;
esac
