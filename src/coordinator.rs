//! In-module coordinator for cluster-wide `HNSW.SEARCH`.
//!
//! The receiving node searches its own graph on the main thread, blocks the
//! client, and hands the rest to a background thread. That thread sends
//! `HNSW.SEARCH ... LOCAL` to every other shard's primary over plain RESP
//! connections (pooled), merges all per-shard top-k lists into the global
//! top-k, and unblocks the client with the result.

use redis_module::{Context, RedisError, RedisValue, raw};
use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_int};
use std::io::{self, BufRead, BufReader, Write};
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

/// Connect, read and write timeout for each remote shard.
pub const TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub sim: f64,
    pub name: String,
}

/// Credentials for the other nodes: the replication credentials
/// (`masteruser` / `masterauth`), which a cluster shares between its nodes.
#[derive(Clone, Debug, Default)]
pub struct Auth {
    pub user: Option<String>,
    pub password: Option<String>,
}

/// Merge per-shard top-k lists into the global top-k. A key seen on two
/// shards (mid-migration) is counted once, with its best similarity.
pub fn merge(lists: impl IntoIterator<Item = Vec<Hit>>, k: usize) -> Vec<Hit> {
    let mut best: HashMap<String, f64> = HashMap::new();
    for hit in lists.into_iter().flatten() {
        let e = best.entry(hit.name).or_insert(f64::NEG_INFINITY);
        if hit.sim > *e {
            *e = hit.sim;
        }
    }
    let mut hits: Vec<Hit> = best
        .into_iter()
        .map(|(name, sim)| Hit { sim, name })
        .collect();
    hits.sort_by(|a, b| b.sim.total_cmp(&a.sim).then_with(|| a.name.cmp(&b.name)));
    hits.truncate(k);
    hits
}

/// The `HNSW.SEARCH` reply: the number of hits, then one
/// `[similarity, <sim>, name, <key>]` array per hit.
pub fn reply(hits: &[Hit]) -> RedisValue {
    let mut reply: Vec<RedisValue> = Vec::with_capacity(hits.len() + 1);
    reply.push(hits.len().into());
    for h in hits {
        let fields: Vec<RedisValue> = vec![
            "similarity".into(),
            RedisValue::Float(h.sim),
            "name".into(),
            h.name.as_str().into(),
        ];
        reply.push(fields.into());
    }
    reply.into()
}

/// Addresses of the primaries whose shards this node does not cover. A
/// replica covers its primary's shard with its own copy of the graph.
pub fn remote_primaries(ctx: &Context) -> Result<Vec<SocketAddr>, RedisError> {
    let my_id = unsafe { raw::RedisModule_GetMyClusterID.unwrap()() };
    if my_id.is_null() {
        return Err(RedisError::Str("cluster mode is not enabled"));
    }
    let my_id = node_id(my_id);
    let me = node_info(ctx, &my_id).ok_or(RedisError::Str("cannot read own cluster node"))?;
    let my_shard = if me.primary { my_id } else { me.master_id };

    let mut numnodes: usize = 0;
    let list = unsafe { raw::RedisModule_GetClusterNodesList.unwrap()(ctx.ctx, &mut numnodes) };
    if list.is_null() {
        return Err(RedisError::Str("cannot read cluster nodes"));
    }
    let mut addrs = Vec::new();
    let mut failed = None;
    for i in 0..numnodes {
        let id = node_id(unsafe { *list.add(i) });
        let Some(node) = node_info(ctx, &id) else {
            continue;
        };
        if !node.primary || id == my_shard {
            continue;
        }
        match node.addr {
            Some(addr) => addrs.push(addr),
            None => failed = Some(String::from_utf8_lossy(&id).into_owned()),
        }
    }
    unsafe { raw::RedisModule_FreeClusterNodesList.unwrap()(list) };
    match failed {
        Some(id) => Err(RedisError::String(format!(
            "cluster node {id} has no address"
        ))),
        None => Ok(addrs),
    }
}

/// Node ids in the cluster API are exactly REDISMODULE_NODE_ID_LEN bytes,
/// without a NUL terminator.
fn node_id(ptr: *const c_char) -> Vec<u8> {
    let len = raw::REDISMODULE_NODE_ID_LEN as usize;
    unsafe { std::slice::from_raw_parts(ptr as *const u8, len) }.to_vec()
}

struct NodeInfo {
    primary: bool,
    master_id: Vec<u8>,
    addr: Option<SocketAddr>,
}

fn node_info(ctx: &Context, id: &[u8]) -> Option<NodeInfo> {
    let len = raw::REDISMODULE_NODE_ID_LEN as usize;
    // Ids come back unterminated, but this call looks them up with strlen().
    let mut id_cstr = id.to_vec();
    id_cstr.push(0);
    let mut ip = [0 as c_char; 46];
    let mut master_id = [0 as c_char; 41];
    let mut port: c_int = 0;
    let mut flags: c_int = 0;
    let ok = unsafe {
        raw::RedisModule_GetClusterNodeInfo.unwrap()(
            ctx.ctx,
            id_cstr.as_ptr() as *const c_char,
            ip.as_mut_ptr(),
            master_id.as_mut_ptr(),
            &mut port,
            &mut flags,
        )
    };
    if ok != raw::REDISMODULE_OK as c_int {
        return None;
    }
    let flags = flags as u32;
    let ip = unsafe { CStr::from_ptr(ip.as_ptr()) }.to_str().ok()?;
    let master_id: Vec<u8> = master_id[..len].iter().map(|&c| c as u8).collect();
    Some(NodeInfo {
        primary: flags & raw::REDISMODULE_NODE_MASTER != 0,
        master_id,
        addr: ip
            .parse::<IpAddr>()
            .ok()
            .filter(|_| port > 0)
            .map(|ip| SocketAddr::new(ip, port as u16)),
    })
}

/// Read `masteruser` / `masterauth` from the server config.
pub fn read_auth(ctx: &Context) -> Auth {
    let get = |name: &str| -> Option<String> {
        let v = ctx.call("CONFIG", &["GET", name]).ok()?;
        let RedisValue::Array(items) = v else {
            return None;
        };
        let value = match items.get(1)? {
            RedisValue::SimpleString(s) => s.clone(),
            RedisValue::BulkString(s) => s.clone(),
            RedisValue::BulkRedisString(s) => s.to_string_lossy(),
            RedisValue::StringBuffer(b) => String::from_utf8_lossy(b).into_owned(),
            _ => return None,
        };
        (!value.is_empty()).then_some(value)
    };
    Auth {
        user: get("masteruser"),
        password: get("masterauth"),
    }
}

/// Run a shard-local search on a remote primary.
pub fn search_remote(addr: SocketAddr, auth: &Auth, args: &[Vec<u8>]) -> Result<Vec<Hit>, String> {
    let reply = call(addr, auth, args).map_err(|e| match e.kind() {
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut => {
            format!("{addr}: timed out after {}s", TIMEOUT.as_secs())
        }
        _ => format!("{addr}: {e}"),
    })?;
    parse_search_reply(&reply).map_err(|e| format!("{addr}: {e}"))
}

// -- RESP client ----------------------------------------------------------

#[derive(Debug, PartialEq)]
pub enum Resp {
    Simple(String),
    Error(String),
    Int(i64),
    Bulk(Option<Vec<u8>>),
    Array(Option<Vec<Resp>>),
}

static POOL: LazyLock<Mutex<HashMap<SocketAddr, Vec<TcpStream>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn pooled(addr: SocketAddr) -> Option<TcpStream> {
    POOL.lock().ok()?.get_mut(&addr)?.pop()
}

fn release(addr: SocketAddr, conn: TcpStream) {
    if let Ok(mut pool) = POOL.lock() {
        let conns = pool.entry(addr).or_default();
        if conns.len() < 8 {
            conns.push(conn);
        }
    }
}

fn connect(addr: SocketAddr, auth: &Auth) -> io::Result<TcpStream> {
    let conn = TcpStream::connect_timeout(&addr, TIMEOUT)?;
    conn.set_read_timeout(Some(TIMEOUT))?;
    conn.set_write_timeout(Some(TIMEOUT))?;
    conn.set_nodelay(true)?;
    if let Some(password) = &auth.password {
        let mut args = vec![b"AUTH".to_vec()];
        if let Some(user) = &auth.user {
            args.push(user.as_bytes().to_vec());
        }
        args.push(password.as_bytes().to_vec());
        if let Resp::Error(e) = roundtrip(&conn, &args)? {
            return Err(io::Error::other(format!("AUTH failed: {e}")));
        }
    }
    Ok(conn)
}

/// Send one command, reusing a pooled connection when there is one. A pooled
/// connection may have been closed by the remote (restart, idle timeout), so
/// a broken connection is retried once on a fresh one. A timeout is not
/// retried: the shard is slow, not gone. A connection that failed in any way
/// is dropped, never pooled, so a late reply cannot be read by the next call.
fn call(addr: SocketAddr, auth: &Auth, args: &[Vec<u8>]) -> io::Result<Resp> {
    if let Some(conn) = pooled(addr) {
        match roundtrip(&conn, args) {
            Ok(reply) => {
                release(addr, conn);
                return Ok(reply);
            }
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                return Err(e);
            }
            Err(_) => {}
        }
    }
    let conn = connect(addr, auth)?;
    let reply = roundtrip(&conn, args)?;
    release(addr, conn);
    Ok(reply)
}

fn roundtrip(conn: &TcpStream, args: &[Vec<u8>]) -> io::Result<Resp> {
    let mut w = conn;
    w.write_all(&encode(args))?;
    read_resp(&mut BufReader::new(conn))
}

pub fn encode(args: &[Vec<u8>]) -> Vec<u8> {
    let mut out = format!("*{}\r\n", args.len()).into_bytes();
    for a in args {
        out.extend_from_slice(format!("${}\r\n", a.len()).as_bytes());
        out.extend_from_slice(a);
        out.extend_from_slice(b"\r\n");
    }
    out
}

fn read_line(r: &mut impl BufRead) -> io::Result<String> {
    let mut line = Vec::new();
    r.read_until(b'\n', &mut line)?;
    if !line.ends_with(b"\r\n") {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "truncated reply",
        ));
    }
    line.truncate(line.len() - 2);
    String::from_utf8(line).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn parse_int(s: &str) -> io::Result<i64> {
    s.parse()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub fn read_resp(r: &mut impl BufRead) -> io::Result<Resp> {
    let line = read_line(r)?;
    let (kind, rest) = line.split_at_checked(1).unwrap_or(("", ""));
    Ok(match kind {
        "+" => Resp::Simple(rest.to_owned()),
        "-" => Resp::Error(rest.to_owned()),
        ":" => Resp::Int(parse_int(rest)?),
        "$" => match parse_int(rest)? {
            n if n < 0 => Resp::Bulk(None),
            n => {
                let mut buf = vec![0; n as usize + 2];
                r.read_exact(&mut buf)?;
                buf.truncate(n as usize);
                Resp::Bulk(Some(buf))
            }
        },
        "*" => match parse_int(rest)? {
            n if n < 0 => Resp::Array(None),
            n => Resp::Array(Some(
                (0..n).map(|_| read_resp(r)).collect::<io::Result<_>>()?,
            )),
        },
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unexpected reply: {line}"),
            ));
        }
    })
}

fn text(r: &Resp) -> Option<String> {
    match r {
        Resp::Simple(s) => Some(s.clone()),
        Resp::Bulk(Some(b)) => String::from_utf8(b.clone()).ok(),
        _ => None,
    }
}

pub fn parse_search_reply(reply: &Resp) -> Result<Vec<Hit>, String> {
    let items = match reply {
        Resp::Error(e) => return Err(e.clone()),
        Resp::Array(Some(items)) => items,
        other => return Err(format!("unexpected reply: {other:?}")),
    };
    items
        .iter()
        .skip(1)
        .map(|item| {
            let Resp::Array(Some(fields)) = item else {
                return Err(format!("unexpected hit: {item:?}"));
            };
            let mut sim = None;
            let mut name = None;
            for pair in fields.chunks(2) {
                match (pair.first().and_then(text).as_deref(), pair.get(1)) {
                    (Some("similarity"), Some(v)) => sim = text(v).and_then(|s| s.parse().ok()),
                    (Some("name"), Some(v)) => name = text(v),
                    _ => {}
                }
            }
            match (sim, name) {
                (Some(sim), Some(name)) => Ok(Hit { sim, name }),
                _ => Err(format!("unexpected hit: {item:?}")),
            }
        })
        .collect()
}
