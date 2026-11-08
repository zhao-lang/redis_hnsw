use crate::coordinator::*;
use std::io::Cursor;

fn hit(name: &str, sim: f64) -> Hit {
    Hit {
        sim,
        name: name.to_owned(),
    }
}

#[test]
fn merge_takes_global_top_k() {
    let a = vec![hit("a", -1.0), hit("b", -3.0)];
    let b = vec![hit("c", -2.0), hit("d", -0.5)];
    let merged = merge([a, b], 3);
    assert_eq!(merged, vec![hit("d", -0.5), hit("a", -1.0), hit("c", -2.0)]);
}

#[test]
fn merge_dedupes_keeping_best() {
    // a key on two shards mid-migration counts once
    let merged = merge(
        [vec![hit("a", -2.0)], vec![hit("a", -1.0), hit("b", -3.0)]],
        5,
    );
    assert_eq!(merged, vec![hit("a", -1.0), hit("b", -3.0)]);
}

#[test]
fn encode_command() {
    let cmd = encode(&[b"PING".to_vec(), b"hi".to_vec()]);
    assert_eq!(cmd, b"*2\r\n$4\r\nPING\r\n$2\r\nhi\r\n");
}

#[test]
fn parse_resp_values() {
    let mut r = Cursor::new(b"*4\r\n+OK\r\n:-7\r\n$-1\r\n$3\r\na\r\n\r\n".to_vec());
    assert_eq!(
        read_resp(&mut r).unwrap(),
        Resp::Array(Some(vec![
            Resp::Simple("OK".into()),
            Resp::Int(-7),
            Resp::Bulk(None),
            Resp::Bulk(Some(b"a\r\n".to_vec())),
        ]))
    );
    let mut r = Cursor::new(b"$5\r\nab".to_vec());
    assert!(read_resp(&mut r).is_err());
}

#[test]
fn parse_search() {
    let raw = b"*3\r\n:2\r\n\
        *4\r\n$10\r\nsimilarity\r\n$4\r\n-0.5\r\n$4\r\nname\r\n$1\r\nx\r\n\
        *4\r\n$10\r\nsimilarity\r\n$2\r\n-2\r\n$4\r\nname\r\n$1\r\ny\r\n";
    let reply = read_resp(&mut Cursor::new(raw.to_vec())).unwrap();
    assert_eq!(
        parse_search_reply(&reply).unwrap(),
        vec![hit("x", -0.5), hit("y", -2.0)]
    );

    let err = read_resp(&mut Cursor::new(b"-ERR nope\r\n".to_vec())).unwrap();
    assert_eq!(parse_search_reply(&err).unwrap_err(), "ERR nope");
}
