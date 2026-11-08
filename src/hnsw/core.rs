// Node<T> hashes by its name, which never changes after creation, so using it
// as a HashSet key is sound despite the RwLock inside.
#![allow(clippy::mutable_key_type)]

use super::metrics;

use ordered_float::OrderedFloat;
use rand::RngExt;
use std::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd, Reverse, min};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::convert::From;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak};

/// Floating point types usable in the index: full float math plus the total
/// ordering `OrderedFloat` requires.
pub trait Float: num_traits::Float + num_traits::float::FloatCore {}
impl<F: num_traits::Float + num_traits::float::FloatCore> Float for F {}

struct SelectParams {
    m: usize,
    lc: usize,
    extend_candidates: bool,
    keep_pruned_connections: bool,
}

#[derive(Debug)]
pub enum HNSWError {
    Str(&'static str),
    String(String),
}

impl From<&'static str> for HNSWError {
    fn from(s: &'static str) -> Self {
        HNSWError::Str(s)
    }
}

impl From<String> for HNSWError {
    fn from(s: String) -> Self {
        HNSWError::String(s)
    }
}

impl HNSWError {
    pub fn error_string(&self) -> String {
        match self {
            HNSWError::Str(s) => (*s).to_owned(),
            HNSWError::String(s) => s.clone(),
        }
    }
}

pub struct SearchResult<T: Float, R: Float> {
    pub sim: OrderedFloat<R>,
    pub name: String,
    pub data: Vec<T>,
}

impl<T: Float, R: Float> SearchResult<T, R> {
    fn new(sim: OrderedFloat<R>, name: &str, data: &[T]) -> Self {
        SearchResult {
            sim,
            name: name.to_owned(),
            data: data.to_vec(),
        }
    }
}

impl<T, R> fmt::Debug for SearchResult<T, R>
where
    T: Float + fmt::Debug,
    R: Float + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "sim: {:?}, \
             name: {:?}, \
             data: {:?}{}",
            self.sim,
            self.name,
            if self.data.len() > 10 {
                &self.data[..10]
            } else {
                &self.data[..]
            },
            if self.data.len() > 10 {
                let more = format!(" + {} more", self.data.len() - 10);
                more
            } else {
                "".to_owned()
            }
        )
    }
}

type NodeRef<T> = Arc<RwLock<_Node<T>>>;
type NodeRefWeak<T> = Weak<RwLock<_Node<T>>>;

#[derive(Clone)]
pub struct _Node<T: Float> {
    pub name: String,
    pub data: Vec<T>,
    /// Outgoing links per layer. As in the paper the graph is directed: a
    /// node's list is capped at Mmax (Mmax0 on layer 0) and pruning it does not
    /// touch the lists of the nodes it drops.
    pub neighbors: Vec<Vec<NodeWeak<T>>>,
    /// Incoming links per layer (nodes whose `neighbors` contain this node).
    /// Not part of the paper and not persisted: it is rebuilt on load and is
    /// only needed so deletion can find and repair every node linking here.
    pub in_neighbors: Vec<Vec<NodeWeak<T>>>,
}

impl<T> fmt::Debug for _Node<T>
where
    T: Float + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "name: {}, \
             data: {:?}, \
             neighbors: {:?}",
            self.name,
            self.data,
            self.neighbors
                .iter()
                .map(|l| {
                    l.iter()
                        .map(|n| n.upgrade().read().name.to_owned())
                        .collect::<Vec<String>>()
                })
                .collect::<Vec<Vec<String>>>(),
        )
    }
}

fn ensure_level<T: Float>(lists: &mut Vec<Vec<NodeWeak<T>>>, level: usize) {
    if lists.len() <= level {
        lists.resize_with(level + 1, Vec::new);
    }
}

fn push_unique<T: Float>(list: &mut Vec<NodeWeak<T>>, n: NodeWeak<T>) {
    if !list.contains(&n) {
        list.push(n);
    }
}

fn remove_link<T: Float>(lists: &mut [Vec<NodeWeak<T>>], level: usize, n: &NodeWeak<T>) {
    if let Some(list) = lists.get_mut(level) {
        list.retain(|x| x != n);
    }
}

#[derive(Debug, Clone)]
pub struct NodeWeak<T: Float>(pub NodeRefWeak<T>);

impl<T: Float> PartialEq for NodeWeak<T> {
    fn eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.0, &other.0)
    }
}

impl<T: Float> Eq for NodeWeak<T> {}

impl<T: Float> Hash for NodeWeak<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.upgrade().read().name.hash(state);
    }
}

impl<T: Float> NodeWeak<T> {
    pub fn upgrade(&self) -> Node<T> {
        Node(self.0.upgrade().unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct Node<T: Float>(pub NodeRef<T>);

impl<T: Float> PartialEq for Node<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<T: Float> Eq for Node<T> {}

impl<T: Float> Hash for Node<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.read().name.hash(state);
    }
}

impl<T: Float> Node<T> {
    pub fn new(name: &str, data: &[T], capacity: usize) -> Self {
        let node = _Node {
            name: name.to_owned(),
            data: data.to_vec(),
            neighbors: Vec::with_capacity(capacity),
            in_neighbors: Vec::new(),
        };
        Node(Arc::new(RwLock::new(node)))
    }

    pub fn read(&self) -> RwLockReadGuard<'_, _Node<T>> {
        self.0.try_read().unwrap()
    }

    pub fn write(&self) -> RwLockWriteGuard<'_, _Node<T>> {
        self.0.try_write().unwrap()
    }

    pub fn downgrade(&self) -> NodeWeak<T> {
        NodeWeak(Arc::downgrade(&self.0))
    }

    fn ptr(&self) -> *const RwLock<_Node<T>> {
        Arc::as_ptr(&self.0)
    }

    /// Make sure the node has link lists for layers 0..=level.
    fn init_levels(&self, level: usize) {
        let mut w = self.write();
        ensure_level(&mut w.neighbors, level);
        ensure_level(&mut w.in_neighbors, level);
    }

    /// Outgoing links of this node at `level` (empty if it is not on that layer).
    fn links(&self, level: usize) -> Vec<Node<T>> {
        self.read()
            .neighbors
            .get(level)
            .map(|l| l.iter().map(NodeWeak::upgrade).collect())
            .unwrap_or_default()
    }
}

/// A node together with its similarity to some base element. Ordered by
/// similarity, so a max-heap pops the nearest element first.
struct Scored<T: Float, R: Float> {
    sim: OrderedFloat<R>,
    node: Node<T>,
}

impl<T: Float, R: Float> PartialEq for Scored<T, R> {
    fn eq(&self, other: &Self) -> bool {
        self.sim == other.sim
    }
}

impl<T: Float, R: Float> Eq for Scored<T, R> {}

impl<T: Float, R: Float> PartialOrd for Scored<T, R> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Float, R: Float> Ord for Scored<T, R> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sim.cmp(&other.sim)
    }
}

#[derive(Clone)]
pub struct Index<T: Float, R: Float> {
    pub name: String,                           // index name
    pub mfunc: Box<metrics::MetricFuncT<T, R>>, // metric function
    pub mfunc_kind: metrics::MetricFuncs,       // kind of the metric function
    pub data_dim: usize,                        // dimensionality of the data
    pub m: usize,                               // out vertexs per node
    pub m_max: usize,                           // max number of vertexes per node
    pub m_max_0: usize,                         // max number of vertexes at layer 0
    pub ef_construction: usize,                 // size of dynamic candidate list
    pub level_mult: f64,                        // level generation factor
    pub node_count: usize,                      // count of nodes
    pub max_layer: usize,                       // idx of top layer
    pub layers: Vec<HashSet<NodeWeak<T>>>,      // distinct nodes in each layer
    pub nodes: HashMap<String, Node<T>>,        // hashmap of nodes
    pub enterpoint: Option<NodeWeak<T>>,        // enterpoint node
}

impl<T: Float, R: Float> Index<T, R> {
    pub fn new(
        name: &str,
        mfunc: Box<metrics::MetricFuncT<T, R>>,
        data_dim: usize,
        m: usize,
        ef_construction: usize,
    ) -> Self {
        Index {
            name: name.to_string(),
            mfunc,
            mfunc_kind: metrics::MetricFuncs::Euclidean,
            data_dim,
            m,
            m_max: m,
            m_max_0: m * 2,
            ef_construction,
            level_mult: 1.0 / (1.0 * m as f64).ln(),
            node_count: 0,
            max_layer: 0,
            layers: Vec::new(),
            nodes: HashMap::new(),
            enterpoint: None,
        }
    }
}

impl<T: Float, R: Float> fmt::Debug for Index<T, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "name: {}\n\
             metric: {:?}\n\
             data_dim: {}\n\
             M: {}\n\
             ef_construction: {}\n\
             level_mult: {}\n\
             node_count: {:?}\n\
             max_layer: {:?}\n\
             enterpoint: {}\n",
            self.name,
            self.mfunc_kind,
            self.data_dim,
            self.m,
            self.ef_construction,
            self.level_mult,
            self.node_count,
            self.max_layer,
            match &self.enterpoint {
                Some(node) => node.upgrade().read().name.clone(),
                None => "null".to_owned(),
            },
        )
    }
}

impl<T, R> Index<T, R>
where
    T: Float + Send + Sync + 'static,
    R: Float,
{
    pub fn add_node(
        &mut self,
        name: &str,
        data: &[T],
        update_fn: impl Fn(String, Node<T>),
    ) -> Result<(), HNSWError> {
        if data.len() != self.data_dim {
            return Err(format!("data dimension: {} does not match Index", data.len()).into());
        }
        if self.nodes.contains_key(name) {
            return Err(format!("Node: {} already exists", name).into());
        }

        self.insert(name, data, update_fn);
        Ok(())
    }

    pub fn delete_node(
        &mut self,
        name: &str,
        update_fn: impl Fn(String, Node<T>),
    ) -> Result<(), HNSWError> {
        let node = match self.nodes.remove(name) {
            Some(node) => node,
            None => return Err(format!("Node: {} does not exist", name).into()),
        };
        self.node_count -= 1;

        let weak = node.downgrade();
        for layer in self.layers.iter_mut() {
            layer.remove(&weak);
        }

        // Repair every node that links to the deleted one. Its candidates are
        // its remaining links plus the deleted node's links (the deleted node
        // was the bridge between them), pruned with the same heuristic as in
        // insertion.
        let mut updated = HashSet::new();
        let levels = node.read().neighbors.len();
        for lc in 0..levels {
            let outs = node.links(lc);
            let ins: Vec<Node<T>> = node
                .read()
                .in_neighbors
                .get(lc)
                .map(|l| l.iter().map(NodeWeak::upgrade).collect())
                .unwrap_or_default();

            for y in &outs {
                remove_link(&mut y.write().in_neighbors, lc, &weak);
            }

            for x in &ins {
                let mut seen: HashSet<_> = [node.ptr(), x.ptr()].into();
                let candidates: Vec<Node<T>> = x
                    .links(lc)
                    .into_iter()
                    .chain(outs.iter().cloned())
                    .filter(|c| seen.insert(c.ptr()))
                    .collect();
                let scored = self.score(&x.read().data, candidates);
                let params = SelectParams {
                    m: self.max_conn(lc),
                    lc,
                    extend_candidates: false,
                    keep_pruned_connections: true,
                };
                let new = self.select_neighbors(x, scored, params);
                self.set_links(x, lc, new);
                updated.insert(x.clone());
            }
        }

        // update nodes in redis
        for n in updated {
            let name = n.read().name.clone();
            update_fn(name, n);
        }

        // pick a new enterpoint from the highest non-empty layer
        if self.enterpoint.as_ref() == Some(&weak) {
            self.enterpoint = None;
            while let Some(top) = self.layers.last() {
                if let Some(n) = top.iter().next() {
                    self.enterpoint = Some(n.clone());
                    break;
                }
                self.layers.pop();
            }
            self.max_layer = self.layers.len().saturating_sub(1);
        }

        Ok(())
    }

    pub fn search_knn(&self, data: &[T], k: usize) -> Result<Vec<SearchResult<T, R>>, HNSWError> {
        if data.len() != self.data_dim {
            return Err(format!("data dimension: {} does not match Index", data.len()).into());
        }
        if self.enterpoint.is_none() || self.node_count == 0 {
            return Ok(Vec::new());
        }

        // The paper requires ef >= K for the search to be able to return K results.
        Ok(self.search_knn_internal(data, k, self.ef_construction.max(k)))
    }

    /// Recompute every node's `in_neighbors` from the outgoing links. Call
    /// after building the graph from storage.
    pub fn rebuild_in_neighbors(&self) {
        for n in self.nodes.values() {
            n.write().in_neighbors.clear();
        }
        for n in self.nodes.values() {
            let outs: Vec<Vec<Node<T>>> = (0..n.read().neighbors.len())
                .map(|lc| n.links(lc))
                .collect();
            let src = n.downgrade();
            for (lc, layer) in outs.iter().enumerate() {
                for y in layer {
                    let mut w = y.write();
                    ensure_level(&mut w.in_neighbors, lc);
                    push_unique(&mut w.in_neighbors[lc], src.clone());
                }
            }
        }
    }

    fn max_conn(&self, lc: usize) -> usize {
        if lc == 0 { self.m_max_0 } else { self.m_max }
    }

    fn sim(&self, a: &[T], b: &[T]) -> OrderedFloat<R> {
        OrderedFloat::from((self.mfunc)(a, b, self.data_dim))
    }

    fn score(&self, base: &[T], nodes: Vec<Node<T>>) -> Vec<Scored<T, R>> {
        nodes
            .into_iter()
            .map(|node| {
                let sim = self.sim(base, &node.read().data);
                Scored { sim, node }
            })
            .collect()
    }

    /// Add the directed edge a -> b on layer `lc`.
    fn link(&self, a: &Node<T>, b: &Node<T>, lc: usize) {
        // Lists are sized defensively: graphs saved by older versions created
        // per-layer lists lazily, so a node may lack lists for its upper layers.
        let mut aw = a.write();
        ensure_level(&mut aw.neighbors, lc);
        push_unique(&mut aw.neighbors[lc], b.downgrade());
        drop(aw);
        let mut bw = b.write();
        ensure_level(&mut bw.in_neighbors, lc);
        push_unique(&mut bw.in_neighbors[lc], a.downgrade());
    }

    /// Replace the outgoing links of `node` on layer `lc` ("set
    /// neighbourhood(e) at layer lc to eNewConn" in alg. 1), keeping the
    /// reverse links of the affected nodes in sync.
    fn set_links(&self, node: &Node<T>, lc: usize, new: Vec<Scored<T, R>>) {
        let me = node.downgrade();
        let new: Vec<NodeWeak<T>> = new.into_iter().map(|s| s.node.downgrade()).collect();
        let old = std::mem::take(&mut node.write().neighbors[lc]);
        for o in old.iter().filter(|o| !new.contains(o)) {
            remove_link(&mut o.upgrade().write().in_neighbors, lc, &me);
        }
        for n in new.iter().filter(|n| !old.contains(n)) {
            let n = n.upgrade();
            let mut w = n.write();
            ensure_level(&mut w.in_neighbors, lc);
            push_unique(&mut w.in_neighbors[lc], me.clone());
        }
        node.write().neighbors[lc] = new;
    }

    /// INSERT (alg. 1)
    fn insert(&mut self, name: &str, data: &[T], update_fn: impl Fn(String, Node<T>)) {
        let l = self.gen_random_level();
        let query = Node::new(name, data, l + 1);
        query.init_levels(l);
        self.nodes.insert(name.to_owned(), query.clone());
        self.node_count += 1;

        let ep = match &self.enterpoint {
            Some(ep) => ep.upgrade(),
            None => {
                // first element: it is the whole graph
                self.max_layer = l;
                self.enterpoint = Some(query.downgrade());
                self.layers = (0..=l).map(|_| HashSet::new()).collect();
                self.layers[l].insert(query.downgrade());
                return;
            }
        };
        let l_max = self.max_layer;

        // phase 1: greedy search (ef = 1) through the layers above l
        let mut eps = vec![ep];
        for lc in (l + 1..=l_max).rev() {
            let w = self.search_layer(data, &eps, 1, lc);
            eps = vec![w.into_iter().next().unwrap().node];
        }

        // phase 2: connect on layers min(L, l)..=0
        let mut updated = HashSet::new();
        for lc in (0..=min(l_max, l)).rev() {
            let w = self.search_layer(data, &eps, self.ef_construction, lc);
            let candidates: Vec<Scored<T, R>> = w
                .iter()
                .map(|s| Scored {
                    sim: s.sim,
                    node: s.node.clone(),
                })
                .collect();
            let params = SelectParams {
                m: self.m,
                lc,
                extend_candidates: false,
                keep_pruned_connections: true,
            };
            let neighbors = self.select_neighbors(&query, candidates, params);

            // add bidirectional connections from neighbors to q
            for e in &neighbors {
                self.link(&query, &e.node, lc);
                self.link(&e.node, &query, lc);
            }

            // shrink the neighbors' connections if they are now over capacity
            let m_max = self.max_conn(lc);
            for e in &neighbors {
                let econn = e.node.links(lc);
                if econn.len() > m_max {
                    let scored = self.score(&e.node.read().data, econn);
                    let params = SelectParams {
                        m: m_max,
                        lc,
                        extend_candidates: false,
                        keep_pruned_connections: true,
                    };
                    let enewconn = self.select_neighbors(&e.node, scored, params);
                    self.set_links(&e.node, lc, enewconn);
                }
                updated.insert(e.node.clone());
            }

            // ep <- W: all found neighbors are entry points for the next layer
            eps = w.into_iter().map(|s| s.node).collect();
        }

        // update nodes in redis
        for n in updated {
            let name = n.read().name.clone();
            update_fn(name, n);
        }

        // new enterpoint if we're in a higher layer
        if l > l_max {
            self.max_layer = l;
            self.enterpoint = Some(query.downgrade());
        }
        while self.layers.len() <= l {
            self.layers.push(HashSet::new());
        }
        self.layers[l].insert(query.downgrade());
    }

    fn gen_random_level(&mut self) -> usize {
        // random() is in [0, 1); flip to (0, 1] so ln() never sees 0.
        let r: f64 = 1.0 - rand::rng().random::<f64>();
        (-r.ln() * self.level_mult) as usize
    }

    /// SEARCH-LAYER (alg. 2). Returns the ef nearest elements found, nearest first.
    fn search_layer(
        &self,
        query: &[T],
        eps: &[Node<T>],
        ef: usize,
        lc: usize,
    ) -> Vec<Scored<T, R>> {
        let mut v = HashSet::new(); // visited
        let mut c = BinaryHeap::new(); // candidates, nearest on top
        let mut w = BinaryHeap::new(); // found nearest, furthest on top
        for ep in eps {
            if v.insert(ep.ptr()) {
                let sim = self.sim(query, &ep.read().data);
                c.push(Scored {
                    sim,
                    node: ep.clone(),
                });
                w.push(Reverse(Scored {
                    sim,
                    node: ep.clone(),
                }));
            }
        }
        while w.len() > ef {
            w.pop();
        }

        while let Some(cur) = c.pop() {
            let furthest = w.peek().unwrap().0.sim;
            if cur.sim < furthest {
                break; // all elements in W are evaluated
            }
            for e in cur.node.links(lc) {
                if !v.insert(e.ptr()) {
                    continue;
                }
                let sim = self.sim(query, &e.read().data);
                if w.len() < ef || sim > w.peek().unwrap().0.sim {
                    c.push(Scored {
                        sim,
                        node: e.clone(),
                    });
                    w.push(Reverse(Scored { sim, node: e }));
                    if w.len() > ef {
                        w.pop();
                    }
                }
            }
        }

        // ascending order of Reverse == descending similarity
        w.into_sorted_vec().into_iter().map(|r| r.0).collect()
    }

    /// SELECT-NEIGHBORS-HEURISTIC (alg. 4). `candidates` are scored against
    /// `base`; returns at most `params.m` elements, nearest first.
    fn select_neighbors(
        &self,
        base: &Node<T>,
        candidates: Vec<Scored<T, R>>,
        params: SelectParams,
    ) -> Vec<Scored<T, R>> {
        let base_data = base.read().data.clone();
        let mut seen: HashSet<_> = [base.ptr()].into();
        let mut w = BinaryHeap::with_capacity(candidates.len());
        let mut originals = Vec::new();
        for s in candidates {
            if seen.insert(s.node.ptr()) {
                if params.extend_candidates {
                    originals.push(s.node.clone());
                }
                w.push(s);
            }
        }

        // extend candidates by their neighbors
        for e in originals {
            for eadj in e.links(params.lc) {
                if seen.insert(eadj.ptr()) {
                    let sim = self.sim(&base_data, &eadj.read().data);
                    w.push(Scored { sim, node: eadj });
                }
            }
        }

        let mut r: Vec<Scored<T, R>> = Vec::with_capacity(params.m);
        let mut wd = Vec::new(); // discarded, in nearest-first order
        while r.len() < params.m {
            let Some(e) = w.pop() else { break };
            // keep e only if it is closer to the base element than to any
            // already selected element
            let good = {
                let ed = e.node.read();
                r.iter()
                    .all(|s| self.sim(&ed.data, &s.node.read().data) <= e.sim)
            };
            if good {
                r.push(e);
            } else {
                wd.push(e);
            }
        }

        // add back some of the discarded connections
        if params.keep_pruned_connections {
            let room = params.m - r.len();
            r.extend(wd.into_iter().take(room));
        }

        r
    }

    /// K-NN-SEARCH (alg. 5)
    fn search_knn_internal(&self, query: &[T], k: usize, ef: usize) -> Vec<SearchResult<T, R>> {
        let mut eps = vec![self.enterpoint.as_ref().unwrap().upgrade()];
        for lc in (1..=self.max_layer).rev() {
            let w = self.search_layer(query, &eps, 1, lc);
            eps = vec![w.into_iter().next().unwrap().node];
        }

        self.search_layer(query, &eps, ef, 0)
            .into_iter()
            .take(k)
            .map(|s| {
                let nr = s.node.read();
                SearchResult::new(s.sim, nr.name.rsplit('.').next().unwrap(), &nr.data)
            })
            .collect()
    }
}
