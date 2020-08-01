use super::metrics;

use num::Float;
use ordered_float::OrderedFloat;
use owning_ref::{RefMutRefMut, RefRef, RwLockReadGuardRef, RwLockWriteGuardRefMut};
use rand::prelude::*;
use std::cell::RefCell;
use std::cmp::{min, Eq, Ord, Ordering, PartialEq, PartialOrd, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::convert::From;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::{Arc, RwLock, Weak};
// use std::thread;

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
        format!("{:?}", self)
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
    pub neighbors: Vec<Vec<NodeWeak<T>>>,
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

impl<T: Float> _Node<T> {
    fn push_levels(&mut self, level: usize, capacity: Option<usize>) {
        let neighbors = &mut self.neighbors;
        while neighbors.len() < level + 1 {
            match capacity {
                Some(cap) => neighbors.push(Vec::with_capacity(cap)),
                None => neighbors.push(Vec::new()),
            }
        }
    }

    fn add_neighbor(&mut self, level: usize, neighbor: NodeWeak<T>, capacity: Option<usize>) {
        self.push_levels(level, capacity);
        let neighbors = &mut self.neighbors;
        if !neighbors[level].contains(&neighbor) {
            neighbors[level].push(neighbor);
        }
    }

    fn rm_neighbor(&mut self, level: usize, neighbor: &NodeWeak<T>) {
        let neighbors = &mut self.neighbors;
        let index = neighbors[level]
            .iter()
            .position(|n| *n == *neighbor)
            .unwrap();
        neighbors[level].remove(index);
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
        };
        Node(Arc::new(RwLock::new(node)))
    }

    pub fn read(&self) -> RwLockReadGuardRef<_Node<T>> {
        RwLockReadGuardRef::new(self.0.try_read().unwrap())
    }

    pub fn write(&self) -> RwLockWriteGuardRefMut<_Node<T>> {
        RwLockWriteGuardRefMut::new(self.0.try_write().unwrap())
    }

    fn push_levels(&self, level: usize, capacity: Option<usize>) {
        let mut node = self.0.try_write().unwrap();
        node.push_levels(level, capacity);
    }

    fn add_neighbor(&self, level: usize, neighbor: NodeWeak<T>, capacity: Option<usize>) {
        let node = &mut self.0.try_write().unwrap();
        node.add_neighbor(level, neighbor, capacity);
    }

    fn rm_neighbor(&self, level: usize, neighbor: &NodeWeak<T>) {
        let node = &mut self.0.try_write().unwrap();
        node.rm_neighbor(level, neighbor);
    }

    pub fn downgrade(&self) -> NodeWeak<T> {
        NodeWeak(Arc::downgrade(&self.0))
    }
}

type SimPairRef<T, R> = Rc<RefCell<_SimPair<T, R>>>;

#[derive(Debug, Clone)]
struct _SimPair<T, R>
where
    T: Float,
    R: Float,
{
    pub sim: OrderedFloat<R>,
    pub node: Node<T>,
}

#[derive(Debug, Clone)]
struct SimPair<T, R>(SimPairRef<T, R>)
where
    T: Float,
    R: Float;

impl<T, R> SimPair<T, R>
where
    T: Float,
    R: Float,
{
    fn new(sim: OrderedFloat<R>, node: Node<T>) -> Self {
        let sp = _SimPair {
            sim,
            node,
        };
        SimPair(Rc::new(RefCell::new(sp)))
    }

    fn read(&self) -> RefRef<_SimPair<T, R>> {
        RefRef::new(self.0.borrow())
    }

    fn write(&mut self) -> RefMutRefMut<_SimPair<T, R>> {
        RefMutRefMut::new(self.0.borrow_mut())
    }
}

impl<T, R> PartialEq for SimPair<T, R>
where
    T: Float,
    R: Float,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.borrow().sim == other.0.borrow().sim
    }
}

impl<T: Float, R: Float> Eq for SimPair<T, R> {}

impl<T, R> PartialOrd for SimPair<T, R>
where
    T: Float,
    R: Float,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.borrow().sim.partial_cmp(&other.0.borrow().sim)
    }
}

impl<T, R> Ord for SimPair<T, R>
where
    T: Float,
    R: Float,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.borrow().sim.cmp(&other.0.borrow().sim)
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
    pub rng_: StdRng,                           // rng for level generation
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
            rng_: StdRng::from_entropy(),
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

        if self.node_count == 0 {
            let node = Node::new(name, data, self.m_max_0);
            self.enterpoint = Some(node.downgrade());

            let mut layer = HashSet::new();
            layer.insert(node.downgrade());
            self.layers.push(layer);

            self.nodes.insert(name.to_owned(), node);
            self.node_count += 1;

            return Ok(());
        }

        if self.nodes.get(name).is_some() {
            return Err(format!("Node: {:?} already exists", name).into());
        }

        self.insert(name, data, update_fn)
    }

    pub fn delete_node(
        &mut self,
        name: &str,
        update_fn: impl Fn(String, Node<T>),
    ) -> Result<(), HNSWError> {
        let node = match self.nodes.remove(name) {
            Some(node) => node,
            None => return Err(format!("Node: {:?} does not exist", name).into()),
        };
        // self.nodes.shrink_to_fit();
        self.node_count -= 1;

        for lc in (0..(self.max_layer + 1)).rev() {
            if self.layers[lc].remove(&node.downgrade()) {
                break;
            }
        }

        let mut updated = HashSet::new();
        let nr = node.read();
        for lc in 0..nr.neighbors.len() {
            let up = self.delete_node_from_neighbors(&node, lc);
            for u in up {
                updated.insert(u);
            }
        }

        // update nodes in redis
        for n in updated {
            let name = n.read().name.clone();
            let node = n.clone();
            update_fn(name, node);
        }

        // update enterpoint if necessary
        match &self.enterpoint {
            Some(ep) if node == ep.upgrade() => {
                let mut new_ep = None;
                for lc in (0..(self.max_layer + 1)).rev() {
                    match self.layers[lc].iter().next() {
                        Some(n) => {
                            new_ep = Some(n.clone());
                            break;
                        }
                        None => {
                            // self.layers[lc].shrink_to_fit();
                            self.layers.pop();
                            if self.max_layer > 0 {
                                self.max_layer -= 1;
                            }
                            continue;
                        }
                    }
                }
                // self.layers.shrink_to_fit();
                self.enterpoint = new_ep;
            }
            _ => (),
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

        Ok(self.search_knn_internal(data, k, self.ef_construction))
    }

    // perform insertion of new nodes into the index
    fn insert(
        &mut self,
        name: &str,
        data: &[T],
        update_fn: impl Fn(String, Node<T>),
    ) -> Result<(), HNSWError> {
        let l = self.gen_random_level();
        let l_max = self.max_layer;

        if l_max == 0 {
            self.nodes
                .insert(name.to_owned(), Node::new(name, data, self.m_max_0));
        } else {
            self.nodes
                .insert(name.to_owned(), Node::new(name, data, self.m_max));
        }
        self.node_count += 1;

        let query = self.nodes.get(name).unwrap();
        let mut ep = self.enterpoint.as_ref().unwrap().clone();
        let mut w: BinaryHeap<SimPair<T, R>>;

        let mut lc = l_max;
        while lc > l {
            w = self.search_level(data, &ep.upgrade(), 1, lc);
            ep = w.pop().unwrap().read().node.downgrade();

            if lc == 0 {
                break;
            }
            lc -= 1;
        }

        let mut updated = HashSet::new();
        for lc in (0..(min(l_max, l) + 1)).rev() {
            w = self.search_level(data, &ep.upgrade(), self.ef_construction, lc);
            let params = SelectParams{
                m: self.m, 
                lc,
                extend_candidates: true, 
                keep_pruned_connections: true
            };
            let mut neighbors = self.select_neighbors(query, &w, params, None);
            self.connect_neighbors(query, &neighbors, lc);

            // add node to list of nodes to be updated in redis
            for npair in &neighbors {
                updated.insert(npair.read().node.clone());
            }

            // shrink connections as needed
            while !neighbors.is_empty() {
                let epair = neighbors.pop().unwrap();
                let er = epair.read();

                let mut econn: BinaryHeap<SimPair<T, R>>;
                {
                    let enr = er.node.read();
                    let eneighbors = &enr.neighbors[lc];
                    econn = BinaryHeap::with_capacity(eneighbors.len());
                    for n in eneighbors {
                        let ensim = OrderedFloat::from((self.mfunc)(
                            &enr.data,
                            &n.upgrade().read().data,
                            self.data_dim,
                        ));
                        let enpair = SimPair::new(ensim, n.upgrade());
                        econn.push(enpair);
                    }
                }

                let m_max = if lc == 0 { self.m_max_0 } else { self.m_max };
                if econn.len() > m_max {
                    let params = SelectParams{
                        m: m_max, 
                        lc,
                        extend_candidates: true, 
                        keep_pruned_connections: true
                    };
                    let enewconn =
                        self.select_neighbors(&er.node, &econn, params, None);
                    let up = self.update_node_connections(&er.node, &enewconn, &econn, lc, None);
                    for u in up {
                        updated.insert(u);
                    }
                }
            }

            ep = w.peek().unwrap().read().node.downgrade();
        }

        // update nodes in redis
        for n in updated {
            let name = n.read().name.clone();
            let node = n.clone();
            update_fn(name, node);
        }

        // new enterpoint if we're in a higher layer
        if l > l_max {
            self.max_layer = l;
            self.enterpoint = Some(query.downgrade());
            while self.layers.len() < l + 1 {
                self.layers.push(HashSet::new());
            }
        }

        // add node to layer set
        self.layers[l].insert(query.downgrade());

        Ok(())
    }

    fn gen_random_level(&mut self) -> usize {
        let dist = rand::distributions::Uniform::from(0_f64..1_f64);
        let r: f64 = dist.sample(&mut self.rng_);
        (-r.ln() * self.level_mult) as usize
    }

    fn search_level(
        &self,
        query: &[T],
        ep: &Node<T>,
        ef: usize,
        level: usize,
    ) -> BinaryHeap<SimPair<T, R>> {
        let mut v = HashSet::with_capacity(ef);

        {
            v.insert(ep.clone());
        }
        let qsim: OrderedFloat<R>;
        {
            qsim = OrderedFloat::from((self.mfunc)(query, &ep.read().data, self.data_dim));
        }
        let qpair = SimPair::new(qsim, ep.clone());

        let mut c = BinaryHeap::with_capacity(ef);
        let mut w = BinaryHeap::with_capacity(ef);
        c.push(qpair.clone());
        w.push(Reverse(qpair));

        while !c.is_empty() {
            let mut cpair = c.pop().unwrap();
            let mut fpair = w.peek().unwrap();

            {
                if cpair.read().sim < fpair.0.read().sim {
                    break;
                }
            }

            // update C and W
            {
                cpair.write().node.push_levels(level, Some(self.m_max_0));
            }
            let cpr = cpair.read();
            let neighbors = &cpr.node.read().neighbors[level];
            for neighbor in neighbors {
                let neighbor = neighbor.upgrade();
                if !v.contains(&neighbor) {
                    v.insert(neighbor.clone());

                    fpair = w.peek().unwrap();
                    let esim = OrderedFloat::from((self.mfunc)(
                        query,
                        &neighbor.read().data,
                        self.data_dim,
                    ));
                    if esim > fpair.0.read().sim || w.len() < ef {
                        let epair = SimPair::new(esim, neighbor.clone());
                        c.push(epair.clone());
                        w.push(Reverse(epair));

                        if w.len() > ef {
                            w.pop();
                        }
                    }
                }
            }
        }

        let mut res = BinaryHeap::new();
        for pair in w {
            res.push(pair.0);
        }
        res
    }

    fn select_neighbors(
        &self,
        query: &Node<T>,
        c: &BinaryHeap<SimPair<T, R>>,
        params: SelectParams,
        ignored_node: Option<&Node<T>>,
    ) -> BinaryHeap<SimPair<T, R>> {
        let mut r: BinaryHeap<SimPair<T, R>> = BinaryHeap::with_capacity(params.m);
        let mut w = c.clone();
        let mut wd = BinaryHeap::new();

        // extend candidates by their neighbors
        if params.extend_candidates {
            let mut ccopy = c.clone();

            let mut v = HashSet::with_capacity(ccopy.capacity());
            while !ccopy.is_empty() {
                let epair = ccopy.pop().unwrap();
                v.insert(epair.read().node.clone());
            }

            ccopy = c.clone();
            while !ccopy.is_empty() {
                let epair = ccopy.pop().unwrap();

                for eneighbor in &epair.read().node.read().neighbors[params.lc] {
                    let eneighbor = eneighbor.upgrade();
                    if eneighbor == *query
                        || (ignored_node.is_some() && eneighbor == *ignored_node.unwrap())
                    {
                        continue;
                    }

                    if !v.contains(&eneighbor) {
                        let ensim = OrderedFloat::from((self.mfunc)(
                            &query.read().data,
                            &eneighbor.read().data,
                            self.data_dim,
                        ));
                        let enpair = SimPair::new(ensim, eneighbor.clone());
                        w.push(enpair);
                        v.insert(eneighbor.clone());
                    }
                }
            }
        }

        while !w.is_empty() && r.len() < params.m {
            let epair = w.pop().unwrap();
            let enr = epair.read();

            if enr.node == *query || (ignored_node.is_some() && enr.node == *ignored_node.unwrap())
            {
                continue;
            }

            if r.is_empty() || enr.sim > r.peek().unwrap().read().sim {
                r.push(epair.clone());
            } else {
                wd.push(epair.clone());
            }
        }

        // add back some of the discarded connections
        if params.keep_pruned_connections {
            while !wd.is_empty() && r.len() < params.m {
                let ppair = wd.pop().unwrap();
                {
                    let pr = ppair.read();
                    if pr.node == *query
                        || (ignored_node.is_some() && pr.node == *ignored_node.unwrap())
                    {
                        continue;
                    }
                }
                r.push(ppair);
            }
        }

        r
    }

    fn connect_neighbors(
        &self,
        query: &Node<T>,
        neighbors: &BinaryHeap<SimPair<T, R>>,
        level: usize,
    ) {
        let mut neighbors = neighbors.clone();
        while !neighbors.is_empty() {
            let npair = neighbors.pop().unwrap();
            let npr = npair.read();

            query.add_neighbor(level, npr.node.downgrade(), Some(self.m_max_0));
            npr.node
                .add_neighbor(level, query.downgrade(), Some(self.m_max_0));
        }
    }

    fn update_node_connections(
        &self,
        node: &Node<T>,
        new_neighbors: &BinaryHeap<SimPair<T, R>>,
        old_neighbors: &BinaryHeap<SimPair<T, R>>,
        level: usize,
        ignored_node: Option<&Node<T>>,
    ) -> HashSet<Node<T>> {
        let mut newconn = new_neighbors.clone();
        let mut rmconn = old_neighbors.clone().into_vec();
        let mut updated = HashSet::new();
        updated.insert(node.clone());

        // bidirectionally connect new neighbors
        while !newconn.is_empty() {
            let newpair = newconn.pop().unwrap();
            let npr = newpair.read();
            node.add_neighbor(level, npr.node.downgrade(), Some(self.m_max_0));
            npr.node
                .add_neighbor(level, node.downgrade(), Some(self.m_max_0));
            updated.insert(npr.node.clone());
            // if new neighbor exists in the old set then we remove it from
            // the set of neighbors to be removed
            if let Some(index) = rmconn.iter().position(|n| n.read().node == npr.node) {
                rmconn.remove(index);
            }
        }

        // bidirectionally remove old connections
        while !rmconn.is_empty() {
            let rmpair = rmconn.pop().unwrap();
            let rmpr = rmpair.read();
            node.rm_neighbor(level, &rmpr.node.downgrade());
            // if node to be removed is the ignored node then pass
            match ignored_node {
                Some(n) if rmpr.node == *n => {
                    continue;
                }
                _ => {
                    rmpr.node.rm_neighbor(level, &node.downgrade());
                    updated.insert(rmpr.node.clone());
                }
            }
        }

        updated
    }

    fn delete_node_from_neighbors(&self, node: &Node<T>, lc: usize) -> HashSet<Node<T>> {
        let r = node.read();
        let neighbors = &r.neighbors[lc];
        let mut updated = HashSet::new();

        for n in neighbors {
            let n = n.upgrade();
            let nnewconn: BinaryHeap<SimPair<T, R>>;
            let mut nconn: BinaryHeap<SimPair<T, R>>;
            {
                let nr = n.read();
                let nneighbors = &nr.neighbors[lc];
                nconn = BinaryHeap::with_capacity(nneighbors.len());

                for nn in nneighbors {
                    let nn = nn.upgrade();
                    let nnsim =
                        OrderedFloat::from((self.mfunc)(&nr.data, &nn.read().data, self.data_dim));
                    let nnpair = SimPair::new(nnsim, nn.to_owned());
                    nconn.push(nnpair);
                }

                let m_max = if lc == 0 { self.m_max_0 } else { self.m_max };
                let params = SelectParams{
                    m: m_max, 
                    lc,
                    extend_candidates: true, 
                    keep_pruned_connections: true
                };
                nnewconn = self.select_neighbors(&n, &nconn, params, Some(node));
            }
            updated.insert(n.clone());
            let up = self.update_node_connections(&n, &nnewconn, &nconn, lc, Some(node));
            for u in up {
                updated.insert(u);
            }
        }

        updated
    }

    fn search_knn_internal(&self, query: &[T], k: usize, ef: usize) -> Vec<SearchResult<T, R>> {
        let mut ep = self.enterpoint.as_ref().unwrap().clone();
        let l_max = self.max_layer;

        let mut lc = l_max;
        while lc > 0 {
            let w = self.search_level(query, &ep.upgrade(), 1, lc);
            ep = w.peek().unwrap().read().node.downgrade();
            lc -= 1;
        }

        let mut w = self.search_level(query, &ep.upgrade(), ef, 0);

        let mut res = Vec::with_capacity(k);
        while res.len() < k && !w.is_empty() {
            let c = w.pop().unwrap();
            let cr = c.read();
            let cnr = cr.node.read();
            res.push(SearchResult::new(
                cr.sim,
                &((&cnr.name).split('.').collect::<Vec<&str>>())
                    .last()
                    .unwrap(),
                &cnr.data,
            ));
        }
        res
    }
}
