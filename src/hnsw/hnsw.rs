use super::metrics;

use ordered_float::OrderedFloat;
use owning_ref::{RefMutRefMut, RefRef, RwLockReadGuardRef};
use rand::prelude::*;
use std::cell::RefCell;
use std::cmp::{min, Eq, Ord, Ordering, PartialEq, PartialOrd, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::{Arc, RwLock};
use std::thread;
use std::{clone, fmt};

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

#[derive(Debug)]
pub struct SearchResult<T> {
    pub sim: OrderedFloat<f32>,
    pub name: String,
    pub data: Vec<T>,
}

impl<T: std::clone::Clone> SearchResult<T> {
    fn new(sim: OrderedFloat<f32>, name: &str, data: &[T]) -> Self {
        SearchResult {
            sim: sim,
            name: name.to_owned(),
            data: data.to_vec(),
        }
    }
}

type NodeRef<T> = Arc<RwLock<_Node<T>>>;

#[derive(Clone)]
pub struct _Node<T>
where
    T: std::clone::Clone,
{
    pub name: String,
    pub data: Vec<T>,
    pub neighbors: Vec<Vec<Node<T>>>,
}

impl<T: fmt::Debug + std::clone::Clone> fmt::Debug for _Node<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "name: {}\n\
             data: {:?}\n\
             neighbors: {:?}\n",
            self.name,
            self.data,
            self.neighbors
                .iter()
                .map(|l| {
                    l.into_iter()
                        .map(|n| n.read().name.to_owned())
                        .collect::<Vec<String>>()
                })
                .collect::<Vec<Vec<String>>>(),
        )
    }
}

impl<T: std::clone::Clone> _Node<T> {
    fn push_levels(&mut self, level: usize, capacity: Option<usize>) {
        let neighbors = &mut self.neighbors;
        while neighbors.len() < level + 1 {
            match capacity {
                Some(cap) => neighbors.push(Vec::with_capacity(cap)),
                None => neighbors.push(Vec::new()),
            }
        }
    }

    fn add_neighbor(&mut self, level: usize, neighbor: Node<T>, capacity: Option<usize>) {
        self.push_levels(level, capacity);
        let neighbors = &mut self.neighbors;
        neighbors[level].push(neighbor);
    }

    fn clear_neighbors(&mut self, level: usize) {
        let neighbors = &mut self.neighbors;
        let cap = neighbors[level].capacity();
        neighbors[level] = Vec::with_capacity(cap);
    }
}

#[derive(Debug, Clone)]
pub struct Node<T>(NodeRef<T>)
where
    T: std::clone::Clone;

impl<T: std::clone::Clone> PartialEq for Node<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<T: std::clone::Clone> Eq for Node<T> {}

impl<T: std::clone::Clone> Hash for Node<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.read().name.hash(state);
    }
}

impl<T: std::clone::Clone> Node<T> {
    fn new(name: &str, data: &[T], capacity: usize) -> Self {
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

    fn push_levels(&self, level: usize, capacity: Option<usize>) {
        let mut node = self.0.try_write().unwrap();
        node.push_levels(level, capacity);
    }

    fn add_neighbor(&self, level: usize, neighbor: Node<T>, capacity: Option<usize>) {
        let node = &mut self.0.try_write().unwrap();
        node.add_neighbor(level, neighbor, capacity);
    }

    fn clear_neighbors(&self, level: usize) {
        let node = &mut self.0.try_write().unwrap();
        node.clear_neighbors(level);
    }
}

type SimPairRef<T> = Rc<RefCell<_SimPair<T>>>;

#[derive(Debug, Clone)]
struct _SimPair<T>
where
    T: std::clone::Clone,
{
    pub sim: OrderedFloat<f32>,
    pub node: Node<T>,
}

#[derive(Debug, Clone)]
struct SimPair<T>(SimPairRef<T>)
where
    T: std::clone::Clone;

impl<T: std::clone::Clone> SimPair<T> {
    fn new(sim: OrderedFloat<f32>, node: Node<T>) -> Self {
        let sp = _SimPair {
            sim: sim,
            node: node,
        };
        SimPair(Rc::new(RefCell::new(sp)))
    }

    fn read(&self) -> RefRef<_SimPair<T>> {
        RefRef::new(self.0.borrow())
    }

    fn write(&mut self) -> RefMutRefMut<_SimPair<T>> {
        RefMutRefMut::new(self.0.borrow_mut())
    }
}

impl<T: std::clone::Clone> PartialEq for SimPair<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.borrow().sim == other.0.borrow().sim
    }
}

impl<T: std::clone::Clone> Eq for SimPair<T> {}

impl<T: std::clone::Clone> PartialOrd for SimPair<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.borrow().sim.partial_cmp(&other.0.borrow().sim)
    }
}

impl<T: std::clone::Clone> Ord for SimPair<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.borrow().sim.cmp(&other.0.borrow().sim)
    }
}

pub struct Index {
    pub name: String,                          // index name
    pub mfunc: Box<metrics::MetricFuncT<f32>>, // metric function
    pub mfunc_kind: metrics::MetricFuncs,      // kind of the metric function
    pub data_dim: usize,                       // dimensionality of the data
    pub m: usize,                              // out vertexts per node
    pub m_max: usize,                          // max number of vertexes per node
    pub m_max_0: usize,                        // max number of vertexes at layer 0
    pub ef_construction: usize,                // size of dynamic candidate list
    pub level_mult: f64,                       // level generation factor
    pub node_count: usize,                     // count of nodes
    pub max_layer: usize,                      // idx of top layer
    pub layers: Vec<HashSet<Node<f32>>>,       // distinct nodes in each layer
    pub nodes: HashMap<String, Node<f32>>,     // hashmap of nodes
    pub enterpoint: Option<Node<f32>>,         // string key to the enterpoint node
    rng_: StdRng,                              // rng for level generation
}

impl Index {
    pub fn new(name: &str, data_dim: usize, m: usize, ef_construction: usize) -> Self {
        Index {
            name: name.to_string(),
            mfunc: Box::new(metrics::euclidean),
            mfunc_kind: metrics::MetricFuncs::Euclidean,
            data_dim: data_dim,
            m: m,
            m_max: m,
            m_max_0: m * 2,
            ef_construction: ef_construction,
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

impl fmt::Debug for Index {
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
                Some(node) => node.read().name.clone(),
                None => "None".to_owned(),
            },
        )
    }
}

impl clone::Clone for Index {
    fn clone(&self) -> Self {
        Index {
            name: self.name.clone(),
            mfunc: self.mfunc.clone(),
            mfunc_kind: self.mfunc_kind,
            data_dim: self.data_dim,
            m: self.m,
            m_max: self.m_max,
            m_max_0: self.m_max_0,
            ef_construction: self.ef_construction,
            level_mult: self.level_mult,
            node_count: self.node_count,
            max_layer: self.max_layer,
            layers: self.layers.clone(),
            nodes: self.nodes.clone(),
            enterpoint: self.enterpoint.clone(),
            rng_: self.rng_.clone(),
        }
    }
}

impl Index {
    pub fn add_node(
        &mut self,
        name: &str,
        data: &[f32],
        update_fn: fn(String, Node<f32>),
    ) -> Result<(), HNSWError> {
        if data.len() != self.data_dim {
            return Err(format!("data dimension: {} does not match Index", data.len()).into());
        }

        if self.node_count == 0 {
            let node = Node::new(name, data, self.m_max_0);
            self.enterpoint = Some(node.clone());

            let mut layer = HashSet::new();
            layer.insert(node.clone());
            self.layers.push(layer);

            self.nodes.insert(name.to_owned(), node);
            self.node_count += 1;

            return Ok(());
        }

        if !self.nodes.get(name).is_none() {
            return Err(format!("Node: {:?} already exists", name).into());
        }

        self.insert(name, data, update_fn)
    }

    pub fn delete_node(
        &mut self,
        name: &str,
        update_fn: fn(String, Node<f32>),
    ) -> Result<(), HNSWError> {
        let node = match self.nodes.remove(name) {
            Some(node) => node,
            None => return Err(format!("Node: {:?} does not exist", name).into()),
        };
        self.node_count -= 1;

        for lc in (0..(self.max_layer + 1)).rev() {
            if self.layers[lc].remove(&node) {
                break;
            }
        }

        let nr = node.read();
        for lc in 0..nr.neighbors.len() {
            self.delete_node_from_neighbors(&node, lc, update_fn);
        }

        // update enterpoint if necessary
        if self.node_count == 0 {
            self.enterpoint = None;
        }
        match &self.enterpoint {
            Some(ep) if node == *ep => {
                let mut new_ep = None;
                for lc in (0..(self.max_layer + 1)).rev() {
                    match self.layers[lc].iter().next() {
                        Some(n) => {
                            new_ep = Some(n.clone());
                            break;
                        }
                        None => {
                            self.layers.pop();
                            self.max_layer -= 1;
                            continue;
                        }
                    }
                }
                self.enterpoint = new_ep;
            }
            _ => (),
        }

        Ok(())
    }

    pub fn search_knn(&self, data: &[f32], k: usize) -> Result<Vec<SearchResult<f32>>, HNSWError> {
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
        data: &[f32],
        update_fn: fn(String, Node<f32>),
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
        let mut w: BinaryHeap<SimPair<f32>>;

        let mut lc = l_max;
        while lc > l {
            w = self.search_level(data, ep.clone(), 1, lc);
            ep = w.pop().unwrap().read().node.clone();

            if lc == 0 {
                break;
            }
            lc -= 1;
        }

        lc = min(l_max, l);
        loop {
            w = self.search_level(data, ep.clone(), self.ef_construction, lc);
            let mut neighbors = self.select_neighbors(query, &w, self.m, lc, true, true, None);
            self.connect_neighbors(query, &neighbors, lc);

            // update nodes on redis
            for npair in &neighbors {
                let npr = npair.read();
                let node = npr.node.clone();
                let name = node.read().name.clone();
                let _ = thread::spawn(move || update_fn(name, node));
            }

            // shrink connections as needed
            while !neighbors.is_empty() {
                let epair = neighbors.pop().unwrap();
                let er = epair.read();

                let mut econn: BinaryHeap<SimPair<f32>>;
                {
                    let enr = er.node.read();
                    let eneighbors = &enr.neighbors[lc];
                    econn = BinaryHeap::with_capacity(eneighbors.len());
                    for n in eneighbors {
                        let ensim = OrderedFloat::from((self.mfunc)(
                            &enr.data,
                            &n.read().data,
                            self.data_dim,
                        ));
                        let enpair = SimPair::new(ensim, n.to_owned());
                        econn.push(enpair);
                    }
                }

                let m_max = if lc == 0 { self.m_max_0 } else { self.m_max };
                if econn.len() > m_max {
                    let enewconn =
                        self.select_neighbors(&er.node, &econn, m_max, lc, true, true, None);
                    self.update_node_connections(&er.node, &enewconn, lc);
                }
            }

            ep = w.peek().unwrap().read().node.clone();

            if lc == 0 {
                break;
            }
            lc -= 1;
        }

        // new enterpoint if we're in a higher layer
        if l > l_max {
            self.max_layer = l;
            self.enterpoint = Some(query.to_owned());
            while self.layers.len() < l + 1 {
                self.layers.push(HashSet::new());
            }
        }

        // add node to layer set
        self.layers[l].insert(query.to_owned());

        Ok(())
    }

    fn gen_random_level(&mut self) -> usize {
        let dist = rand::distributions::Uniform::from(0_f64..1_f64);
        let r: f64 = dist.sample(&mut self.rng_);
        (-r.ln() * self.level_mult) as usize
    }

    fn search_level(
        &self,
        query: &[f32],
        ep: Node<f32>,
        ef: usize,
        level: usize,
    ) -> BinaryHeap<SimPair<f32>> {
        let mut v = HashSet::with_capacity(ef);

        {
            v.insert(ep.clone());
        }
        let qsim: OrderedFloat<f32>;
        {
            qsim = OrderedFloat::from((self.mfunc)(query, &ep.read().data, self.data_dim));
        }
        let qpair = SimPair::new(qsim, ep.clone());

        let mut c = BinaryHeap::with_capacity(ef);
        let mut w = BinaryHeap::with_capacity(ef);
        c.push(qpair.clone());
        w.push(Reverse(qpair.clone()));

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
                        w.push(Reverse(epair.clone()));

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
        query: &Node<f32>,
        c: &BinaryHeap<SimPair<f32>>,
        m: usize,
        lc: usize,
        extend_candidates: bool,
        keep_pruned_connections: bool,
        ignored_node: Option<&Node<f32>>,
    ) -> BinaryHeap<SimPair<f32>> {
        let mut r: BinaryHeap<SimPair<f32>> = BinaryHeap::with_capacity(m);
        let mut w = c.clone();
        let mut wd = BinaryHeap::new();

        // extend candidates by their neighbors
        if extend_candidates {
            let mut ccopy = c.clone();

            let mut v = HashSet::with_capacity(ccopy.capacity());
            while !ccopy.is_empty() {
                let epair = ccopy.pop().unwrap();
                v.insert(epair.read().node.clone());
            }

            ccopy = c.clone();
            while !ccopy.is_empty() {
                let epair = ccopy.pop().unwrap();

                for eneighbor in &epair.read().node.read().neighbors[lc] {
                    if *eneighbor == *query
                        || (ignored_node.is_some() && *eneighbor == *ignored_node.unwrap())
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

        while !w.is_empty() && r.len() < m {
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
        if keep_pruned_connections {
            while !wd.is_empty() && r.len() < m {
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
        query: &Node<f32>,
        neighbors: &BinaryHeap<SimPair<f32>>,
        level: usize,
    ) {
        let mut neighbors = neighbors.clone();
        while !neighbors.is_empty() {
            let npair = neighbors.pop().unwrap();
            let npr = npair.read();

            query.add_neighbor(level, npr.node.clone(), Some(self.m_max_0));
            npr.node
                .add_neighbor(level, query.clone(), Some(self.m_max_0));
        }
    }

    fn update_node_connections(
        &self,
        node: &Node<f32>,
        conn: &BinaryHeap<SimPair<f32>>,
        level: usize,
    ) {
        let mut conn = conn.clone();
        node.clear_neighbors(level);
        while !conn.is_empty() {
            let newpair = conn.pop().unwrap();
            node.add_neighbor(level, newpair.read().node.clone(), Some(self.m_max_0));
        }
    }

    // TODO revisit this logic, some connections are not getting deleted
    // (or maybe some connections are not bidirectional?)
    fn delete_node_from_neighbors(
        &self,
        node: &Node<f32>,
        lc: usize,
        update_fn: fn(String, Node<f32>),
    ) {
        let r = node.read();
        let neighbors = &r.neighbors[lc];

        for n in neighbors {
            let nnewconn: BinaryHeap<SimPair<f32>>;
            {
                let nr = n.read();
                let nneighbors = &nr.neighbors[lc];
                let mut nconn = BinaryHeap::with_capacity(nneighbors.len());

                for nn in nneighbors {
                    let nnsim =
                        OrderedFloat::from((self.mfunc)(&nr.data, &nn.read().data, self.data_dim));
                    let nnpair = SimPair::new(nnsim, nn.to_owned());
                    nconn.push(nnpair);
                }

                let m_max = if lc == 0 { self.m_max_0 } else { self.m_max };
                nnewconn = self.select_neighbors(n, &nconn, m_max, lc, true, true, Some(node));
            }
            self.update_node_connections(n, &nnewconn, lc);
        }

        // update nodes on redis
        for n in neighbors {
            let name = n.read().name.clone();
            let node = n.clone();
            let _ = thread::spawn(move || update_fn(name, node));
        }
    }

    fn search_knn_internal(&self, query: &[f32], k: usize, ef: usize) -> Vec<SearchResult<f32>> {
        let mut ep = self.enterpoint.as_ref().unwrap().clone();
        let l_max = self.max_layer;

        let mut lc = l_max;
        while lc > 0 {
            let w = self.search_level(query, ep, 1, lc);
            ep = w.peek().unwrap().read().node.clone();
            lc -= 1;
        }

        let mut w = self.search_level(query, ep, ef, 0);

        let mut res = Vec::with_capacity(k);
        while res.len() < k && !w.is_empty() {
            let c = w.pop().unwrap();
            let cr = c.read();
            let cnr = cr.node.read();
            res.push(SearchResult::new(cr.sim, &cnr.name, &cnr.data));
        }
        res
    }
}
