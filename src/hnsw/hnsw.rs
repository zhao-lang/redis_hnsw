use super::metrics;

use ordered_float::OrderedFloat;
use rand::prelude::*;
use std::cell::RefCell;
use std::cmp::{min, Eq, Ord, Ordering, PartialEq, PartialOrd, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::rc::Rc;
use std::sync::{Arc, RwLock};
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

#[derive(Copy, Clone, Debug)]
pub enum HNSWRedisMode {
    Source,
    Storage,
}

impl fmt::Display for HNSWRedisMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format!("{:?}", self).to_uppercase())
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub name: String,
    pub data: Vec<f32>,
    pub neighbors: Vec<Vec<Arc<RwLock<Node>>>>,
}

impl Node {
    fn new(name: &str, data: &Vec<f32>, capacity: usize) -> Self {
        Node {
            name: name.to_owned(),
            data: data.to_owned(),
            neighbors: Vec::with_capacity(capacity),
        }
    }

    fn push_levels(&mut self, level: usize) {
        while self.neighbors.len() < level + 1 {
            self.neighbors.push(Vec::new());
        }
    }
}

#[derive(Debug, Clone)]
struct SimPair {
    pub sim: OrderedFloat<f32>,
    pub node: Arc<RwLock<Node>>,
}

impl PartialEq for SimPair {
    fn eq(&self, other: &Self) -> bool {
        self.sim == other.sim
    }
}

impl Eq for SimPair {}

impl PartialOrd for SimPair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.sim.partial_cmp(&other.sim)
    }
}

impl Ord for SimPair {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sim.cmp(&other.sim)
    }
}

pub struct Index {
    pub name: String,                              // index name
    pub mode: HNSWRedisMode,                       // redis mode
    pub mfunc: Box<metrics::MetricFuncT>,          // metric function
    pub mfunc_kind: metrics::MetricFuncs,          // kind of the metric function
    pub data_dim: usize,                           // dimensionality of the data
    pub m: usize,                                  // out vertexts per node
    pub m_max: usize,                              // max number of vertexes per node
    pub m_max_0: usize,                            // max number of vertexes at layer 0
    pub ef_construction: usize,                    // size of dynamic candidate list
    pub level_mult: f64,                           // level generation factor
    pub node_count: usize,                         // count of nodes
    pub max_layer: usize,                          // idx of top layer
    pub nodes: HashMap<String, Arc<RwLock<Node>>>, // hashmap of nodes
    pub enterpoint: Option<Arc<RwLock<Node>>>,     // string key to the enterpoint node
    rng_: StdRng,                                  // rng for level generation
}

impl Index {
    pub fn new(
        name: &str,
        mode: HNSWRedisMode,
        data_dim: usize,
        m: usize,
        ef_construction: usize,
    ) -> Self {
        Index {
            name: name.to_string(),
            mode: mode,
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
            nodes: HashMap::new(),
            enterpoint: None,
            rng_: StdRng::from_entropy(),
        }
    }
}

impl fmt::Display for Index {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "name: {}\n\
             mode: {}\n\
             metric: {:?}\n\
             data_dim: {}\n\
             M: {}\n\
             ef_construction: {}\n\
             node_count: {:?}\n\
             max_layer: {:?}\n\
             enterpoint: {:?}\n",
            self.name,
            self.mode,
            self.mfunc_kind,
            self.data_dim,
            self.m,
            self.level_mult,
            self.node_count,
            self.max_layer,
            self.enterpoint,
        )
    }
}

impl clone::Clone for Index {
    fn clone(&self) -> Self {
        Index {
            name: self.name.clone(),
            mode: self.mode,
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
            nodes: HashMap::new(), // blank map for now, rehydrate nodes from node set in redis
            enterpoint: self.enterpoint.clone(),
            rng_: self.rng_.clone(),
        }
    }
}

impl Index {
    pub fn add_node(&mut self, name: &str, data: &Vec<f32>) -> Result<(), HNSWError> {
        if self.node_count == 0 {
            let node = Arc::new(RwLock::new(Node::new(name, data, self.m_max_0)));
            self.enterpoint = Some(node.clone());
            self.nodes.insert(name.to_owned(), node);
            self.node_count += 1;

            return Ok(());
        }

        if self.nodes.get(name).is_none() {
            return Err(format!("Node: {:?} already exists", name).into());
        }

        self.insert(name, data)
    }

    // perform insertion of new nodes into the index
    fn insert(&mut self, name: &str, data: &Vec<f32>) -> Result<(), HNSWError> {
        let l = self.gen_random_level();
        let l_max = self.max_layer;

        if l_max == 0 {
            self.nodes.insert(
                name.to_owned(),
                Arc::new(RwLock::new(Node::new(name, data, self.m_max_0))),
            );
        } else {
            self.nodes.insert(
                name.to_owned(),
                Arc::new(RwLock::new(Node::new(name, data, self.m_max))),
            );
        }

        let mut ep = self.enterpoint.as_ref().unwrap().clone();
        let mut w: BinaryHeap<SimPair>;

        let mut lc = l_max;
        while lc > l {
            w = self.search_level(data, ep.clone(), 1, lc);
            ep = w.pop().unwrap().node;

            if lc == 0 {
                break;
            }
            lc -= 1;
        }

        // lc = min(l_max, l);
        // while lc >= 0 {
        //     w = self.search_level(data, ep.clone(), self.ef_construction_, lc);
        // }

        Ok(())
    }

    fn gen_random_level(&mut self) -> usize {
        let dist = rand::distributions::Uniform::from(0_f64..1_f64);
        let r: f64 = dist.sample(&mut self.rng_);
        (-r.ln() * self.level_mult) as usize
    }

    fn search_level(
        &self,
        query: &Vec<f32>,
        ep: Arc<RwLock<Node>>,
        ef: usize,
        level: usize,
    ) -> BinaryHeap<SimPair> {
        let mut v = HashSet::new();
        v.insert(ep.read().unwrap().name.clone());

        let qsim = (self.mfunc)(query, &ep.read().unwrap().data, self.data_dim);
        let qpair = Rc::new(RefCell::new(SimPair {
            sim: OrderedFloat::from(qsim),
            node: ep.clone(),
        }));

        let mut c = BinaryHeap::with_capacity(ef);
        let mut w = BinaryHeap::with_capacity(ef);
        c.push(qpair.clone());
        w.push(Reverse(qpair.clone()));

        while c.len() > 0 {
            let cpair = c.pop().unwrap();
            let fpair = w.peek().unwrap();

            if cpair.borrow().sim < fpair.0.borrow().sim {
                break;
            }

            // update C and W
            cpair.borrow_mut().node.write().unwrap().push_levels(level);
            for neighbor in &cpair.borrow().node.read().unwrap().neighbors[level] {
                if v.contains(&neighbor.read().unwrap().name) {}
            }
        }

        let mut res = BinaryHeap::new();
        for pair in c {
            res.push(pair.borrow().clone());
        }
        res
    }
}
