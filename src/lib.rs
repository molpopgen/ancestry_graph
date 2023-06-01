use nohash::BuildNoHashHasher;
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::BuildHasherDefault;

#[repr(transparent)]
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, PartialOrd, Ord)]
struct Node(usize);

#[derive(Debug, Clone, Copy)]
enum NodeStatus {
    None,
    Alive,
}

/// TODO: could be a newtype?
type NodeHash = HashSet<Node, BuildNoHashHasher<usize>>;
/// TODO: could be a newtype?
type ChildMap = HashMap<Node, Vec<Segment>, BuildNoHashHasher<usize>>;

#[derive(Debug, Copy, Clone)]
struct Segment {
    left: i64,
    right: i64,
}

impl Segment {
    fn new(left: i64, right: i64) -> Option<Self> {
        if left >= 0 && right > left {
            Some(Self { left, right })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
enum AncestryType {
    ToSelf,
    Unary(usize),
    Overlap(NodeHash),
}

#[derive(Debug, Clone)]
struct Ancestry {
    segment: Segment,
    ancestry: AncestryType,
}

#[derive(Debug)]
struct Graph {
    status: Vec<NodeStatus>,
    birth_time: Vec<Option<i64>>,
    parents: Vec<NodeHash>,
    children: Vec<ChildMap>,
    ancestry: Vec<Vec<Ancestry>>,
    free_nodes: Vec<usize>,
    genome_length: i64,
}

impl Graph {
    fn new(genome_length: i64) -> Option<Self> {
        Self::with_capacity(0, genome_length)
    }

    fn with_capacity(capacity: usize, genome_length: i64) -> Option<Self> {
        if genome_length < 1 {
            return None;
        }
        let parents = Vec::with_capacity(capacity);
        let status = Vec::with_capacity(capacity);
        let birth_time = Vec::with_capacity(capacity);
        let children = Vec::with_capacity(capacity);
        let ancestry = Vec::with_capacity(capacity);
        let free_nodes = Vec::new();
        Some(Self {
            status,
            birth_time,
            parents,
            children,
            ancestry,
            free_nodes,
            genome_length,
        })
    }

    fn with_initial_nodes(num_nodes: usize, genome_length: i64) -> Option<Self> {
        let status = vec![NodeStatus::None; num_nodes];
        let birth_time = vec![Some(0); num_nodes];
        let parents = vec![NodeHash::with_hasher(BuildHasherDefault::default()); num_nodes];
        let children = vec![ChildMap::with_hasher(BuildHasherDefault::default()); num_nodes];
        let initial_ancestry = Ancestry {
            segment: Segment::new(0, genome_length)?,
            ancestry: AncestryType::ToSelf,
        };
        let ancestry = vec![vec![initial_ancestry]; num_nodes];
        let free_nodes = Vec::new();
        Some(Self {
            status,
            birth_time,
            parents,
            children,
            ancestry,
            free_nodes,
            genome_length,
        })
    }

    pub fn genome_length(&self) -> i64 {
        self.genome_length
    }

    /// # Complexity
    ///
    /// `O(N)` where `N` is the number of nodes allocated in the graph.
    pub fn iter_nodes_with_ancestry(&self) -> impl Iterator<Item = Node> + '_ {
        self.ancestry
            .iter()
            .enumerate()
            .filter_map(|(i, a)| if a.is_empty() { None } else { Some(Node(i)) })
    }

    pub fn add_node(&mut self, status: NodeStatus, birth_time: i64) -> Node {
        match self.free_nodes.pop() {
            Some(index) => todo!("Some"),
            None => todo!("None"),
        }
    }
}

#[test]
fn design_test_0() {
    let mut graph = Graph::with_initial_nodes(10, 1000000).unwrap();
    assert_eq!(graph.genome_length(), 1000000);
}

#[test]
fn design_test_1() {
    let graph = Graph::with_initial_nodes(10, 1000000).unwrap();
    assert_eq!(graph.iter_nodes_with_ancestry().count(), 10);
}

#[test]
fn design_test_2() {
    let mut graph = Graph::new(100).unwrap();
    let parent = graph.add_node(NodeStatus::None, 0);
    let child = graph.add_node(NodeStatus::Alive, 1);
}
