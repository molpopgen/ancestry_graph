use nohash::BuildNoHashHasher;
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::BuildHasherDefault;

#[repr(transparent)]
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, PartialOrd, Ord)]
struct Node(usize);

impl Node {
    fn to_index(&self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy)]
enum NodeStatus {
    None,
    Birth,
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

impl Ancestry {
    fn birth(genome_length: i64) -> Option<Self> {
        Some(Self {
            segment: Segment::new(0, genome_length)?,
            ancestry: AncestryType::ToSelf,
        })
    }
}

enum AncestryChange {
    None,
    Gain((Node, Segment)),
    Loss((Node, Segment)),
}

#[derive(Debug)]
struct Graph {
    status: Vec<NodeStatus>,
    birth_time: Vec<Option<i64>>,
    parents: Vec<NodeHash>,
    // NOTE: for many scenarios, it may be preferable
    // to manage a Vec<(Node, Segment)> as the inner
    // value. We would sort the inner Vec during simplification.
    // The potential plus is that we'd avoid many hash lookups
    // during evolution. Also, sorting small Vectors tends
    // to be really fast.
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
            Some(_index) => todo!("Some"),
            None => {
                self.birth_time.push(Some(birth_time));
                self.status.push(status);
                self.parents
                    .push(NodeHash::with_hasher(BuildHasherDefault::default()));
                self.children
                    .push(ChildMap::with_hasher(BuildHasherDefault::default()));
                match status {
                    NodeStatus::Birth => self
                        .ancestry
                        .push(vec![Ancestry::birth(self.genome_length).unwrap()]),
                    _ => self.ancestry.push(vec![]),
                }
                Node(self.birth_time.len() - 1)
            }
        }
    }

    // TODO: we need a real error type
    fn validate_parent_child_birth_time(&self, parent: Node, child: Node) -> Result<(), ()> {
        let ptime = self
            .birth_time
            .get(parent.to_index())
            .ok_or_else(|| ())?
            .ok_or_else(|| ())?;
        let ctime = self
            .birth_time
            .get(child.to_index())
            .ok_or_else(|| ())?
            .ok_or_else(|| ())?;
        if ctime > ptime {
            Ok(())
        } else {
            Err(())
        }
    }

    // TODO: we need a real error type
    pub fn record_transmission(
        &mut self,
        left: i64,
        right: i64,
        parent: Node,
        child: Node,
    ) -> Result<(), ()> {
        self.validate_parent_child_birth_time(parent, child)?;
        let segment = Segment::new(left, right).ok_or_else(|| ())?;
        // We now "know" that parent, child are both in range.
        // (The only uncertainty is that we haven't checked that all our arrays
        //  are equal length.)
        let children = &mut self.children[parent.to_index()];
        match children.get_mut(&child) {
            Some(ref mut vec) => vec.push(segment),
            None => {
                let _ = children.insert(child, vec![segment]);
            }
        }
        let _ = self.parents[child.to_index()].insert(parent);

        Ok(())
    }

    // NOTE: panics if child is out of bounds
    pub fn parents(&self, child: Node) -> impl Iterator<Item = &Node> + '_ {
        self.parents[child.to_index()].iter()
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
    let child0 = graph.add_node(NodeStatus::Birth, 1);
    let child1 = graph.add_node(NodeStatus::Birth, 1);

    graph
        .record_transmission(0, graph.genome_length(), parent, child0)
        .unwrap();
    graph
        .record_transmission(0, graph.genome_length(), parent, child1)
        .unwrap();
    assert_eq!(graph.parents(child0).count(), 1);
    assert_eq!(graph.parents(child1).count(), 1);
    // WARNING: tests internal details
    assert_eq!(graph.children[parent.to_index()].len(), 2);
}

#[test]
fn design_test_3() {
    let mut graph = Graph::new(100).unwrap();
    let parent = graph.add_node(NodeStatus::None, 0);
    let child0 = graph.add_node(NodeStatus::Birth, 1);
    let child1 = graph.add_node(NodeStatus::Birth, 1);

    graph
        .record_transmission(0, graph.genome_length(), parent, child0)
        .unwrap();
    graph
        .record_transmission(0, graph.genome_length(), parent, child1)
        .unwrap();
    // TODO:
    // What needs to happen:
    // * The child nodes are "born", which means:
    //   * They have ancestry ToSelf
    //   * Birth + ToSelf is an ANCESTRY CHANGE
    // * The ancestry changes need to be sent to the parent.
    //   * We need something that says "Ancestry ToSelf is coming in from Node Y"
    // * We need to recoginze that the ancestry changes
    //   overlap with all of the parent ancestry,
    //   updating the parent ancestry to Overlap(...)
    // * ???
    todo!("this test is to work out sending ancestry changes up the tree");
}
