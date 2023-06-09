use nohash::BuildNoHashHasher;
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::BuildHasherDefault;

// NOTE: for design purposes -- delete later.
mod overlapper_experiments;

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

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
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

    fn overlaps(&self, other: &Self) -> bool {
        self.right > other.left && other.right > self.left
    }
    fn left(&self) -> i64 {
        self.left
    }
    fn right(&self) -> i64 {
        self.right
    }
}

// NOTE: we may want to remove PartialEq
// later because it may only be used for TDD
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum AncestryType {
    ToSelf,
    Unary(Node),
    Overlap(Node),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Ancestry {
    segment: Segment,
    ancestry: AncestryType,
}

impl Ancestry {
    /// Create ancestry for a new birth:
    /// * Maps to self
    /// * Segment over whole genome.
    fn birth(genome_length: i64) -> Option<Self> {
        Some(Self {
            segment: Segment::new(0, genome_length)?,
            ancestry: AncestryType::ToSelf,
        })
    }

    fn overlaps_change(&self, other: &AncestryChange) -> bool {
        self.segment.overlaps(&other.segment)
    }

    fn left(&self) -> i64 {
        self.segment.left()
    }

    fn right(&self) -> i64 {
        self.segment.right()
    }

    fn identical_segment(&self, other: &Ancestry) -> bool {
        self.segment == other.segment
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct AncestryChange {
    segment: Segment,
    node: Node,
    change_type: AncestryChangeType,
}

impl AncestryChange {
    fn left(&self) -> i64 {
        self.segment.left()
    }
    fn right(&self) -> i64 {
        self.segment.right()
    }
}

// TODO: test PartialOrd, Ord implementations
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum AncestryChangeType {
    ToOverlap,
    // NOTE: ToUnary is overloaded with "Unary and is sample"
    // We may want to un-overload the term later
    ToUnary,
    ToLoss,
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
    // Used to cache nodes that are born.
    // Simplification will traverse this.
    births: Vec<Node>,
    free_nodes: Vec<usize>,
    genome_length: i64,
}

// Constructors
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
        let births = vec![];
        let free_nodes = Vec::new();
        Some(Self {
            status,
            birth_time,
            parents,
            children,
            ancestry,
            births,
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
        let births = vec![];
        let free_nodes = Vec::new();
        Some(Self {
            status,
            birth_time,
            parents,
            children,
            ancestry,
            births,
            free_nodes,
            genome_length,
        })
    }

    // TODO: rename. This fn really calculates the changes
    // needed to send to node's parents.
    // Panics if node is invalid
    // NOTE: this probably needs to return a Vec<AncestryChange>?
    // NOTE: OR, instead of a Vec, we push changes to some STACK?
    // NOTE: OR, it returns an iterator over changes, meaning
    //       that pushing to some STACK is handled elsewhere?
    fn calculate_ancestry_changes(&self, node: Node) -> Vec<AncestryChange> {
        match self.status[node.to_index()] {
            NodeStatus::Birth => vec![AncestryChange {
                node,
                segment: Segment {
                    left: 0,
                    right: self.genome_length(),
                },
                change_type: AncestryChangeType::ToUnary,
            }],
            _ => todo!(),
        }
    }

    fn propagage_ancestry_changes_to(&mut self, parent: Node, changes: &[AncestryChange]) {
        // TODO list:
        // * We need to identify all overlaps beteen parent
        //   ancestry and changes.
        //   (We can steal ideas from earlier prototypes.)
        // * Update all current ancestry.
        // * If there are any changes, propagate them to
        //   the (unimplemented) changes stack for all of parent's parents.
        // TECHNICALITIES:
        // * Imagine current parent ancestry ToSelf.
        //   The ancestry will be updated to contain Overlap([L/2, L)).
        // * If the node is a "sample", the ancestry should be updated
        //   to Unary(parent) and Overlap(...).
        // QUESTIONS:
        // * What if parent also has children (births)?
        // * We have NOT resolved the data model -- is self.children
        //   children or is it births?
        //   * If births, then the ancestry changes of these
        //     MUST be resolved already
        //   * If children, do we have a problem?
        todo!()
    }

    pub fn genome_length(&self) -> i64 {
        self.genome_length
    }

    /// # Complexity
    ///
    /// `O(N)` where `N` is the number of nodes allocated in the graph.
    // NOTE: this MAY not be pub in the long run
    pub fn iter_nodes_with_ancestry(&self) -> impl Iterator<Item = Node> + '_ {
        self.ancestry
            .iter()
            .enumerate()
            .filter_map(|(i, a)| if a.is_empty() { None } else { Some(Node(i)) })
    }

    // NOTE: this probably SHOULD NOT be pub in the long run
    // What we need instead is a way to separate "start
    // with a set of initial nodes" from "add a new node that is a birth"
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

    pub fn add_birth(&mut self, birth_time: i64) -> Result<Node, ()> {
        let rv = self.add_node(NodeStatus::Birth, birth_time);
        debug_assert_eq!(self.birth_time[rv.to_index()], Some(birth_time));
        // births are (locally) in increasing order
        match self.births.len() {
            i if i > 0 => {
                if self.birth_time[self.births[self.births.len() - 1].to_index()].unwrap()
                    > birth_time
                {
                    return Err(());
                } else {
                    ()
                }
            }
            _ => (),
        }
        self.births.push(rv);
        Ok(rv)
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
    // NOTE: may not need to be pub
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
    let child0 = graph.add_birth(1).unwrap();
    let child1 = graph.add_birth(1).unwrap();

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
    let child0 = graph.add_birth(1).unwrap();
    let child1 = graph.add_birth(1).unwrap();

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

    // WARNING: we are probably calling non-public functions below
    // NOTE: API is "wrong" -- changes "should" be updating
    // something modeling "The complete set of changes related
    // to parent 'i'".
    let mut a0 = graph.calculate_ancestry_changes(child0);
    let a1 = graph.calculate_ancestry_changes(child1);
    a0.extend_from_slice(&a1);
    graph.propagage_ancestry_changes_to(parent, &a0);
}

#[test]
fn test_births_out_of_order() {
    let mut graph = Graph::new(100).unwrap();
    let _ = graph.add_node(NodeStatus::None, 0);
    let _ = graph.add_birth(2).unwrap();
    assert!(graph.add_birth(1).is_err());
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum AncestryOverlapSource {
    // Segment comes from the current parental node
    Parent(AncestryType),
    // Segment is a change that we need to
    // consider for the parental node.
    Change(AncestryChangeType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct AncestryOverlap {
    segment: Segment,
    source: AncestryOverlapSource,
    node: Node,
}

impl AncestryOverlap {
    fn left(&self) -> i64 {
        self.segment.left()
    }
    fn right(&self) -> i64 {
        self.segment.right()
    }
}

fn generate_overlap_queue(
    parent: Node,
    parental_node_ancestry: &[Ancestry],
    ancestry_changes: &[AncestryChange],
) -> Vec<AncestryOverlap> {
    // TODO: should have a way to detect this and return an Error
    assert!(parental_node_ancestry.windows(2).all(|w| w[0] <= w[1]));
    assert!(ancestry_changes.windows(2).all(|w| w[0] <= w[1]));
    let mut queue = vec![];
    let mut d = 0_usize;

    while d < parental_node_ancestry.len() {
        queue.push(AncestryOverlap {
            segment: parental_node_ancestry[d].segment,
            source: AncestryOverlapSource::Parent(parental_node_ancestry[d].ancestry),
            node: parent,
        });
        for &ac in ancestry_changes {
            if ac.right() > parental_node_ancestry[d].left()
                && parental_node_ancestry[d].right() > ac.left()
            {
                let left = std::cmp::max(ac.left(), parental_node_ancestry[d].left());
                let right = std::cmp::min(ac.right(), parental_node_ancestry[d].right());
                // NOTE: we are NOT adding loss segments
                // to the overlap queue.
                // If we have problems later on w/unary node
                // propagation, we may need to revisit this.
                // (Unary nodes can be fiddly to think about,
                // hence the note.)
                match ac.change_type {
                    AncestryChangeType::ToLoss => (),
                    AncestryChangeType::ToOverlap | AncestryChangeType::ToUnary => {
                        queue.push(AncestryOverlap {
                            segment: Segment::new(left, right).unwrap(),
                            source: AncestryOverlapSource::Change(ac.change_type),
                            node: ac.node,
                        })
                    }
                }
            }
        }
        let update = parental_node_ancestry
            .iter()
            .skip(d)
            .take_while(|x| x.identical_segment(&parental_node_ancestry[d]))
            .count();
        d += update + 1;
    }

    queue
}

struct AncestryOverlapper {
    queue: Vec<AncestryOverlap>,
    num_overlaps: usize,
    current_overlap: usize,
    left: i64,
    right: i64,
    overlaps: Vec<AncestryOverlap>,
}

impl AncestryOverlapper {
    fn new(
        parent: Node,
        parental_node_ancestry: &[Ancestry],
        ancestry_changes: &[AncestryChange],
    ) -> Self {
        let mut queue = generate_overlap_queue(parent, parental_node_ancestry, ancestry_changes);
        let num_overlaps = queue.len();
        // Add sentinel
        queue.push(AncestryOverlap {
            segment: Segment::new(i64::MAX - 1, i64::MAX).unwrap(),
            source: AncestryOverlapSource::Parent(AncestryType::ToSelf),
            node: Node(usize::MAX),
        });
        let right = if num_overlaps > 0 {
            queue[0].right()
        } else {
            i64::MAX
        };
        let left = i64::MAX;
        Self {
            queue,
            num_overlaps,
            current_overlap: 0,
            left,
            right,
            overlaps: vec![],
        }
    }

    fn update_right_from_overlaps(&mut self) {
        self.right = match self.overlaps.iter().map(|&o| o.right()).min() {
            Some(right) => right,
            None => self.right,
        }
    }

    fn filter_overlaps(&mut self) {
        self.overlaps.retain(|x| x.right() > self.left);
    }

    fn make_overlaps(&self) -> Overlaps<'_> {
        Overlaps {
            left: self.left,
            right: self.right,
            overlaps: &self.overlaps,
            current_overlap: 0,
        }
    }

    fn calculate_next_overlap_set(&mut self) -> Option<Overlaps<'_>> {
        if self.current_overlap < self.num_overlaps {
            self.left = self.right;
            self.filter_overlaps();
            if self.overlaps.is_empty() {
                self.left = self.queue[self.current_overlap].left();
            }
            while self.current_overlap < self.num_overlaps
                && self.queue[self.current_overlap].left() == self.left
            {
                let x = self.queue[self.current_overlap];
                self.right = std::cmp::min(self.right, x.right());
                self.overlaps.push(x);
                self.current_overlap += 1;
            }
            self.update_right_from_overlaps();
            self.right = std::cmp::min(self.right, self.queue[self.current_overlap].left());
            Some(self.make_overlaps())
        } else {
            if !self.overlaps.is_empty() {
                self.left = self.right;
                self.filter_overlaps();
            }
            if !self.overlaps.is_empty() {
                self.update_right_from_overlaps();
                Some(self.make_overlaps())
            } else {
                None
            }
        }
    }
}

#[derive(Debug)]
struct Overlaps<'overlapper> {
    left: i64,
    right: i64,
    overlaps: &'overlapper [AncestryOverlap],
    current_overlap: usize,
}

impl<'overlapper> Iterator for Overlaps<'overlapper> {
    // NOTE: this could just return AncestryOverlap?
    type Item = (i64, i64, AncestryOverlap);
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_overlap < self.overlaps.len() {
            self.current_overlap += 1;
            Some((
                self.left,
                self.right,
                self.overlaps[self.current_overlap - 1],
            ))
        } else {
            None
        }
    }
}

fn calculate_ancestry_overlaps(
    parent: Node,
    parental_node_ancestry: &[Ancestry],
    ancestry_changes: &[AncestryChange],
) -> Vec<(i64, i64, AncestryOverlap)> {
    let mut rv_overlaps = vec![];
    let mut overlapper = AncestryOverlapper::new(parent, parental_node_ancestry, ancestry_changes);
    while let Some(overlaps) = overlapper.calculate_next_overlap_set() {
        for i in overlaps {
            rv_overlaps.push(i);
        }
    }
    rv_overlaps
}

#[test]
fn design_overlapper_test_0() {
    let genome_length = 100_i64;
    let ancestry_changes = vec![
        AncestryChange {
            node: Node(0),
            segment: Segment::new(0, genome_length).unwrap(),
            change_type: AncestryChangeType::ToUnary,
        },
        AncestryChange {
            node: Node(1),
            segment: Segment::new(0, genome_length).unwrap(),
            change_type: AncestryChangeType::ToUnary,
        },
    ];
    let parental_ancestry = vec![Ancestry {
        segment: Segment::new(0, genome_length).unwrap(),
        ancestry: AncestryType::ToSelf,
    }];
    let expected_updated_ancestry = vec![
        (
            0_i64,
            100_i64,
            AncestryOverlap {
                node: Node(3),
                segment: Segment {
                    left: 0,
                    right: 100,
                },
                source: AncestryOverlapSource::Parent(AncestryType::ToSelf),
            },
        ),
        (
            0,
            100,
            AncestryOverlap {
                node: Node(0),
                segment: Segment {
                    left: 0,
                    right: 100,
                },
                source: AncestryOverlapSource::Change(AncestryChangeType::ToUnary),
            },
        ),
        (
            0,
            100,
            AncestryOverlap {
                node: Node(1),
                segment: Segment {
                    left: 0,
                    right: 100,
                },
                source: AncestryOverlapSource::Change(AncestryChangeType::ToUnary),
            },
        ),
    ];
    let overlaps = calculate_ancestry_overlaps(Node(3), &parental_ancestry, &ancestry_changes);
    assert_eq!(overlaps, expected_updated_ancestry);
}

#[test]
fn design_overlapper_test_1() {
    let genome_length = 10_i64;
    let ancestry_changes = vec![
        AncestryChange {
            node: Node(0),
            segment: Segment::new(3, 4).unwrap(),
            change_type: AncestryChangeType::ToUnary,
        },
        AncestryChange {
            node: Node(0),
            segment: Segment::new(3, 5).unwrap(),
            change_type: AncestryChangeType::ToUnary,
        },
        AncestryChange {
            node: Node(1),
            segment: Segment::new(7, 9).unwrap(),
            change_type: AncestryChangeType::ToUnary,
        },
        AncestryChange {
            node: Node(1),
            segment: Segment::new(8, 10).unwrap(),
            change_type: AncestryChangeType::ToUnary,
        },
    ];
    let parental_ancestry = vec![Ancestry {
        segment: Segment::new(0, genome_length).unwrap(),
        ancestry: AncestryType::ToSelf,
    }];
    let expected_overlaps = [
        (
            0_i64,
            3_i64,
            AncestryOverlap {
                node: Node(3),
                segment: Segment { left: 0, right: 10 },
                source: AncestryOverlapSource::Parent(AncestryType::ToSelf),
            },
        ),
        (
            3_i64,
            4_i64,
            AncestryOverlap {
                node: Node(3),
                segment: Segment { left: 0, right: 10 },
                source: AncestryOverlapSource::Parent(AncestryType::ToSelf),
            },
        ),
        (
            3_i64,
            4_i64,
            AncestryOverlap {
                node: Node(0),
                segment: Segment { left: 3, right: 4 },
                source: AncestryOverlapSource::Change(AncestryChangeType::ToUnary),
            },
        ),
        (
            3_i64,
            4_i64,
            AncestryOverlap {
                node: Node(0),
                segment: Segment { left: 3, right: 5 },
                source: AncestryOverlapSource::Change(AncestryChangeType::ToUnary),
            },
        ),
        (
            4_i64,
            5_i64,
            AncestryOverlap {
                node: Node(3),
                segment: Segment { left: 0, right: 10 },
                source: AncestryOverlapSource::Parent(AncestryType::ToSelf),
            },
        ),
        (
            4_i64,
            5_i64,
            AncestryOverlap {
                node: Node(0),
                segment: Segment { left: 3, right: 5 },
                source: AncestryOverlapSource::Change(AncestryChangeType::ToUnary),
            },
        ),
        (
            5_i64,
            7_i64,
            AncestryOverlap {
                node: Node(3),
                segment: Segment { left: 0, right: 10 },
                source: AncestryOverlapSource::Parent(AncestryType::ToSelf),
            },
        ),
        (
            7_i64,
            8_i64,
            AncestryOverlap {
                node: Node(3),
                segment: Segment { left: 0, right: 10 },
                source: AncestryOverlapSource::Parent(AncestryType::ToSelf),
            },
        ),
        (
            7_i64,
            8_i64,
            AncestryOverlap {
                node: Node(1),
                segment: Segment { left: 7, right: 9 },
                source: AncestryOverlapSource::Change(AncestryChangeType::ToUnary),
            },
        ),
        (
            8_i64,
            9_i64,
            AncestryOverlap {
                node: Node(3),
                segment: Segment { left: 0, right: 10 },
                source: AncestryOverlapSource::Parent(AncestryType::ToSelf),
            },
        ),
        (
            8_i64,
            9_i64,
            AncestryOverlap {
                node: Node(1),
                segment: Segment { left: 7, right: 9 },
                source: AncestryOverlapSource::Change(AncestryChangeType::ToUnary),
            },
        ),
        (
            8_i64,
            9_i64,
            AncestryOverlap {
                node: Node(1),
                segment: Segment { left: 8, right: 10 },
                source: AncestryOverlapSource::Change(AncestryChangeType::ToUnary),
            },
        ),
        (
            9_i64,
            10_i64,
            AncestryOverlap {
                node: Node(3),
                segment: Segment { left: 0, right: 10 },
                source: AncestryOverlapSource::Parent(AncestryType::ToSelf),
            },
        ),
        (
            9_i64,
            10_i64,
            AncestryOverlap {
                node: Node(1),
                segment: Segment { left: 8, right: 10 },
                source: AncestryOverlapSource::Change(AncestryChangeType::ToUnary),
            },
        ),
    ];
    let overlaps = calculate_ancestry_overlaps(Node(3), &parental_ancestry, &ancestry_changes);
    assert_eq!(overlaps, expected_overlaps);
}

//        3    <- A "recent death"
//        |
//     -------
//     |     |
//     0     1 <- Births
//
// The final ancestry of 3 is Overlap(0), Overlap(1) for "whole genome".
#[test]
fn design_ancestry_update_calculation_test_0() {
    let genome_length = 100_i64;
    let ancestry_changes = vec![
        AncestryChange {
            node: Node(0),
            segment: Segment::new(0, genome_length).unwrap(),
            change_type: AncestryChangeType::ToUnary,
        },
        AncestryChange {
            node: Node(1),
            segment: Segment::new(0, genome_length).unwrap(),
            change_type: AncestryChangeType::ToUnary,
        },
    ];
    let parental_ancestry = vec![Ancestry {
        segment: Segment::new(0, genome_length).unwrap(),
        ancestry: AncestryType::ToSelf,
    }];

    let mut overlapper = AncestryOverlapper::new(Node(3), &parental_ancestry, &ancestry_changes);
    let mut new_ancestry: Vec<Ancestry> = vec![];
    // NOTE:
    // what this does NOT accomplish is determine whether
    // or not ancestry actually change in a way that needs
    // to be propagated further up the graph.
    while let Some(overlaps) = overlapper.calculate_next_overlap_set() {
        //let overlaps = overlapper.overlaps().collect::<Vec<_>>();
        // Separate parental stuff from change stuff
        let mut parental_ancestry = vec![];
        let mut changes = vec![];
        println!("{overlaps:?}");
        for o in overlaps {
            match o.2.source {
                AncestryOverlapSource::Parent(_) => parental_ancestry.push(o.2),
                AncestryOverlapSource::Change(_) => changes.push(o.2),
            }
        }
        if changes.len() > 1 {
            for c in changes {
                new_ancestry.push(Ancestry {
                    //  TODO: left/right should be getters
                    segment: Segment::new(overlapper.left, overlapper.right).unwrap(),
                    ancestry: AncestryType::Overlap(c.node),
                });
            }
        }
    }
    assert_eq!(new_ancestry.len(), 2);
    println!("{new_ancestry:?}");
    todo!("this needs to lead to an API")
}

//        0    <- Death
//        |
//     -------
//     |     |
//     |     2 <- Death (loss of ancestry)
//     |     |
//     1     3 <- Births
//
// The final ancestry of 0 is Overlap(3), Overlap(1) for "whole genome".
#[test]
fn design_ancestry_update_calculation_test_1() {
    let genome_length = 100_i64;
    let ancestry_changes = vec![
        AncestryChange {
            node: Node(1),
            segment: Segment::new(0, genome_length).unwrap(),
            change_type: AncestryChangeType::ToUnary,
        },
        AncestryChange {
            node: Node(2),
            segment: Segment::new(0, genome_length).unwrap(),
            change_type: AncestryChangeType::ToLoss,
        },
        AncestryChange {
            node: Node(3),
            segment: Segment::new(0, genome_length).unwrap(),
            change_type: AncestryChangeType::ToUnary,
        },
    ];
    let parental_ancestry = vec![
        Ancestry {
            segment: Segment::new(0, genome_length).unwrap(),
            ancestry: AncestryType::Overlap(Node(1)),
        },
        Ancestry {
            segment: Segment::new(0, genome_length).unwrap(),
            ancestry: AncestryType::Overlap(Node(2)),
        },
    ];
    let mut overlapper = AncestryOverlapper::new(Node(0), &parental_ancestry, &ancestry_changes);
    let mut new_ancestry: Vec<Ancestry> = vec![];
    // NOTE:
    // what this does NOT accomplish is determine whether
    // or not ancestry actually change in a way that needs
    // to be propagated further up the graph.
    while let Some(overlaps) = overlapper.calculate_next_overlap_set() {
        // Separate parental stuff from change stuff
        let mut parental_ancestry = vec![];
        let mut changes = vec![];
        for o in overlaps {
            match o.2.source {
                AncestryOverlapSource::Parent(_) => parental_ancestry.push(o.2),
                AncestryOverlapSource::Change(_) => changes.push(o.2),
            }
        }
        if changes.len() > 1 {
            for c in changes {
                new_ancestry.push(Ancestry {
                    //  TODO: left/right should be getters
                    segment: Segment::new(overlapper.left, overlapper.right).unwrap(),
                    ancestry: AncestryType::Overlap(c.node),
                });
            }
        }
    }
    assert_eq!(new_ancestry.len(), 2);
    assert!(new_ancestry.iter().any(|i| match i.ancestry {
        AncestryType::Overlap(x) => x == Node(1),
        _ => false,
    }));
    assert!(new_ancestry.iter().any(|i| match i.ancestry {
        AncestryType::Overlap(x) => x == Node(3),
        _ => false,
    }));
    todo!("this needs to lead to an API")
}

//        0    <- A "recent death"
//        |
//     -------
//     |     |
//     1     2 <- 1 is a birth, 2 is a death/ancestry loss
//
// The final ancestry of 0 is Empty:
// * It starts as Overlap on 1, 2
// * But the loss of ancestry in 2 and the death of 0
//   means that 0 ends up Unary(1A)
// * Since 0 is "not a sample", we do not retain
//   a single overlp.
#[test]
fn design_ancestry_update_calculation_test_2() {
    let genome_length = 100_i64;
    let ancestry_changes = vec![
        AncestryChange {
            node: Node(1),
            segment: Segment::new(0, genome_length).unwrap(),
            change_type: AncestryChangeType::ToUnary,
        },
        AncestryChange {
            node: Node(2),
            segment: Segment::new(0, genome_length).unwrap(),
            change_type: AncestryChangeType::ToLoss,
        },
    ];
    let parental_ancestry = vec![
        Ancestry {
            segment: Segment::new(0, genome_length).unwrap(),
            ancestry: AncestryType::Overlap(Node(1)),
        },
        Ancestry {
            segment: Segment::new(0, genome_length).unwrap(),
            ancestry: AncestryType::Overlap(Node(2)),
        },
    ];
    let mut overlapper = AncestryOverlapper::new(Node(0), &parental_ancestry, &ancestry_changes);
    let mut new_ancestry: Vec<Ancestry> = vec![];
    // NOTE:
    // what this does NOT accomplish is determine whether
    // or not ancestry actually change in a way that needs
    // to be propagated further up the graph.
    while let Some(overlaps) = overlapper.calculate_next_overlap_set() {
        // Separate parental stuff from change stuff
        let mut parental_ancestry = vec![];
        let mut changes = vec![];
        for o in overlaps {
            match o.2.source {
                AncestryOverlapSource::Parent(_) => parental_ancestry.push(o.2),
                AncestryOverlapSource::Change(_) => changes.push(o.2),
            }
        }
        // FAR too much nesting...
        if changes.len() > 1 {
            for c in changes {
                new_ancestry.push(Ancestry {
                    //  TODO: left/right should be getters
                    segment: Segment::new(overlapper.left, overlapper.right).unwrap(),
                    ancestry: AncestryType::Overlap(c.node),
                });
            }
        }
        // ... else have to handle unary overlap
    }
    assert!(new_ancestry.is_empty());
    todo!("this needs to lead to an API")
}

// Tree 0 is on [0, 3)
//
//        0
//        |
//     -------
//     |     |
//     1     2
//
// Tree 1 is on [7, 8)
//
//        0
//        |
//     -------
//     |     |
//     1     2
//
// * 0 starts out as Overlap(1, 2) for both trees.
// * Nodes 1, 2 lose ancestry on [0, 3)
// * Nodes 1, 2 convert to Unary on [7, 8)
//
// The result is that 0 loses all ancestry on [0, 3)
// and has no ancestry change on [7, 8)
#[test]
fn design_ancestry_update_calculation_test_3() {
    let ancestry_changes = vec![
        AncestryChange {
            node: Node(1),
            segment: Segment::new(0, 3).unwrap(),
            change_type: AncestryChangeType::ToLoss,
        },
        AncestryChange {
            node: Node(2),
            segment: Segment::new(0, 3).unwrap(),
            change_type: AncestryChangeType::ToLoss,
        },
        AncestryChange {
            node: Node(1),
            segment: Segment::new(7, 8).unwrap(),
            change_type: AncestryChangeType::ToUnary,
        },
        AncestryChange {
            node: Node(2),
            segment: Segment::new(7, 8).unwrap(),
            change_type: AncestryChangeType::ToUnary,
        },
    ];
    let parental_ancestry = vec![
        Ancestry {
            segment: Segment::new(0, 3).unwrap(),
            ancestry: AncestryType::Overlap(Node(1)),
        },
        Ancestry {
            segment: Segment::new(0, 3).unwrap(),
            ancestry: AncestryType::Overlap(Node(2)),
        },
        Ancestry {
            segment: Segment::new(7, 8).unwrap(),
            ancestry: AncestryType::Overlap(Node(1)),
        },
        Ancestry {
            segment: Segment::new(7, 8).unwrap(),
            ancestry: AncestryType::Overlap(Node(2)),
        },
    ];
    let mut overlapper = AncestryOverlapper::new(Node(0), &parental_ancestry, &ancestry_changes);
    let mut new_ancestry: Vec<Ancestry> = vec![];
    // NOTE:
    // what this does NOT accomplish is determine whether
    // or not ancestry actually change in a way that needs
    // to be propagated further up the graph.
    while let Some(overlaps) = overlapper.calculate_next_overlap_set() {
        // Separate parental stuff from change stuff
        let mut parental_ancestry = vec![];
        let mut changes = vec![];
        for o in overlaps {
            match o.2.source {
                AncestryOverlapSource::Parent(_) => parental_ancestry.push(o.2),
                AncestryOverlapSource::Change(_) => changes.push(o.2),
            }
        }
        // FAR too much nesting...
        if changes.len() > 1 {
            for c in changes {
                new_ancestry.push(Ancestry {
                    //  TODO: left/right should be getters
                    segment: Segment::new(overlapper.left, overlapper.right).unwrap(),
                    ancestry: AncestryType::Overlap(c.node),
                });
            }
        }
        // ... else have to handle unary overlap
    }
    assert_eq!(new_ancestry.len(), 2);
    let needle = Segment::new(7, 8).unwrap();
    for node in [1_usize, 2] {
        assert_eq!(
            new_ancestry
                .iter()
                .filter(|i| match i.ancestry {
                    AncestryType::Overlap(x) => x == Node(node),
                    _ => false,
                } && i.segment == needle)
                .count(),
            1
        );
    }
    todo!("this needs to lead to an API")
}

// Tree 0 is on [0, 3)
//
//        0
//        |
//     -------
//     |     |
//     1     2
//
// Tree 1 is on [7, 8)
//
//        0
//        |
//     -------
//     |     |
//     1     2
//
// * 0 starts out as Overlap(1, 2) for both trees.
// * Node 1 loses ancestry on both segments
// * Node 2 converts to Unary on both segments
//
// The result is that 0 loses all ancestry 
// because all ancestry becomes unary and 0
// is "not a sample"
#[test]
fn design_ancestry_update_calculation_test_4() {
    let ancestry_changes = vec![
        AncestryChange {
            node: Node(1),
            segment: Segment::new(0, 3).unwrap(),
            change_type: AncestryChangeType::ToLoss,
        },
        AncestryChange {
            node: Node(2),
            segment: Segment::new(0, 3).unwrap(),
            change_type: AncestryChangeType::ToUnary,
        },
        AncestryChange {
            node: Node(1),
            segment: Segment::new(7, 8).unwrap(),
            change_type: AncestryChangeType::ToLoss,
        },
        AncestryChange {
            node: Node(2),
            segment: Segment::new(7, 8).unwrap(),
            change_type: AncestryChangeType::ToUnary,
        },
    ];
    let parental_ancestry = vec![
        Ancestry {
            segment: Segment::new(0, 3).unwrap(),
            ancestry: AncestryType::Overlap(Node(1)),
        },
        Ancestry {
            segment: Segment::new(0, 3).unwrap(),
            ancestry: AncestryType::Overlap(Node(2)),
        },
        Ancestry {
            segment: Segment::new(7, 8).unwrap(),
            ancestry: AncestryType::Overlap(Node(1)),
        },
        Ancestry {
            segment: Segment::new(7, 8).unwrap(),
            ancestry: AncestryType::Overlap(Node(2)),
        },
    ];
    let mut overlapper = AncestryOverlapper::new(Node(0), &parental_ancestry, &ancestry_changes);
    let mut new_ancestry: Vec<Ancestry> = vec![];
    // NOTE:
    // what this does NOT accomplish is determine whether
    // or not ancestry actually change in a way that needs
    // to be propagated further up the graph.
    while let Some(overlaps) = overlapper.calculate_next_overlap_set() {
        // Separate parental stuff from change stuff
        let mut parental_ancestry = vec![];
        let mut changes = vec![];
        for o in overlaps {
            match o.2.source {
                AncestryOverlapSource::Parent(_) => parental_ancestry.push(o.2),
                AncestryOverlapSource::Change(_) => changes.push(o.2),
            }
        }
        // FAR too much nesting...
        if changes.len() > 1 {
            for c in changes {
                new_ancestry.push(Ancestry {
                    //  TODO: left/right should be getters
                    segment: Segment::new(overlapper.left, overlapper.right).unwrap(),
                    ancestry: AncestryType::Overlap(c.node),
                });
            }
        }
        // ... else have to handle unary overlap
    }
    assert!(new_ancestry.is_empty());
    todo!("this needs to lead to an API")
}
#[test]
fn test_ancestry_change_ordering() {
    {
        let mut changes = vec![
            AncestryChange {
                node: Node(1),
                segment: Segment::new(0, 10).unwrap(),
                change_type: AncestryChangeType::ToUnary,
            },
            AncestryChange {
                node: Node(0),
                segment: Segment::new(0, 10).unwrap(),
                change_type: AncestryChangeType::ToUnary,
            },
        ];
        changes.sort_unstable();
        assert_eq!(
            changes,
            [
                AncestryChange {
                    node: Node(0),
                    segment: Segment::new(0, 10).unwrap(),
                    change_type: AncestryChangeType::ToUnary
                },
                AncestryChange {
                    node: Node(1),
                    segment: Segment::new(0, 10).unwrap(),
                    change_type: AncestryChangeType::ToUnary
                },
            ]
        );
    }

    {
        let mut changes = vec![
            AncestryChange {
                node: Node(1),
                segment: Segment::new(0, 10).unwrap(),
                change_type: AncestryChangeType::ToUnary,
            },
            AncestryChange {
                node: Node(0),
                segment: Segment::new(6, 10).unwrap(),
                change_type: AncestryChangeType::ToUnary,
            },
        ];
        changes.sort_unstable();
        assert_eq!(
            changes,
            [
                AncestryChange {
                    node: Node(1),
                    segment: Segment::new(0, 10).unwrap(),
                    change_type: AncestryChangeType::ToUnary,
                },
                AncestryChange {
                    node: Node(0),
                    segment: Segment::new(6, 10).unwrap(),
                    change_type: AncestryChangeType::ToUnary,
                },
            ]
        );
    }

    {
        let mut changes = vec![
            AncestryChange {
                node: Node(0),
                segment: Segment::new(6, 7).unwrap(),
                change_type: AncestryChangeType::ToUnary,
            },
            AncestryChange {
                node: Node(0),
                segment: Segment::new(6, 10).unwrap(),
                change_type: AncestryChangeType::ToUnary,
            },
        ];
        changes.sort_unstable();
        assert_eq!(
            changes,
            [
                AncestryChange {
                    node: Node(0),
                    segment: Segment::new(6, 7).unwrap(),
                    change_type: AncestryChangeType::ToUnary,
                },
                AncestryChange {
                    node: Node(0),
                    segment: Segment::new(6, 10).unwrap(),
                    change_type: AncestryChangeType::ToUnary,
                },
            ]
        );
    }
}
