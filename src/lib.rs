use nohash::BuildNoHashHasher;
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::BuildHasherDefault;

// NOTE: for design purposes -- delete later.
mod overlapper_experiments;

mod flags;

use flags::NodeFlags;

#[repr(transparent)]
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, PartialOrd, Ord)]
struct Node(usize);

impl Node {
    fn as_index(&self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeStatus {
    Ancestor,
    Birth,
    Death,
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

    fn sentinel() -> Self {
        Self {
            left: i64::MAX,
            right: i64::MAX,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
    Overlap,
    // NOTE: ToUnary is overloaded with "Unary and is sample"
    // We may want to un-overload the term later
    Unary,
    Loss,
}

#[derive(Copy, Clone, Debug)]
struct Transmission {
    parent: Node,
    child: Node,
    left: i64,
    right: i64,
}

#[derive(Debug)]
struct Graph {
    status: Vec<NodeStatus>,
    flags: Vec<NodeFlags>,
    birth_time: Vec<Option<i64>>,
    parents: Vec<NodeHash>,
    // NOTE: for many scenarios, it may be preferable
    // to manage a Vec<(Node, Segment)> as the inner
    // value. We would sort the inner Vec during simplification.
    // The potential plus is that we'd avoid many hash lookups
    // during evolution. Also, sorting small Vectors tends
    // to be really fast.
    transmissions: Vec<Transmission>,
    ancestry: Vec<Vec<Ancestry>>,
    // Used to cache nodes that are born.
    // Simplification will traverse this.
    births: Vec<Node>,
    deaths: Vec<Node>,
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
        let flags = Vec::with_capacity(capacity);
        let birth_time = Vec::with_capacity(capacity);
        let children = Vec::with_capacity(capacity);
        let ancestry = Vec::with_capacity(capacity);
        let births = vec![];
        let deaths = vec![];
        let free_nodes = Vec::new();
        Some(Self {
            status,
            flags,
            birth_time,
            parents,
            transmissions: children,
            ancestry,
            births,
            deaths,
            free_nodes,
            genome_length,
        })
    }

    // TODO: remove code duplication to with_capacity if we keep this fn
    fn with_initial_nodes(num_nodes: usize, genome_length: i64) -> Option<Self> {
        let status = vec![NodeStatus::Ancestor; num_nodes];
        let flags = vec![NodeFlags::default(); num_nodes];
        let birth_time = vec![Some(0); num_nodes];
        let parents = vec![NodeHash::with_hasher(BuildHasherDefault::default()); num_nodes];
        let transmissions = vec![];
        let initial_ancestry = Ancestry {
            segment: Segment::new(0, genome_length)?,
            ancestry: AncestryType::ToSelf,
        };
        let ancestry = vec![vec![initial_ancestry]; num_nodes];
        let births = vec![];
        let deaths = vec![];
        let free_nodes = Vec::new();
        Some(Self {
            status,
            flags,
            birth_time,
            parents,
            transmissions,
            ancestry,
            births,
            deaths,
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
        match self.status[node.as_index()] {
            NodeStatus::Birth => vec![AncestryChange {
                node,
                segment: Segment {
                    left: 0,
                    right: self.genome_length(),
                },
                change_type: AncestryChangeType::Unary,
            }],
            _ => todo!(),
        }
    }

    fn birth_ancestry_change_type(&self, child: Node) -> AncestryChange {
        AncestryChange {
            segment: Segment {
                left: 0,
                right: self.genome_length,
            },
            node: child,
            change_type: AncestryChangeType::Unary,
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
        debug_assert_eq!(self.birth_time[rv.as_index()], Some(birth_time));
        // births are (locally) in increasing order
        match self.births.len() {
            i if i > 0 => {
                if self.birth_time[self.births[self.births.len() - 1].as_index()].unwrap()
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
            .get(parent.as_index())
            .ok_or(())?
            .ok_or(())?;
        let ctime = self.birth_time.get(child.as_index()).ok_or(())?.ok_or(())?;
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
        // We now "know" that parent, child are both in range.
        // (The only uncertainty is that we haven't checked that all our arrays
        //  are equal length.)
        self.transmissions.push(Transmission {
            parent,
            child,
            left,
            right,
        });
        let _ = self.parents[child.as_index()].insert(parent);

        Ok(())
    }

    // NOTE: panics if child is out of bounds
    // NOTE: may not need to be pub
    pub fn parents(&self, child: Node) -> impl Iterator<Item = &Node> + '_ {
        self.parents[child.as_index()].iter()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum AncestryOverlap {
    Parental(Ancestry),
    Change(AncestryChange),
}

impl AncestryOverlap {
    fn left(&self) -> i64 {
        match self {
            AncestryOverlap::Parental(p) => p.segment.left,
            AncestryOverlap::Change(c) => c.segment.left,
        }
    }
    fn right(&self) -> i64 {
        match self {
            AncestryOverlap::Parental(p) => p.segment.left,
            AncestryOverlap::Change(c) => c.segment.left,
        }
    }
}

fn queue_identical_parental_segments(
    current_parental_node: &Ancestry,
    parental_node_ancestry: &[Ancestry],
    queue: &mut Vec<AncestryOverlap>,
) -> usize {
    parental_node_ancestry
        .iter()
        .take_while(|x| x.identical_segment(current_parental_node))
        .inspect(|&x| queue.push(AncestryOverlap::Parental(*x)))
        .count()
}

fn generate_overlap_queue(
    parent: Node,
    parental_node_ancestry: &[Ancestry],
    ancestry_changes: &[AncestryChange],
) -> Vec<AncestryOverlap> {
    // TODO: should have a way to detect this and return an Error
    assert!(parental_node_ancestry.windows(2).all(|w| w[0] <= w[1]));
    assert!(
        ancestry_changes
            .windows(2)
            .all(|w| w[0].left() <= w[1].left()),
        "{ancestry_changes:?}"
    );
    assert!(!ancestry_changes.is_empty());
    let mut queue = vec![];
    let mut d = 0_usize;

    while d < parental_node_ancestry.len() {
        // Take the current node here to minimize bounds checks
        let current_parental_node = &parental_node_ancestry[d];
        let update = queue_identical_parental_segments(
            current_parental_node,
            // Another bounds check...
            &parental_node_ancestry[d..],
            &mut queue,
        );
        for ac in ancestry_changes.iter() {
            if ac.right() > current_parental_node.left()
                && current_parental_node.right() > ac.left()
            {
                let left = std::cmp::max(ac.left(), current_parental_node.left());
                let right = std::cmp::min(ac.right(), current_parental_node.right());
                queue.push(AncestryOverlap::Change(AncestryChange {
                    segment: Segment::new(left, right).unwrap(),
                    node: ac.node,
                    change_type: ac.change_type,
                }));
            }
        }
        d += update;
    }

    // TODO: should be an error?.
    // But, an error/assert means that,
    // internally, we MUST not send
    // "no changes" up to parents.
    assert!(!queue.is_empty());

    debug_assert!(queue.windows(2).all(|w| w[0].left() <= w[1].left()));

    queue
}

struct AncestryOverlapper {
    queue: Vec<AncestryOverlap>,
    num_overlaps: usize,
    current_overlap: usize,
    parent: Node,
    left: i64,
    right: i64,
    overlaps: Vec<Ancestry>,
    output_nodes: Vec<Node>,
    parental_overlaps: Vec<Ancestry>,
    change_overlaps: Vec<AncestryChange>,
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
        queue.push(AncestryOverlap::Parental(Ancestry {
            segment: Segment::sentinel(),
            ancestry: AncestryType::ToSelf,
        }));
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
            parent,
            left,
            right,
            overlaps: vec![],
            output_nodes: vec![],
            parental_overlaps: vec![],
            change_overlaps: vec![],
        }
    }

    fn update_right_from_overlaps(&mut self) {
        self.right = match self
            .parental_overlaps
            .iter()
            .map(|&overlap| overlap.right())
            .chain(self.change_overlaps.iter().map(|&overlap| overlap.right()))
            .min()
        {
            Some(right) => right,
            None => self.right,
        };
    }

    fn filter_overlaps(&mut self) {
        self.parental_overlaps.retain(|x| x.right() > self.left);
        self.change_overlaps.retain(|x| x.right() > self.left);
    }

    fn make_overlaps(&mut self) -> Overlaps {
        let ancestry_change = output_overlaps(
            self.left,
            self.right,
            self.parent,
            &self.parental_overlaps,
            &self.change_overlaps,
            &mut self.overlaps,
            &mut self.output_nodes,
        );
        Overlaps::new(
            self.left,
            self.right,
            self.parent,
            &self.overlaps,
            ancestry_change,
        )
    }

    fn calculate_next_overlap_set(&mut self) -> Option<Overlaps> {
        if self.current_overlap < self.num_overlaps {
            self.left = self.right;
            self.filter_overlaps();

            // TODO: this should be a function call
            if self.parental_overlaps.len() + self.change_overlaps.len() == 0 {
                self.left = self.queue[self.current_overlap].left();
            }
            self.current_overlap += self
                .queue
                .iter()
                .skip(self.current_overlap)
                .take_while(|x| x.left() == self.left)
                .inspect(|&&x| {
                    self.right = std::cmp::min(self.right, x.right());
                    match x {
                        AncestryOverlap::Parental(o) => self.parental_overlaps.push(o),
                        AncestryOverlap::Change(o) => self.change_overlaps.push(o),
                    }
                })
                .count();
            self.update_right_from_overlaps();
            self.right = std::cmp::min(self.right, self.queue[self.current_overlap].left());
            Some(self.make_overlaps())
        } else {
            if self.parental_overlaps.len() + self.change_overlaps.len() != 0 {
                self.left = self.right;
                self.filter_overlaps();
            }
            if self.parental_overlaps.len() + self.change_overlaps.len() != 0 {
                self.update_right_from_overlaps();
                Some(self.make_overlaps())
            } else {
                None
            }
        }
    }

    fn output_ancestry(&mut self) -> Option<Overlaps<'_>> {
        self.calculate_next_overlap_set()
    }
}

fn calculate_ancestry_change(
    left: i64,
    right: i64,
    parent: Node,
    parental_overlaps: &[Ancestry],
    overlaps: &[Ancestry],
) -> Option<AncestryChange> {
    let (parental_node, parental_ancestry) = match parental_overlaps.first() {
        Some(ancestry) => (parent, ancestry.ancestry),
        None => panic!(),
    };

    let change_type = match parental_ancestry {
        AncestryType::Overlap(_) => {
            assert!(parental_overlaps.len() > 1);
            if overlaps.len() > 1 {
                None
            }
            // else if overlaps.len() == 1 {
            //     Some(AncestryChangeType::ToLoss)
            // }
            else {
                Some(AncestryChangeType::Loss)
            }
        }
        AncestryType::Unary(_) | AncestryType::ToSelf => {
            assert!(parental_overlaps.len() == 1);
            if overlaps.len() > 1 {
                Some(AncestryChangeType::Overlap)
            }
            //else if overlaps.len() == 1 {
            //    Some(AncestryChangeType::ToLoss)
            //}
            else {
                Some(AncestryChangeType::Loss)
            }
        }
    };

    change_type.map(|change_type| AncestryChange {
        segment: Segment { left, right },
        node: parental_node,
        change_type,
    })
}

fn output_overlaps(
    left: i64,
    right: i64,
    parent: Node,
    parental_overlaps: &[Ancestry],
    change_overlaps: &[AncestryChange],
    output_ancestry: &mut Vec<Ancestry>,
    output_nodes: &mut Vec<Node>,
) -> Option<AncestryChange> {
    output_ancestry.clear();
    output_nodes.clear();
    for co in change_overlaps {
        if !matches!(co.change_type, AncestryChangeType::Loss) {
            output_ancestry.push(Ancestry {
                segment: Segment::new(left, right).unwrap(),
                ancestry: AncestryType::Overlap(co.node),
            })
        }
        output_nodes.push(co.node);
    }
    for po in parental_overlaps {
        match po.ancestry {
            AncestryType::ToSelf => (),
            AncestryType::Unary(node) | AncestryType::Overlap(node) => {
                if !output_nodes.contains(&node) {
                    output_ancestry.push(Ancestry {
                        segment: Segment::new(left, right).unwrap(),
                        ancestry: AncestryType::Overlap(node),
                    });
                }
            }
        }
    }
    calculate_ancestry_change(left, right, parent, parental_overlaps, output_ancestry)
}

// TODO: we need some data
// here about the parent (sample?, etc.)
// and overall policies (keep unary, etc.)
#[derive(Debug)]
struct Overlaps<'overlapper> {
    left: i64,
    right: i64,
    parent: Node,
    parental_ancestry_change: Option<AncestryChange>,
    overlaps: &'overlapper [Ancestry],
}

impl<'overlapper> Overlaps<'overlapper> {
    fn new(
        left: i64,
        right: i64,
        parent: Node,
        overlaps: &'overlapper [Ancestry],
        parental_ancestry_change: Option<AncestryChange>,
    ) -> Self {
        Self {
            left,
            right,
            parent,
            parental_ancestry_change,
            overlaps,
        }
    }

    fn iter_overlaps(&self) -> Option<impl Iterator<Item = &Ancestry> + '_> {
        match self.parental_ancestry_change {
            Some(x) => match x.change_type {
                AncestryChangeType::Loss => None,
                _ => Some(self.overlaps.iter()),
            },
            _ => Some(self.overlaps.iter()),
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
    let parent = graph.add_node(NodeStatus::Ancestor, 0);
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
    assert_eq!(graph.transmissions.len(), 2);
}

#[test]
fn test_births_out_of_order() {
    let mut graph = Graph::new(100).unwrap();
    let _ = graph.add_node(NodeStatus::Ancestor, 0);
    let _ = graph.add_birth(2).unwrap();
    assert!(graph.add_birth(1).is_err());
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
            change_type: AncestryChangeType::Unary,
        },
        AncestryChange {
            node: Node(1),
            segment: Segment::new(0, genome_length).unwrap(),
            change_type: AncestryChangeType::Unary,
        },
    ];
    let parental_ancestry = vec![Ancestry {
        segment: Segment::new(0, genome_length).unwrap(),
        ancestry: AncestryType::ToSelf,
    }];

    let mut overlapper = AncestryOverlapper::new(Node(3), &parental_ancestry, &ancestry_changes);
    let mut new_ancestry: Vec<Ancestry> = vec![];
    let mut changes = vec![];

    while let Some(overlaps) = overlapper.output_ancestry() {
        changes.push(overlaps.parental_ancestry_change);
        if let Some(iter) = overlaps.iter_overlaps() {
            for &a in iter {
                new_ancestry.push(a);
            }
        }
    }

    assert_eq!(new_ancestry.len(), 2);
    for node in [0, 1] {
        assert!(new_ancestry.contains(&Ancestry {
            segment: Segment {
                left: 0,
                right: 100
            },
            ancestry: AncestryType::Overlap(Node(node))
        }));
    }
    assert!(changes.contains(&Some(AncestryChange {
        segment: Segment {
            left: 0,
            right: 100
        },
        node: Node(3),
        change_type: AncestryChangeType::Overlap
    })),);
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
            change_type: AncestryChangeType::Unary,
        },
        AncestryChange {
            node: Node(2),
            segment: Segment::new(0, genome_length).unwrap(),
            change_type: AncestryChangeType::Loss,
        },
        AncestryChange {
            node: Node(3),
            segment: Segment::new(0, genome_length).unwrap(),
            change_type: AncestryChangeType::Unary,
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
    let mut changes = vec![];

    while let Some(overlaps) = overlapper.output_ancestry() {
        changes.push(overlaps.parental_ancestry_change);
        if let Some(iter) = overlaps.iter_overlaps() {
            for &a in iter {
                new_ancestry.push(a);
            }
        }
    }
    assert_eq!(new_ancestry.len(), 2);
    for node in [3, 1] {
        assert!(new_ancestry.contains(&Ancestry {
            segment: Segment {
                left: 0,
                right: 100
            },
            ancestry: AncestryType::Overlap(Node(node))
        }));
    }
    assert_eq!(changes, &[None]);
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
//   a single overlap.
#[test]
fn design_ancestry_update_calculation_test_2() {
    let genome_length = 100_i64;
    let ancestry_changes = vec![
        AncestryChange {
            node: Node(1),
            segment: Segment::new(0, genome_length).unwrap(),
            change_type: AncestryChangeType::Unary,
        },
        AncestryChange {
            node: Node(2),
            segment: Segment::new(0, genome_length).unwrap(),
            change_type: AncestryChangeType::Loss,
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
    let mut changes = vec![];

    while let Some(overlaps) = overlapper.output_ancestry() {
        changes.push(overlaps.parental_ancestry_change);
        if let Some(iter) = overlaps.iter_overlaps() {
            for &a in iter {
                new_ancestry.push(a);
            }
        }
    }
    assert!(new_ancestry.is_empty());
    assert_eq!(
        changes,
        &[Some(AncestryChange {
            segment: Segment {
                left: 0,
                right: 100
            },
            node: Node(0),
            change_type: AncestryChangeType::Loss
        })],
        "CHANGES = {changes:?}"
    )
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
            change_type: AncestryChangeType::Loss,
        },
        AncestryChange {
            node: Node(2),
            segment: Segment::new(0, 3).unwrap(),
            change_type: AncestryChangeType::Loss,
        },
        AncestryChange {
            node: Node(1),
            segment: Segment::new(7, 8).unwrap(),
            change_type: AncestryChangeType::Unary,
        },
        AncestryChange {
            node: Node(2),
            segment: Segment::new(7, 8).unwrap(),
            change_type: AncestryChangeType::Unary,
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
    let mut changes = vec![];

    while let Some(overlaps) = overlapper.output_ancestry() {
        changes.push(overlaps.parental_ancestry_change);
        if let Some(iter) = overlaps.iter_overlaps() {
            for &a in iter {
                new_ancestry.push(a);
            }
        }
    }

    assert_eq!(new_ancestry.len(), 2);
    for node in [1_usize, 2] {
        assert!(new_ancestry.contains(&Ancestry {
            segment: Segment { left: 7, right: 8 },
            ancestry: AncestryType::Overlap(Node(node))
        }));
    }
    assert_eq!(changes.len(), 2);
    assert!(changes.contains(&None));
    assert!(
        changes.contains(&Some(AncestryChange {
            segment: Segment { left: 0, right: 3 },
            node: Node(0),
            change_type: AncestryChangeType::Loss
        })),
        "CHANGES = {changes:?}"
    );
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
            change_type: AncestryChangeType::Loss,
        },
        AncestryChange {
            node: Node(2),
            segment: Segment::new(0, 3).unwrap(),
            change_type: AncestryChangeType::Unary,
        },
        AncestryChange {
            node: Node(1),
            segment: Segment::new(7, 8).unwrap(),
            change_type: AncestryChangeType::Loss,
        },
        AncestryChange {
            node: Node(2),
            segment: Segment::new(7, 8).unwrap(),
            change_type: AncestryChangeType::Unary,
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
    let mut changes = vec![];

    while let Some(overlaps) = overlapper.output_ancestry() {
        changes.push(overlaps.parental_ancestry_change);
        if let Some(iter) = overlaps.iter_overlaps() {
            for &a in iter {
                new_ancestry.push(a);
            }
        }
    }
    assert!(new_ancestry.is_empty());
    assert_eq!(changes.len(), 2);
    for losses in [(0, 3), (7, 8)] {
        assert!(
            changes.contains(&Some(AncestryChange {
                segment: Segment {
                    left: losses.0,
                    right: losses.1
                },
                node: Node(0),
                change_type: AncestryChangeType::Loss
            })),
            "CHANGES = {changes:?}"
        );
    }
}

// Tree 0 is on [0, 3)
//
//        0    <- Born "ancestral to time 1"
//        |
//        ------------
//        |    |     |
//        |    3     4 <- Nodes that were born at time 1
//        |
//     -------
//     |     |
//     1     2 <- births as time 2
//
// Tree 1 is on [7, 8)
//
//        0
//        |
//     -------
//     |     |
//     3     4 <- Nodes that were born at time 1
//
// * 0 starts out as Overlap(3, 4) on [7, 8)
// * Nodes 1 and 2 send ancestry gains up the tree
// * Nodes 3 and 4 send no changes up the tree.
//
// The result is that 0 gains Overlap(1, 2) on [0, 3)
// and remains Overlap(3, 4) on [0, 3) and [7, 8).
#[test]
fn design_ancestry_update_calculation_test_5() {
    let ancestry_changes = vec![
        AncestryChange {
            node: Node(1),
            segment: Segment::new(0, 3).unwrap(),
            change_type: AncestryChangeType::Unary,
        },
        AncestryChange {
            node: Node(2),
            segment: Segment::new(0, 3).unwrap(),
            change_type: AncestryChangeType::Unary,
        },
    ];
    let parental_ancestry = vec![
        Ancestry {
            segment: Segment::new(0, 3).unwrap(),
            ancestry: AncestryType::Overlap(Node(3)),
        },
        Ancestry {
            segment: Segment::new(0, 3).unwrap(),
            ancestry: AncestryType::Overlap(Node(4)),
        },
        Ancestry {
            segment: Segment::new(7, 8).unwrap(),
            ancestry: AncestryType::Overlap(Node(3)),
        },
        Ancestry {
            segment: Segment::new(7, 8).unwrap(),
            ancestry: AncestryType::Overlap(Node(4)),
        },
    ];
    let mut overlapper = AncestryOverlapper::new(Node(0), &parental_ancestry, &ancestry_changes);
    let mut new_ancestry: Vec<Ancestry> = vec![];
    let mut changes = vec![];

    while let Some(overlaps) = overlapper.output_ancestry() {
        changes.push(overlaps.parental_ancestry_change);
        if let Some(iter) = overlaps.iter_overlaps() {
            for &a in iter {
                new_ancestry.push(a);
            }
        }
    }
    assert_eq!(new_ancestry.len(), 6);
    for node in [1_usize, 2, 3, 4] {
        let needle = Segment::new(0, 3).unwrap();
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
    for node in [3, 4] {
        let needle = Segment::new(7, 8).unwrap();
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
    for node in [1, 2] {
        let needle = Segment::new(7, 8).unwrap();
        assert_eq!(
            new_ancestry
                .iter()
                .filter(|i| match i.ancestry {
                    AncestryType::Overlap(x) => x == Node(node),
                    _ => false,
                } && i.segment == needle)
                .count(),
            0
        );
    }
}

// Tree 0 is on [0, 3)
//
//        0    <- Born "ancestral to time 1"
//        |
//        -----------
//        |   |     |
//        |   1     2 <- births as time 1
//        |
//        3           <- birth at time 2 on [0, 5)
// Tree 1 is on [7, 8)
//
//        0    <- Born "ancestral to time 1"
//        |
//        -----------
//            |     |
//            1     2 <- births as time 1
//
// * 0 starts out as Overlap(1, 2) on [0, 3) and [7, 8)
// * Nodes 3 send ancestry gains up the tree
// * Nodes 1 and 2 send no changes up the tree.
//
// The result is that 0 gains Overlap(3) on [0, 3)
// and remains Overlap(3, 4) on [0, 3) and [7, 8).
#[test]
fn design_ancestry_update_calculation_test_6() {
    let ancestry_changes = vec![AncestryChange {
        node: Node(3),
        segment: Segment::new(0, 5).unwrap(),
        change_type: AncestryChangeType::Unary,
    }];
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
    let mut changes = vec![];

    while let Some(overlaps) = overlapper.output_ancestry() {
        changes.push(overlaps.parental_ancestry_change);
        if let Some(iter) = overlaps.iter_overlaps() {
            for &a in iter {
                new_ancestry.push(a);
            }
        }
    }
    assert_eq!(new_ancestry.len(), 5);
    for node in [1_usize, 2, 3] {
        let needle = Segment::new(0, 3).unwrap();
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
    for node in [1, 2] {
        let needle = Segment::new(7, 8).unwrap();
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
    for node in [3] {
        let needle = Segment::new(7, 8).unwrap();
        assert_eq!(
            new_ancestry
                .iter()
                .filter(|i| match i.ancestry {
                    AncestryType::Overlap(x) => x == Node(node),
                    _ => false,
                } && i.segment == needle)
                .count(),
            0
        );
    }
    assert_eq!(changes, [None, None]);
}

// Note: the result of this kind of action
// is a loss of ancestry.
#[test]
fn design_ancestry_update_calculation_test_7() {
    let ancestry_changes = vec![AncestryChange {
        node: Node(3),
        segment: Segment::new(0, 5).unwrap(),
        change_type: AncestryChangeType::Unary,
    }];
    let parental_ancestry = vec![Ancestry {
        segment: Segment::new(0, 3).unwrap(),
        ancestry: AncestryType::ToSelf,
    }];
    let mut overlapper = AncestryOverlapper::new(Node(0), &parental_ancestry, &ancestry_changes);
    let mut new_ancestry: Vec<Ancestry> = vec![];
    let mut changes = vec![];

    while let Some(overlaps) = overlapper.output_ancestry() {
        changes.push(overlaps.parental_ancestry_change);
        if let Some(iter) = overlaps.iter_overlaps() {
            for &a in iter {
                new_ancestry.push(a);
            }
        }
    }
    assert!(new_ancestry.is_empty());
    assert_eq!(changes.len(), 1);
    assert!(changes.contains(&Some(AncestryChange {
        segment: Segment { left: 0, right: 3 },
        node: Node(0),
        change_type: AncestryChangeType::Loss
    })));
}

// Tree 0 is on [0, 3)
//
//        0
//        |
//     -------
//     |     |
//     1     2
//
// Tree 1 is on [3, 8)
//
//        0
//        |
//     -------
//     |     |
//     1     2
//
// * 0 starts out as Overlap(1, 2) for both trees.
// * Node 1, 2 loses ancestry on [0, 3)
//
// The result is that 0 loses all ancestry on [3, 8)
// and remains overlap on [0, 3)
#[test]
fn design_ancestry_update_calculation_test_8() {
    let parental_ancestry = vec![
        Ancestry {
            segment: Segment::new(0, 8).unwrap(),
            ancestry: AncestryType::Overlap(Node(1)),
        },
        Ancestry {
            segment: Segment::new(0, 8).unwrap(),
            ancestry: AncestryType::Overlap(Node(2)),
        },
    ];
    let ancestry_changes = vec![
        AncestryChange {
            node: Node(1),
            segment: Segment::new(3, 8).unwrap(),
            change_type: AncestryChangeType::Loss,
        },
        AncestryChange {
            node: Node(2),
            segment: Segment::new(3, 8).unwrap(),
            change_type: AncestryChangeType::Loss,
        },
    ];
    let mut overlapper = AncestryOverlapper::new(Node(0), &parental_ancestry, &ancestry_changes);
    let mut new_ancestry: Vec<Ancestry> = vec![];
    let mut changes = vec![];

    while let Some(overlaps) = overlapper.output_ancestry() {
        changes.push(overlaps.parental_ancestry_change);
        for &a in overlaps.overlaps.iter() {
            new_ancestry.push(a);
        }
    }
    assert_eq!(new_ancestry.len(), 2);
    for node in [1, 2] {
        assert!(new_ancestry.contains(&Ancestry {
            segment: Segment { left: 0, right: 3 },
            ancestry: AncestryType::Overlap(Node(node))
        }))
    }
    assert_eq!(changes.len(), 2);
    assert!(changes.contains(&None));
    assert!(changes.contains(&Some(AncestryChange {
        segment: Segment { left: 3, right: 8 },
        node: Node(0),
        change_type: AncestryChangeType::Loss
    })));
}

#[test]
fn test_ancestry_change_ordering() {
    {
        let mut changes = vec![
            AncestryChange {
                node: Node(1),
                segment: Segment::new(0, 10).unwrap(),
                change_type: AncestryChangeType::Unary,
            },
            AncestryChange {
                node: Node(0),
                segment: Segment::new(0, 10).unwrap(),
                change_type: AncestryChangeType::Unary,
            },
        ];
        changes.sort_unstable();
        assert_eq!(
            changes,
            [
                AncestryChange {
                    node: Node(0),
                    segment: Segment::new(0, 10).unwrap(),
                    change_type: AncestryChangeType::Unary
                },
                AncestryChange {
                    node: Node(1),
                    segment: Segment::new(0, 10).unwrap(),
                    change_type: AncestryChangeType::Unary
                },
            ]
        );
    }

    {
        let mut changes = vec![
            AncestryChange {
                node: Node(1),
                segment: Segment::new(0, 10).unwrap(),
                change_type: AncestryChangeType::Unary,
            },
            AncestryChange {
                node: Node(0),
                segment: Segment::new(6, 10).unwrap(),
                change_type: AncestryChangeType::Unary,
            },
        ];
        changes.sort_unstable();
        assert_eq!(
            changes,
            [
                AncestryChange {
                    node: Node(1),
                    segment: Segment::new(0, 10).unwrap(),
                    change_type: AncestryChangeType::Unary,
                },
                AncestryChange {
                    node: Node(0),
                    segment: Segment::new(6, 10).unwrap(),
                    change_type: AncestryChangeType::Unary,
                },
            ]
        );
    }

    {
        let mut changes = vec![
            AncestryChange {
                node: Node(0),
                segment: Segment::new(6, 7).unwrap(),
                change_type: AncestryChangeType::Unary,
            },
            AncestryChange {
                node: Node(0),
                segment: Segment::new(6, 10).unwrap(),
                change_type: AncestryChangeType::Unary,
            },
        ];
        changes.sort_unstable();
        assert_eq!(
            changes,
            [
                AncestryChange {
                    node: Node(0),
                    segment: Segment::new(6, 7).unwrap(),
                    change_type: AncestryChangeType::Unary,
                },
                AncestryChange {
                    node: Node(0),
                    segment: Segment::new(6, 10).unwrap(),
                    change_type: AncestryChangeType::Unary,
                },
            ]
        );
    }
}

#[cfg(test)]
mod test_ancestry_change_propagation {
    use super::*;

    //BOILER PLATE ALERT
    #[derive(Debug)]
    struct QueuedNode {
        node: Node,
        birth_time: i64,
    }

    impl PartialEq for QueuedNode {
        fn eq(&self, other: &Self) -> bool {
            self.node == other.node
        }
    }

    impl Eq for QueuedNode {}

    impl PartialOrd for QueuedNode {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.birth_time.partial_cmp(&other.birth_time)
        }
    }

    impl Ord for QueuedNode {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.birth_time.cmp(&other.birth_time)
        }
    }

    fn update_internal_stuff(
        node: Node,
        hashed_nodes: &mut HashSet<Node>,
        parent_queue: &mut std::collections::BinaryHeap<QueuedNode>,
        graph: &Graph,
    ) {
        if !hashed_nodes.contains(&node) {
            hashed_nodes.insert(node);
            parent_queue.push(QueuedNode {
                node,
                birth_time: graph.birth_time[node.as_index()].unwrap(),
            });
        }
    }

    fn push_ancestry_changes_to_parent<I: Iterator<Item = AncestryChange>>(
        parent: Node,
        ancestry_changes: I,
        ancestry_changes_to_process: &mut HashMap<Node, Vec<AncestryChange>>,
    ) {
        match ancestry_changes_to_process.get_mut(&parent) {
            Some(changes) => changes.extend(ancestry_changes),
            None => {
                ancestry_changes_to_process.insert(parent, Vec::from_iter(ancestry_changes));
            }
        }
    }

    fn process_node_death(
        queued_parent: QueuedNode,
        hashed_nodes: &mut HashSet<Node>,
        parent_queue: &mut std::collections::BinaryHeap<QueuedNode>,
        ancestry_changes_to_process: &mut HashMap<Node, Vec<AncestryChange>>,
        graph: &mut Graph,
    ) {
        let changes_for_this_node = match ancestry_changes_to_process.get(&queued_parent.node) {
            Some(changes) => changes.to_vec(),
            None => vec![],
        };
        println!("this dead node should pass on {changes_for_this_node:?}");
        for parent in graph.parents(queued_parent.node) {
            println!("parent of death node = {parent:?}");
            println!(
                "current changes for parent = {:?}",
                ancestry_changes_to_process.get(parent)
            );
            update_internal_stuff(*parent, hashed_nodes, parent_queue, graph);
            push_ancestry_changes_to_parent(
                *parent,
                //ancestry_changes,
                graph.ancestry[queued_parent.node.as_index()]
                    .iter()
                    .map(|a| AncestryChange {
                        segment: a.segment,
                        node: queued_parent.node,
                        change_type: AncestryChangeType::Loss,
                    })
                    .chain(changes_for_this_node.iter().cloned()),
                ancestry_changes_to_process,
            );
            println!(
                "final changes for parent = {:?}",
                ancestry_changes_to_process.get(parent)
            );
        }

        // TODO: separate function to "remove node" from Graph
        graph.ancestry[queued_parent.node.as_index()].clear();
        graph.parents[queued_parent.node.as_index()].clear();
        assert!(graph.birth_time[queued_parent.node.as_index()].is_some());
        graph.birth_time[queued_parent.node.as_index()].take();
        graph.free_nodes.push(queued_parent.node.as_index());
    }

    fn process_queued_node(
        queued_parent: QueuedNode,
        hashed_nodes: &mut HashSet<Node>,
        parent_queue: &mut std::collections::BinaryHeap<QueuedNode>,
        ancestry_changes_to_process: &mut HashMap<Node, Vec<AncestryChange>>,
        graph: &mut Graph,
    ) {
        match ancestry_changes_to_process.get_mut(&queued_parent.node) {
            Some(ancestry_changes) => {
                ancestry_changes.sort_unstable_by_key(|ac| ac.left());
                let mut overlapper = AncestryOverlapper::new(
                    queued_parent.node,
                    &graph.ancestry[queued_parent.node.as_index()],
                    ancestry_changes,
                );
                // Clear parental ancestry
                graph.ancestry[queued_parent.node.as_index()].clear();
                while let Some(overlaps) = overlapper.output_ancestry() {
                    if let Some(ancestry_change) = overlaps.parental_ancestry_change {
                        println!("change for {queued_parent:?} is {ancestry_change:?}");
                        for parent in graph.parents(queued_parent.node) {
                            update_internal_stuff(*parent, hashed_nodes, parent_queue, graph);
                            push_ancestry_changes_to_parent(
                                *parent,
                                [ancestry_change].into_iter(),
                                ancestry_changes_to_process,
                            );
                        }
                    } else {
                        println!("no ancestry change detected for {queued_parent:?}");
                    }
                    // Output the new ancestry for the parent
                    if let Some(iter) = overlaps.iter_overlaps() {
                        for &a in iter {
                            println!("adding new ancestry {a:?} to {queued_parent:?}");
                            graph.ancestry[queued_parent.node.as_index()].push(a);
                        }
                    }
                }
            }
            None => panic!(),
        }
    }

    fn propagate_ancestry_changes(graph: &mut Graph) {
        use std::collections::BinaryHeap;

        let mut hashed_nodes: HashSet<Node> = HashSet::new();
        let mut parent_queue: BinaryHeap<QueuedNode> = BinaryHeap::new();
        let mut ancestry_changes_to_process: HashMap<Node, Vec<AncestryChange>> = HashMap::new();

        for tranmission in graph.transmissions.iter() {
            let change = AncestryChange {
                segment: Segment {
                    left: tranmission.left,
                    right: tranmission.right,
                },
                node: tranmission.child,
                change_type: AncestryChangeType::Unary,
            };
            for parent in graph.parents(tranmission.child) {
                if parent == &tranmission.parent {
                    println!(
                    "updating transmission from {:?} to {:?} on [{}, {}) to parent {parent:?}, and the actual parent is {:?}",
                    parent, tranmission.child, tranmission.left, tranmission.right, tranmission.parent
                );
                    update_internal_stuff(*parent, &mut hashed_nodes, &mut parent_queue, graph);
                    push_ancestry_changes_to_parent(
                        *parent,
                        [change].into_iter(),
                        &mut ancestry_changes_to_process,
                    );
                }
            }
        }

        for death in graph.deaths.iter() {
            update_internal_stuff(*death, &mut hashed_nodes, &mut parent_queue, graph)
        }

        for q in parent_queue.iter() {
            println!("{q:?} -> {:?}", ancestry_changes_to_process.get(&q.node));
        }

        while let Some(queued_parent) = parent_queue.pop() {
            println!(
                "processing {queued_parent:?} => {:?}, {:?}",
                graph.status[queued_parent.node.as_index()],
                ancestry_changes_to_process.get(&queued_parent.node)
            );
            assert!(hashed_nodes.contains(&queued_parent.node));
            match graph.status[queued_parent.node.as_index()] {
                NodeStatus::Death => process_node_death(
                    queued_parent,
                    &mut hashed_nodes,
                    &mut parent_queue,
                    &mut ancestry_changes_to_process,
                    graph,
                ),
                _ => process_queued_node(
                    queued_parent,
                    &mut hashed_nodes,
                    &mut parent_queue,
                    &mut ancestry_changes_to_process,
                    graph,
                ),
            }
        }
    }

    #[test]
    fn test_simple_overlap() {
        let mut graph = Graph::new(100).unwrap();
        let parent = graph.add_node(NodeStatus::Ancestor, 0);
        let child0 = graph.add_birth(1).unwrap();
        let child1 = graph.add_birth(1).unwrap();

        graph
            .record_transmission(0, graph.genome_length(), parent, child0)
            .unwrap();
        graph
            .record_transmission(0, graph.genome_length(), parent, child1)
            .unwrap();

        // NOTE: we need to add "dummy" ancestry to the parent
        // to have a valid data structure for testing.
        graph.ancestry[parent.as_index()].push(Ancestry {
            segment: Segment {
                left: 0,
                right: graph.genome_length,
            },
            ancestry: AncestryType::ToSelf,
        });

        for c in [child0, child1] {
            assert_eq!(graph.status[c.as_index()], NodeStatus::Birth);
            assert!(graph.parents(c).any(|&p| p == parent));
        }

        propagate_ancestry_changes(&mut graph);

        assert_eq!(graph.ancestry[parent.as_index()].len(), 2);
        for c in [child0, child1] {
            assert!(graph.ancestry[parent.as_index()].contains(&Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length
                },
                ancestry: AncestryType::Overlap(c)
            }));
            assert_eq!(graph.ancestry[c.as_index()].len(), 1);
        }
    }

    //  p0     p1   <- deaths
    // /  \    |
    // c0  c1  c2   <- births
    //
    //  Simplifies to
    //  p0
    // /  \
    // c0  c1  c2   <- births
    #[test]
    fn test_simple_case_with_two_parents() {
        let mut graph = Graph::new(100).unwrap();
        let parent0 = graph.add_node(NodeStatus::Ancestor, 0);
        let parent1 = graph.add_node(NodeStatus::Ancestor, 0);
        let child0 = graph.add_birth(1).unwrap();
        let child1 = graph.add_birth(1).unwrap();
        let child2 = graph.add_birth(1).unwrap();

        graph
            .record_transmission(0, graph.genome_length(), parent0, child0)
            .unwrap();
        graph
            .record_transmission(0, graph.genome_length(), parent0, child1)
            .unwrap();
        graph
            .record_transmission(0, graph.genome_length(), parent1, child2)
            .unwrap();

        // NOTE: we need to add "dummy" ancestry to the parent
        // to have a valid data structure for testing.
        for parent in [parent0, parent1] {
            graph.ancestry[parent.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::ToSelf,
            });
        }

        for c in [child0, child1] {
            assert_eq!(graph.status[c.as_index()], NodeStatus::Birth);
            assert!(graph.parents(c).any(|&p| p == parent0));
        }
        assert_eq!(graph.status[child2.as_index()], NodeStatus::Birth);
        assert!(graph.parents(child2).any(|&p| p == parent1));

        propagate_ancestry_changes(&mut graph);

        // Parent 1 should have no ancestry
        assert!(graph.ancestry[parent1.as_index()].is_empty());

        assert_eq!(graph.ancestry[parent0.as_index()].len(), 2);
        for c in [child0, child1] {
            assert!(graph.ancestry[parent0.as_index()].contains(&Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length
                },
                ancestry: AncestryType::Overlap(c)
            }));
            assert_eq!(graph.ancestry[c.as_index()].len(), 1);
        }
    }

    //        0      <- "Rando ancestor from before"
    //        |
    //     -------
    //     |     |
    //     |     1   <- Death
    //     2     |   <- Death
    //   -----   |
    //   |   |   |
    //   3   4   5   <- Birth
    // first test involving "death" of a node!!!
    #[test]
    fn test_simple_case_of_propagation_over_multiple_generations() {
        let mut graph = Graph::new(100).unwrap();
        let node0 = graph.add_node(NodeStatus::Ancestor, 0);
        let node1 = graph.add_node(NodeStatus::Death, 1);
        let node2 = graph.add_node(NodeStatus::Death, 2);
        let node3 = graph.add_birth(3).unwrap();
        let node4 = graph.add_birth(3).unwrap();
        let node5 = graph.add_birth(3).unwrap();

        for node in [node1, node2] {
            graph.parents[node.as_index()].insert(node0);
        }
        for child in [node3, node4] {
            graph
                .record_transmission(0, graph.genome_length(), node2, child)
                .unwrap();
        }
        graph
            .record_transmission(0, graph.genome_length(), node1, node5)
            .unwrap();

        // NOTE: we need to add "dummy" ancestry to the parents
        // to have a valid data structure for testing.
        for node in [node1, node2] {
            graph.ancestry[node0.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::Overlap(node),
            });
            graph.ancestry[node.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::ToSelf,
            });
        }

        propagate_ancestry_changes(&mut graph);

        // Okay, now we can test the output
        // These two nodes are dropped from the graph
        for extinct_node in [node1, node2] {
            assert!(graph.ancestry[extinct_node.as_index()].is_empty());
            assert!(graph.parents[extinct_node.as_index()].is_empty());
            assert!(graph.birth_time[extinct_node.as_index()].is_none());
            assert!(graph.free_nodes.contains(&extinct_node.as_index()));
        }

        // Node 0
        assert_eq!(graph.ancestry[node0.as_index()].len(), 3);
        for child in [node3, node4, node5] {
            assert!(
                graph.ancestry[node0.as_index()].contains(&Ancestry {
                    segment: Segment {
                        left: 0,
                        right: graph.genome_length
                    },
                    ancestry: AncestryType::Overlap(child)
                }),
                "failing child node = {child:?}"
            );
        }
    }

    //        0      <- "Rando ancestor from before"
    //        |
    //     -------
    //     |     |
    //     |     1   <- Death
    //     2         <- Death
    //   -----
    //   |   |
    //   3   4       <- Birth
    // first test involving "death" of a node!!!
    #[test]
    fn test_simple_case_of_propagation_over_multiple_generations_with_dangling_death() {
        let mut graph = Graph::new(100).unwrap();
        let node0 = graph.add_node(NodeStatus::Ancestor, 0);
        let node1 = graph.add_node(NodeStatus::Death, 1);
        let node2 = graph.add_node(NodeStatus::Death, 2);
        let node3 = graph.add_birth(3).unwrap();
        let node4 = graph.add_birth(3).unwrap();

        for node in [node1, node2] {
            graph.parents[node.as_index()].insert(node0);
            graph.deaths.push(node);
        }
        for child in [node3, node4] {
            graph
                .record_transmission(0, graph.genome_length(), node2, child)
                .unwrap();
        }

        // NOTE: we need to add "dummy" ancestry to the parents
        // to have a valid data structure for testing.
        for node in [node1, node2] {
            graph.ancestry[node0.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::Overlap(node),
            });
            graph.ancestry[node.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::ToSelf,
            });
        }

        propagate_ancestry_changes(&mut graph);

        // Okay, now we can test the output
        // These two nodes are dropped from the graph
        for extinct_node in [node1, node2] {
            assert!(graph.ancestry[extinct_node.as_index()].is_empty());
            assert!(graph.parents[extinct_node.as_index()].is_empty());
            assert!(graph.birth_time[extinct_node.as_index()].is_none());
            assert!(graph.free_nodes.contains(&extinct_node.as_index()));
        }

        // Node 0
        assert_eq!(graph.ancestry[node0.as_index()].len(), 2);
        for child in [node3, node4] {
            assert!(
                graph.ancestry[node0.as_index()].contains(&Ancestry {
                    segment: Segment {
                        left: 0,
                        right: graph.genome_length
                    },
                    ancestry: AncestryType::Overlap(child)
                }),
                "failing child node = {child:?}"
            );
        }
    }

    // Tree 1:
    //        0      <- "Rando ancestor from before"
    //        |
    //     -------
    //     |     |
    //     |     1   <- Death
    //     2     |   <- Death
    //           |
    //         -----
    //         |   |
    //         3   4  <- Birth
    //
    // Tree 2:
    //        0      <- "Rando ancestor from before"
    //        |
    //     -------
    //     |     |
    //     |     1   <- Death
    //     2         <- Death
    //     |
    //   -----
    //   |   |
    //   3   4       <- Birth
    #[test]
    fn test_simple_case_of_propagation_over_multiple_generations_two_trees() {
        let mut graph = Graph::new(100).unwrap();
        let crossover_pos = graph.genome_length() / 2;
        let node0 = graph.add_node(NodeStatus::Ancestor, 0);
        let node1 = graph.add_node(NodeStatus::Death, 1);
        let node2 = graph.add_node(NodeStatus::Death, 2);
        let node3 = graph.add_birth(3).unwrap();
        let node4 = graph.add_birth(3).unwrap();

        for node in [node1, node2] {
            graph.parents[node.as_index()].insert(node0);
            //record the deaths!!
            graph.deaths.push(node);
        }

        graph
            .record_transmission(0, crossover_pos, node1, node3)
            .unwrap();
        graph
            .record_transmission(0, crossover_pos, node1, node4)
            .unwrap();
        graph
            .record_transmission(crossover_pos, graph.genome_length(), node2, node4)
            .unwrap();
        graph
            .record_transmission(crossover_pos, graph.genome_length(), node2, node3)
            .unwrap();

        println!("tranmissions = {:?}", graph.transmissions);

        // NOTE: we need to add "dummy" ancestry to the parents
        // to have a valid data structure for testing.
        for node in [node1, node2] {
            graph.ancestry[node0.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::Overlap(node),
            });
            graph.ancestry[node.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::ToSelf,
            });
        }

        propagate_ancestry_changes(&mut graph);

        // Okay, now we can test the output
        // These two nodes are dropped from the graph
        for extinct_node in [node1, node2] {
            assert!(graph.ancestry[extinct_node.as_index()].is_empty());
            assert!(graph.parents[extinct_node.as_index()].is_empty());
            assert!(graph.birth_time[extinct_node.as_index()].is_none());
            assert!(graph.free_nodes.contains(&extinct_node.as_index()));
        }

        // Node 0
        assert_eq!(graph.ancestry[node0.as_index()].len(), 4);
        for child in [node3, node4] {
            assert!(
                graph.ancestry[node0.as_index()].contains(&Ancestry {
                    segment: Segment {
                        left: 0,
                        right: crossover_pos,
                    },
                    ancestry: AncestryType::Overlap(child)
                }),
                "failing child node = {child:?}"
            );
            assert!(
                graph.ancestry[node0.as_index()].contains(&Ancestry {
                    segment: Segment {
                        left: crossover_pos,
                        right: graph.genome_length
                    },
                    ancestry: AncestryType::Overlap(child)
                }),
                "failing child node = {child:?}"
            );
        }
    }

    // Tree 1:
    //        0                 <- "Rando ancestor from before"
    //        |
    //     -------
    //     |     |
    //     |     1   2          <- Death, Death
    //     3     |              <- Death
    //     |     |
    //     4     5              <- Birth
    //
    // Tree 2:
    //        0                 <- "Rando ancestor from before"
    //        |
    //     -------
    //     |     |
    //     |     1   2          <- Death, Death
    //     3         |          <- Death
    //     |         |
    //     5         4          <- Birth
    #[test]
    fn test_second_case_of_propagation_over_multiple_generations_two_trees() {
        let mut graph = Graph::new(100).unwrap();
        let crossover_pos = graph.genome_length() / 2;
        let node0 = graph.add_node(NodeStatus::Ancestor, 0);
        let node1 = graph.add_node(NodeStatus::Death, 1);
        let node2 = graph.add_node(NodeStatus::Death, 1);
        let node3 = graph.add_node(NodeStatus::Death, 2);
        let node4 = graph.add_birth(3).unwrap();
        let node5 = graph.add_birth(3).unwrap();

        for node in [node1, node3] {
            graph.parents[node.as_index()].insert(node0);
            //record the deaths!!
            graph.deaths.push(node);
        }
        //record the deaths!!
        graph.deaths.push(node2);

        // NOTE: we need to add "dummy" ancestry to the parents
        // to have a valid data structure for testing.
        for node in [node1, node3] {
            graph.ancestry[node0.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::Overlap(node),
            });
            graph.ancestry[node.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::ToSelf,
            });
        }
        graph.ancestry[node2.as_index()].push(Ancestry {
            segment: Segment {
                left: 0,
                right: graph.genome_length,
            },
            ancestry: AncestryType::ToSelf,
        });

        graph
            .record_transmission(0, crossover_pos, node3, node4)
            .unwrap();
        graph
            .record_transmission(0, crossover_pos, node1, node5)
            .unwrap();
        graph
            .record_transmission(crossover_pos, graph.genome_length(), node3, node5)
            .unwrap();
        graph
            .record_transmission(crossover_pos, graph.genome_length(), node2, node3)
            .unwrap();

        assert_eq!(graph.deaths.len(), 3);
        assert_eq!(graph.births.len(), 2);

        for extinct_node in [node1, node2, node3] {
            assert!(!graph.ancestry[extinct_node.as_index()].is_empty());
            assert!(graph.birth_time[extinct_node.as_index()].is_some());
        }
        propagate_ancestry_changes(&mut graph);
        for extinct_node in [node1, node2, node3] {
            assert!(graph.ancestry[extinct_node.as_index()].is_empty());
            assert!(graph.parents[extinct_node.as_index()].is_empty());
            assert!(graph.birth_time[extinct_node.as_index()].is_none());
            assert!(graph.free_nodes.contains(&extinct_node.as_index()));
        }

        assert_eq!(graph.ancestry[node0.as_index()].len(), 2);
        for child in [node4, node5] {
            assert!(
                graph.ancestry[node0.as_index()].contains(&Ancestry {
                    segment: Segment {
                        left: 0,
                        right: crossover_pos,
                    },
                    ancestry: AncestryType::Overlap(child)
                }),
                "failing child node = {child:?}"
            );
            assert_eq!(graph.ancestry[child.as_index()].len(), 1);
            assert!(
                graph.ancestry[child.as_index()].contains(&Ancestry {
                    segment: Segment {
                        left: 0,
                        right: graph.genome_length(),
                    },
                    ancestry: AncestryType::ToSelf
                }),
                "failing child node = {child:?}"
            );
        }
    }
}
