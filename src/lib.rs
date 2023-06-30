use nohash::BuildNoHashHasher;
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::BuildHasherDefault;

// NOTE: for design purposes -- delete later.
mod overlapper_experiments;

mod flags;

use flags::PropagationOptions;

#[repr(transparent)]
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub struct Node(usize);

impl Node {
    fn as_index(&self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    Ancestor,
    Birth,
    Death,
    Alive,
    Sample,
    Extinct,
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
pub struct Graph {
    status: Vec<NodeStatus>,
    birth_time: Vec<Option<i64>>,
    parents: Vec<NodeHash>,
    children: Vec<NodeHash>,
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
        let children = Vec::with_capacity(capacity);
        let status = Vec::with_capacity(capacity);
        let birth_time = Vec::with_capacity(capacity);
        let transmissions = Vec::with_capacity(capacity);
        let ancestry = Vec::with_capacity(capacity);
        let births = vec![];
        let deaths = vec![];
        let free_nodes = Vec::new();
        Some(Self {
            status,
            birth_time,
            parents,
            children,
            transmissions,
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
        let birth_time = vec![Some(0); num_nodes];
        let parents = vec![NodeHash::with_hasher(BuildHasherDefault::default()); num_nodes];
        let children = vec![NodeHash::with_hasher(BuildHasherDefault::default()); num_nodes];
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
            birth_time,
            parents,
            children,
            transmissions,
            ancestry,
            births,
            deaths,
            free_nodes,
            genome_length,
        })
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
            Some(index) => {
                assert!(self.children[index].is_empty());
                assert!(self.parents[index].is_empty());
                assert!(self.ancestry[index].is_empty());
                assert!(matches!(self.status[index], NodeStatus::Extinct));
                self.birth_time[index] = Some(birth_time);
                self.status[index] = status;
                if matches!(status, NodeStatus::Birth) {
                    self.ancestry[index].push(Ancestry::birth(self.genome_length).unwrap());
                }
                Node(index)
            }
            None => {
                self.birth_time.push(Some(birth_time));
                self.status.push(status);
                self.parents
                    .push(NodeHash::with_hasher(BuildHasherDefault::default()));
                self.children
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
                    != birth_time
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
        let _ = self.children[parent.as_index()].insert(child);

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
    println!("parental_node_ancestry = {parental_node_ancestry:?}");
    println!("ancestry_changes = {ancestry_changes:?}");
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
    println!("queue = {queue:?}");

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
    parent_status: NodeStatus,
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
        parent_status: NodeStatus,
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
            parent_status,
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

    fn make_overlaps(&mut self, options: PropagationOptions) -> Overlaps {
        let ancestry_change = output_overlaps(
            options,
            Segment::new(self.left, self.right).unwrap(),
            self.parent,
            self.parent_status,
            &self.parental_overlaps,
            &self.change_overlaps,
            &mut self.overlaps,
            &mut self.output_nodes,
        );
        println!("self.overlaps = {:?} | {ancestry_change:?}", self.overlaps);
        Overlaps::new(
            self.left,
            self.right,
            self.parent,
            &self.overlaps,
            ancestry_change,
        )
    }

    // TODO: perhaps we could reurn an enum from here that
    // handles things like internal sample ancestry gaps?
    fn calculate_next_overlap_set(&mut self, options: PropagationOptions) -> Option<Overlaps> {
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
            Some(self.make_overlaps(options))
        } else {
            if self.parental_overlaps.len() + self.change_overlaps.len() != 0 {
                self.left = self.right;
                self.filter_overlaps();
            }
            if self.parental_overlaps.len() + self.change_overlaps.len() != 0 {
                self.update_right_from_overlaps();
                Some(self.make_overlaps(options))
            } else {
                None
            }
        }
    }

    fn output_ancestry(&mut self, options: PropagationOptions) -> Option<Overlaps<'_>> {
        self.calculate_next_overlap_set(options)
    }
}

fn calculate_ancestry_change(
    options: PropagationOptions,
    segment: Segment,
    parent: Node,
    parent_status: NodeStatus,
    parental_overlaps: &[Ancestry],
    overlaps: &[Ancestry],
) -> Option<AncestryChange> {
    let (parental_node, parental_ancestry) = match parental_overlaps.first() {
        Some(ancestry) => (parent, ancestry.ancestry),
        None => panic!(),
    };

    println!("the overlaps are {overlaps:?}");
    println!("the ancestry is {parental_ancestry:?}");
    println!(
        "the options are {options:?}, {}",
        options.keep_unary_nodes()
    );
    let change_type = match parental_ancestry {
        AncestryType::Overlap(_) => {
            assert!(parental_overlaps.len() > 1);
            match overlaps.len() {
                0 => Some(AncestryChangeType::Loss),
                1 => {
                    if options.keep_unary_nodes() || matches!(parent_status, NodeStatus::Sample) {
                        Some(AncestryChangeType::Unary)
                    } else {
                        Some(AncestryChangeType::Loss)
                    }
                }
                _ => None,
            }
        }
        AncestryType::Unary(_) | AncestryType::ToSelf => {
            assert!(parental_overlaps.len() == 1);
            match overlaps.len() {
                0 => {
                    if matches!(parent_status, NodeStatus::Sample) {
                        Some(AncestryChangeType::Unary)
                    } else {
                        Some(AncestryChangeType::Loss)
                    }
                }
                1 => {
                    if options.keep_unary_nodes() || matches!(parent_status, NodeStatus::Sample) {
                        // NOTE: we may need to revisit this case
                        Some(AncestryChangeType::Unary)
                    } else {
                        Some(AncestryChangeType::Loss)
                    }
                }
                _ => Some(AncestryChangeType::Overlap),
            }
        }
    };
    println!("the ancestry change is {change_type:?}");

    change_type.map(|change_type| AncestryChange {
        segment,
        node: parental_node,
        change_type,
    })
}

fn output_overlaps(
    options: PropagationOptions,
    segment: Segment,
    parent: Node,
    parent_status: NodeStatus,
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
                segment,
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
                        segment,
                        ancestry: AncestryType::Overlap(node),
                    });
                }
            }
        }
    }
    calculate_ancestry_change(
        options,
        segment,
        parent,
        parent_status,
        parental_overlaps,
        output_ancestry,
    )
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

    fn overlaps(&self) -> &[Ancestry] {
        self.overlaps
    }
}

//BOILER PLATE ALERT
#[derive(Debug, Copy, Clone)]
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
    hashed_nodes: &mut NodeHash,
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

fn node_is_extinct(node: Node, graph: &Graph) -> bool {
    let index = node.as_index();
    graph.birth_time[index].is_none()
        && graph.ancestry[index].is_empty()
        && graph.children[index].is_empty()
        && graph.parents[index].is_empty()
}

struct ReachableNodes<'graph> {
    graph: &'graph Graph,
    queued_nodes: NodeHash,
    node_queue: std::collections::BinaryHeap<QueuedNode>,
}

impl<'graph> ReachableNodes<'graph> {
    fn new(graph: &'graph Graph) -> Self {
        let mut node_queue: std::collections::BinaryHeap<QueuedNode> =
            std::collections::BinaryHeap::new();
        let mut queued_nodes: NodeHash = HashSet::with_hasher(BuildHasherDefault::default());
        for node in &graph.births {
            if !queued_nodes.contains(node) {
                node_queue.push(QueuedNode {
                    node: *node,
                    birth_time: graph.birth_time[node.as_index()].unwrap(),
                });
                queued_nodes.insert(*node);
            } else {
                panic!()
            }
        }
        Self {
            graph,
            queued_nodes,
            node_queue,
        }
    }
}

impl<'graph> Iterator for ReachableNodes<'graph> {
    type Item = Node;
    fn next(&mut self) -> Option<Self::Item> {
        match self.node_queue.pop() {
            Some(node) => {
                println!("popped {node:?}");
                assert!(self.queued_nodes.remove(&node.node));
                for parent in &self.graph.parents[node.node.as_index()] {
                    if !self.queued_nodes.contains(parent) {
                        self.queued_nodes.insert(*parent);
                        self.node_queue.push(QueuedNode {
                            node: *parent,
                            birth_time: self.graph.birth_time[parent.as_index()].unwrap(),
                        });
                        println!("pushing {parent:?}");
                    }
                }
                Some(node.node)
            }
            None => None,
        }
    }
}

fn reachable_nodes(graph: &Graph) -> impl Iterator<Item = Node> + '_ {
    ReachableNodes::new(graph)
}

// NOTE: current tests are only able to trigger this fn
// in the case of "dangling deaths" isolated to a given tree.
fn process_node_death(
    queued_parent: QueuedNode,
    hashed_nodes: &mut NodeHash,
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
    graph.ancestry[queued_parent.node.as_index()].clear();
}

// NOTE: if we require that nodes are marked as SAMPLES
// when born, and prior to propagation, then we should (?)
// be able to eliminate the logic needed for filling in
// ancestry gaps for sample nodes??
fn process_queued_node(
    options: PropagationOptions,
    queued_parent: QueuedNode,
    parent_status: NodeStatus,
    hashed_nodes: &mut NodeHash,
    parent_queue: &mut std::collections::BinaryHeap<QueuedNode>,
    ancestry_changes_to_process: &mut HashMap<Node, Vec<AncestryChange>>,
    graph: &mut Graph,
) {
    assert!(!matches!(
        graph.status[queued_parent.node.as_index()],
        NodeStatus::Birth
    ));
    match ancestry_changes_to_process.get_mut(&queued_parent.node) {
        Some(ancestry_changes) => {
            ancestry_changes.sort_unstable_by_key(|ac| ac.left());
            let mut overlapper = AncestryOverlapper::new(
                queued_parent.node,
                graph.status[queued_parent.node.as_index()],
                &graph.ancestry[queued_parent.node.as_index()],
                ancestry_changes,
            );
            // Clear parental ancestry
            graph.ancestry[queued_parent.node.as_index()].clear();
            let mut previous_right: i64 = 0;
            while let Some(overlaps) = overlapper.output_ancestry(options) {
                println!(
                    "COORDS: {previous_right} -> ({}, {}), {}",
                    overlaps.left,
                    overlaps.right,
                    matches!(parent_status, NodeStatus::Sample)
                );

                // There is some ugliness here: sample nodes are getting
                // ancestry changes marked as not None, which is befuddling
                // all of the logic below.
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
                match overlaps.parental_ancestry_change {
                    Some(AncestryChange {
                        change_type: AncestryChangeType::Loss,
                        ..
                    }) => {
                        println!(
                            "parents of lost node = {:?}",
                            graph.parents[queued_parent.node.as_index()]
                        );
                        for parent in graph.parents(queued_parent.node) {
                            push_ancestry_changes_to_parent(
                                *parent,
                                overlaps.overlaps().iter().map(|a| {
                                    let node = match a.ancestry {
                                        AncestryType::Unary(n) | AncestryType::Overlap(n) => n,
                                        AncestryType::ToSelf => panic!(),
                                    };
                                    AncestryChange {
                                        segment: a.segment,
                                        node,
                                        change_type: AncestryChangeType::Unary,
                                    }
                                }),
                                ancestry_changes_to_process,
                            )
                        }
                    }
                    _ => {
                        if matches!(parent_status, NodeStatus::Sample)
                            && overlaps.overlaps.is_empty()
                        {
                            println!("EMPTY on {} {}", overlaps.left, overlaps.right);
                            graph.ancestry[queued_parent.node.as_index()].push(Ancestry {
                                segment: Segment {
                                    left: overlaps.left,
                                    right: overlaps.right,
                                },
                                ancestry: AncestryType::ToSelf,
                            });
                        } else {
                            for &a in overlaps.overlaps() {
                                println!("adding new ancestry {a:?} to {queued_parent:?}");
                                graph.ancestry[queued_parent.node.as_index()].push(a);
                                match a.ancestry {
                                    AncestryType::ToSelf => panic!(),
                                    AncestryType::Unary(node) | AncestryType::Overlap(node) => {
                                        graph.parents[node.as_index()].insert(queued_parent.node);
                                        graph.children[queued_parent.node.as_index()].insert(node);
                                    }
                                }
                            }
                        }
                    }
                }
                previous_right = overlaps.right;
                println!("last right = {previous_right}");
            }
            if matches!(parent_status, NodeStatus::Sample)
                && previous_right != graph.genome_length()
            {
                graph.ancestry[queued_parent.node.as_index()].push(Ancestry {
                    segment: Segment::new(previous_right, graph.genome_length).unwrap(),
                    ancestry: AncestryType::ToSelf,
                });
            }
            match graph.status[queued_parent.node.as_index()] {
                NodeStatus::Death => {
                    if !graph.ancestry[queued_parent.node.as_index()].is_empty() {
                        graph.status[queued_parent.node.as_index()] = NodeStatus::Ancestor;
                    }
                }
                NodeStatus::Extinct => panic!(),
                _ => (),
            }
        }
        None => match graph.status[queued_parent.node.as_index()] {
            NodeStatus::Death => {
                process_node_death(
                    queued_parent,
                    hashed_nodes,
                    parent_queue,
                    ancestry_changes_to_process,
                    graph,
                );
            }
            _ => panic!(),
        },
    }

    // TODO: DUPLICATION (not really, but it is too related to code happening in
    // the processing of death nodes)
    println!(
        "final ancestry len = {}",
        graph.ancestry[queued_parent.node.as_index()].len()
    );
    if graph.ancestry[queued_parent.node.as_index()].is_empty() {
        for child in &graph.children[queued_parent.node.as_index()] {
            graph.parents[child.as_index()].remove(&queued_parent.node);
        }
        for parent in &graph.parents[queued_parent.node.as_index()] {
            graph.children[parent.as_index()].remove(&queued_parent.node);
        }
        graph.parents[queued_parent.node.as_index()].clear();
        graph.children[queued_parent.node.as_index()].clear();
        assert!(graph.birth_time[queued_parent.node.as_index()].is_some());
        graph.birth_time[queued_parent.node.as_index()].take();
        graph.free_nodes.push(queued_parent.node.as_index());
        graph.status[queued_parent.node.as_index()] = NodeStatus::Extinct;
    }
}

fn propagate_ancestry_changes(options: PropagationOptions, graph: &mut Graph) {
    println!(
        "the input options are {options:?}, {}",
        options.keep_unary_nodes()
    );
    let mut hashed_nodes: NodeHash = NodeHash::with_hasher(BuildHasherDefault::default());
    let mut parent_queue: std::collections::BinaryHeap<QueuedNode> =
        std::collections::BinaryHeap::new();
    let mut ancestry_changes_to_process: HashMap<Node, Vec<AncestryChange>> = HashMap::new();

    let mut unique_child_visits: NodeHash = HashSet::with_hasher(BuildHasherDefault::default());
    for tranmission in graph.transmissions.iter() {
        assert!(matches!(
            graph.status[tranmission.child.as_index()],
            NodeStatus::Birth
        ));
        let change = AncestryChange {
            segment: Segment {
                left: tranmission.left,
                right: tranmission.right,
            },
            node: tranmission.child,
            change_type: AncestryChangeType::Unary,
        };
        for parent in graph.parents(tranmission.child) {
            unique_child_visits.insert(tranmission.child);
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
    assert_eq!(unique_child_visits.len(), graph.births.len());
    for birth in unique_child_visits.into_iter() {
        graph.status[birth.as_index()] = NodeStatus::Alive;
    }

    for death in graph.deaths.iter() {
        update_internal_stuff(*death, &mut hashed_nodes, &mut parent_queue, graph)
    }

    // for q in parent_queue.iter() {
    //     println!("{q:?} -> {:?}", ancestry_changes_to_process.get(&q.node));
    // }

    while let Some(queued_parent) = parent_queue.pop() {
        println!(
            "processing {queued_parent:?} => {:?}, {:?}",
            graph.status[queued_parent.node.as_index()],
            ancestry_changes_to_process.get(&queued_parent.node)
        );
        assert!(hashed_nodes.contains(&queued_parent.node));
        match graph.status[queued_parent.node.as_index()] {
            //NodeStatus::Death => process_node_death(
            //    queued_parent,
            //    &mut hashed_nodes,
            //    &mut parent_queue,
            //    &mut ancestry_changes_to_process,
            //    graph,
            //),
            _ => process_queued_node(
                options,
                queued_parent,
                graph.status[queued_parent.node.as_index()],
                &mut hashed_nodes,
                &mut parent_queue,
                &mut ancestry_changes_to_process,
                graph,
            ),
        }
        hashed_nodes.remove(&queued_parent.node);
        ancestry_changes_to_process.remove(&queued_parent.node);
    }

    // TODO: Seems silly to have a whole separate status for this?
    graph
        .births
        .iter()
        .for_each(|b| graph.status[b.as_index()] = NodeStatus::Alive);
    assert!(parent_queue.is_empty());
    assert!(hashed_nodes.is_empty());
    assert!(ancestry_changes_to_process.is_empty());
}

// We could do all this with rstest,
// but we still need to pattern match,
// etc., to use the fixtures.
// IMO, this doesn't seem like rstest
// saves much boiler plate here...
#[cfg(test)]
mod graph_fixtures {
    use super::*;

    //        0      <- "Rando ancestor from before"
    //        |
    //     -------
    //     |     |
    //     |     1   <- Death
    //     2     |   <- Death
    //   -----   |
    //   |   |   |
    //   3   4   5   <- Birth
    //
    // After propagation with default options, the topology is:
    //        0
    //        |
    //     -------
    //     |     |
    //     |     |
    //     2     |
    //   -----   |
    //   |   |   |
    //   3   4   5
    pub struct Topology0 {
        pub node0: Node,
        pub node1: Node,
        pub node2: Node,
        pub node3: Node,
        pub node4: Node,
        pub node5: Node,
        pub graph: Graph,
    }

    impl Topology0 {
        pub fn new() -> Self {
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
            Self {
                node0,
                node1,
                node2,
                node3,
                node4,
                node5,
                graph,
            }
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
    //
    //   After propagation with default options:
    //
    //     2         <- Death
    //   -----
    //   |   |
    //   3   4       <- Birth
    pub struct Topology1 {
        pub node0: Node,
        pub node1: Node,
        pub node2: Node,
        pub node3: Node,
        pub node4: Node,
        pub graph: Graph,
    }

    impl Topology1 {
        pub fn new() -> Self {
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
            Self {
                node0,
                node1,
                node2,
                node3,
                node4,
                graph,
            }
        }
    }

    //                     0
    //                   -----
    //                   |   |
    //                   1   2
    //                 ----  ---
    //                 |  |  | |
    //                 3  4  5 6
    //
    // # NOTES
    //
    // * deaths vector is not filled
    // * All nodes intially marked as Ancestor
    pub struct Topology2 {
        pub node0: Node,
        pub node1: Node,
        pub node2: Node,
        pub node3: Node,
        pub node4: Node,
        pub node5: Node,
        pub node6: Node,
        pub graph: Graph,
    }

    impl Topology2 {
        pub fn new() -> Self {
            let mut graph = Graph::new(100).unwrap();
            let node0 = graph.add_node(NodeStatus::Ancestor, 0);
            let node1 = graph.add_node(NodeStatus::Ancestor, 1);
            let node2 = graph.add_node(NodeStatus::Ancestor, 1);
            let node3 = graph.add_node(NodeStatus::Ancestor, 2);
            let node4 = graph.add_node(NodeStatus::Ancestor, 2);
            let node5 = graph.add_node(NodeStatus::Ancestor, 2);
            let node6 = graph.add_node(NodeStatus::Ancestor, 2);

            for node in [node1, node2] {
                graph.children[node0.as_index()].insert(node);
                graph.parents[node.as_index()].insert(node0);
                graph.ancestry[node0.as_index()].push(Ancestry {
                    segment: Segment {
                        left: 0,
                        right: graph.genome_length,
                    },
                    ancestry: AncestryType::Overlap(node),
                });
            }
            for node in [node3, node4] {
                graph.children[node1.as_index()].insert(node);
                graph.parents[node.as_index()].insert(node1);
                graph.ancestry[node1.as_index()].push(Ancestry {
                    segment: Segment {
                        left: 0,
                        right: graph.genome_length,
                    },
                    ancestry: AncestryType::Overlap(node),
                });
            }
            for node in [node5, node6] {
                graph.children[node2.as_index()].insert(node);
                graph.parents[node.as_index()].insert(node2);
                graph.ancestry[node2.as_index()].push(Ancestry {
                    segment: Segment {
                        left: 0,
                        right: graph.genome_length,
                    },
                    ancestry: AncestryType::Overlap(node),
                });
            }

            for node in [node3, node4, node5, node6] {
                graph.ancestry[node.as_index()].push(Ancestry {
                    segment: Segment {
                        left: 0,
                        right: graph.genome_length,
                    },
                    ancestry: AncestryType::ToSelf,
                });
            }
            Self {
                node0,
                node1,
                node2,
                node3,
                node4,
                node5,
                node6,
                graph,
            }
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
fn test_births_out_of_order() {
    let mut graph = Graph::new(100).unwrap();
    let _ = graph.add_node(NodeStatus::Ancestor, 0);
    let _ = graph.add_birth(2).unwrap();
    assert!(graph.add_birth(1).is_err());
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
mod test_standard_case {
    use super::*;

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

        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);

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

        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);

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
    //
    // After propagation, the topology is:
    //        0
    //        |
    //     -------
    //     |     |
    //     |     |
    //     2     |
    //   -----   |
    //   |   |   |
    //   3   4   5
    #[test]
    fn test_simple_case_of_propagation_over_multiple_generations() {
        let graph_fixtures::Topology0 {
            node0,
            node1,
            node2,
            node3,
            node4,
            node5,
            mut graph,
        } = graph_fixtures::Topology0::new();

        for node in [node3, node4, node5] {
            assert!(matches!(graph.status[node.as_index()], NodeStatus::Birth));
        }

        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);

        assert!(graph.ancestry[node1.as_index()].is_empty());
        assert!(graph.parents[node1.as_index()].is_empty());
        assert!(graph.birth_time[node1.as_index()].is_none());
        assert!(graph.free_nodes.contains(&node1.as_index()));

        assert_eq!(graph.ancestry[node2.as_index()].len(), 2);
        for child in [node3, node4] {
            assert!(
                graph.ancestry[node2.as_index()].contains(&Ancestry {
                    segment: Segment {
                        left: 0,
                        right: graph.genome_length
                    },
                    ancestry: AncestryType::Overlap(child)
                }),
                "failing child node = {child:?}"
            );
        }

        // Node 0
        assert_eq!(graph.ancestry[node0.as_index()].len(), 2);
        for child in [node2, node5] {
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

        // Verify status changes
        for node in [node3, node4, node5] {
            assert!(matches!(graph.status[node.as_index()], NodeStatus::Alive));
        }
        for node in [node0,node2] {
            assert!(matches!(graph.status[node.as_index()], NodeStatus::Ancestor));
        }
        assert!(matches!(
            graph.status[node1.as_index()],
            NodeStatus::Extinct
        ));
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
    //
    //   After propagation:
    //
    //     2         <- Death
    //   -----
    //   |   |
    //   3   4       <- Birth
    #[test]
    fn test_simple_case_of_propagation_over_multiple_generations_with_dangling_death() {
        let graph_fixtures::Topology1 {
            node0,
            node1,
            node2,
            node3,
            node4,
            mut graph,
        } = graph_fixtures::Topology1::new();

        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);

        // Okay, now we can test the output
        // These two nodes are dropped from the graph
        for extinct_node in [node0, node1] {
            assert!(
                node_is_extinct(extinct_node, &graph),
                "failing node = {extinct_node:?}"
            )
        }

        // Node 0
        assert_eq!(graph.ancestry[node0.as_index()].len(), 0);
        for child in [node3, node4] {
            assert!(
                graph.ancestry[node2.as_index()].contains(&Ancestry {
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
    //
    //  After propagation the topologies are:
    //
    //           1   <- Death
    //           |   <- Death
    //           |
    //         -----
    //         |   |
    //         3   4  <- Birth
    //
    // Tree 2:
    //
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

        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);

        assert!(node_is_extinct(node0, &graph));

        // Node 0
        assert_eq!(graph.ancestry[node0.as_index()].len(), 0);
        for child in [node3, node4] {
            assert_eq!(graph.parents[child.as_index()].len(), 2);
            assert!(
                graph.ancestry[node1.as_index()].contains(&Ancestry {
                    segment: Segment {
                        left: 0,
                        right: crossover_pos,
                    },
                    ancestry: AncestryType::Overlap(child)
                }),
                "failing child node = {child:?}"
            );
        }
        for child in [node3, node4] {
            assert_eq!(graph.parents[child.as_index()].len(), 2);
            assert!(
                graph.ancestry[node2.as_index()].contains(&Ancestry {
                    segment: Segment {
                        left: crossover_pos,
                        right: graph.genome_length,
                    },
                    ancestry: AncestryType::Overlap(child)
                }),
                "failing child node = {child:?}"
            );
        }
        for node in [node1, node2] {
            assert_eq!(graph.children[node.as_index()].len(), 2);
            for child in [node3, node4] {
                assert!(graph.children[node.as_index()].contains(&child));
            }
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
    //
    // After propagation:
    //
    //        0                 <- "Rando ancestor from before"
    //        |
    //     -------
    //     |     |
    //     |     |
    //     |     |
    //     |     |
    //     4     5              <- Birth
    //
    // Tree 2:
    //
    // (Unary all the way down)
    //
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
            .record_transmission(crossover_pos, graph.genome_length(), node2, node4)
            .unwrap();

        assert_eq!(graph.deaths.len(), 3);
        assert_eq!(graph.births.len(), 2);

        for extinct_node in [node1, node2, node3] {
            assert!(!graph.ancestry[extinct_node.as_index()].is_empty());
            assert!(graph.birth_time[extinct_node.as_index()].is_some());
        }
        println!("{:?}", graph.parents);
        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
        for extinct_node in [node1, node2, node3] {
            assert!(
                node_is_extinct(extinct_node, &graph),
                "failing node is {extinct_node:?}"
            );
        }

        assert_eq!(graph.ancestry[node0.as_index()].len(), 2);
        for child in [node4, node5] {
            assert!(graph.children[node0.as_index()].contains(&child));
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
            assert_eq!(graph.parents[child.as_index()].len(), 1);
            assert!(
                graph.parents[child.as_index()].contains(&node0),
                "{:?}",
                graph.parents[child.as_index()]
            );
        }
        for node in [node0, node4, node5] {
            assert!(reachable_nodes(&graph).any(|n| n == node));
        }
        for node in [node1, node2, node3] {
            assert!(!reachable_nodes(&graph).any(|n| n == node));
        }
    }

    //                 0
    //                 |
    //            -----------
    //            |         |
    //            |         1
    //            |       -----
    //            |       |   |
    //            |       2   |
    //            |     ----  |
    //            |     |  |  |
    //            3     4  5  6
    //
    //          If we kill node 4, node 2 becomes unary
    //          Node 1 remains overlap and we never visit node 0
    //
    // NOTE:
    //
    // The bug is that node1 is getting "ancestry loss on 2" when
    // it should ALSO get "to unary on 5"
    #[test]
    fn test_overlap_propagation() {
        let mut graph = Graph::new(100).unwrap();
        let node0 = graph.add_node(NodeStatus::Ancestor, 0);
        let node1 = graph.add_node(NodeStatus::Ancestor, 1);
        let node2 = graph.add_node(NodeStatus::Ancestor, 2);
        let node3 = graph.add_node(NodeStatus::Ancestor, 3);
        let node4 = graph.add_node(NodeStatus::Death, 3);
        let node5 = graph.add_node(NodeStatus::Ancestor, 3);
        let node6 = graph.add_node(NodeStatus::Ancestor, 3);

        for node in [node1, node3] {
            graph.children[node0.as_index()].insert(node);
            graph.parents[node.as_index()].insert(node0);
            graph.ancestry[node0.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::Overlap(node),
            });
        }
        for node in [node2, node6] {
            graph.children[node1.as_index()].insert(node);
            graph.parents[node.as_index()].insert(node1);
            graph.ancestry[node1.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::Overlap(node),
            });
        }
        for node in [node4, node5] {
            graph.children[node2.as_index()].insert(node);
            graph.parents[node.as_index()].insert(node2);
            graph.ancestry[node2.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::Overlap(node),
            });
        }
        for node in [node3, node4, node5, node6] {
            graph.ancestry[node.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::ToSelf,
            });
        }

        graph.deaths.push(node4);
        println!("{graph:?}");
        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
        println!("{graph:?}");

        assert_eq!(graph.ancestry[node1.as_index()].len(), 2);
        assert!(graph.ancestry[node2.as_index()].is_empty());
        assert!(graph.birth_time[node2.as_index()].is_none());
        assert!(graph.parents[node2.as_index()].is_empty());
        assert!(graph.children[node2.as_index()].is_empty());
    }

    //                     0
    //                   -----
    //                   |   |
    //                   1   |
    //                 ----  |
    //                 |  |  |
    //                 2  3  4
    //
    //          If we kill node 2, node 1 becomes unary
    //          Node 0 remains overlap to 3, 4.
    //
    // NOTE:
    //
    // The bug is that node1 is getting "ancestry loss on 1" when
    // it should ALSO get "to unary on 3"
    #[test]
    fn test_subtree_propagation() {
        let mut graph = Graph::new(100).unwrap();
        let node0 = graph.add_node(NodeStatus::Ancestor, 0);
        let node1 = graph.add_node(NodeStatus::Ancestor, 1);
        let node2 = graph.add_node(NodeStatus::Death, 2);
        let node3 = graph.add_node(NodeStatus::Ancestor, 2);
        let node4 = graph.add_node(NodeStatus::Ancestor, 2);

        for node in [node1, node4] {
            graph.children[node0.as_index()].insert(node);
            graph.parents[node.as_index()].insert(node0);
            graph.ancestry[node0.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::Overlap(node),
            });
        }
        for node in [node2, node3] {
            graph.children[node1.as_index()].insert(node);
            graph.parents[node.as_index()].insert(node1);
            graph.ancestry[node1.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::Overlap(node),
            });
        }

        for node in [node2, node3, node4] {
            graph.ancestry[node.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::ToSelf,
            });
        }

        graph.deaths.push(node2);
        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);

        assert_eq!(graph.ancestry[node0.as_index()].len(), 2);
        assert!(graph.ancestry[node1.as_index()].is_empty());
        for node in [node3, node4] {
            assert!(graph.ancestry[node0.as_index()].contains(&Ancestry {
                ancestry: AncestryType::Overlap(node),
                segment: Segment {
                    left: 0,
                    right: graph.genome_length()
                }
            }))
        }
        for node in [node1, node2] {
            assert!(node_is_extinct(node, &graph))
        }
    }

    //                     0
    //                   -----
    //                   |   |
    //                   1   2
    //                 ----  ---
    //                 |  |  | |
    //                 3  4  5 6
    //
    //          If we kill node 4 and 6,
    //          nodes 1 and 2 become unary
    //          and node 0 becomes overlap to 3 and 5
    //
    // NOTE:
    //
    // The bug is that node1 is getting "ancestry loss on 1" when
    // it should ALSO get "to unary on 3"
    #[test]
    fn test_subtree_propagation_2() {
        let mut graph = Graph::new(100).unwrap();
        let node0 = graph.add_node(NodeStatus::Ancestor, 0);
        let node1 = graph.add_node(NodeStatus::Ancestor, 1);
        let node2 = graph.add_node(NodeStatus::Ancestor, 1);
        let node3 = graph.add_node(NodeStatus::Ancestor, 2);
        let node4 = graph.add_node(NodeStatus::Death, 2);
        let node5 = graph.add_node(NodeStatus::Ancestor, 2);
        let node6 = graph.add_node(NodeStatus::Death, 2);

        for node in [node1, node2] {
            graph.children[node0.as_index()].insert(node);
            graph.parents[node.as_index()].insert(node0);
            graph.ancestry[node0.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::Overlap(node),
            });
        }
        for node in [node3, node4] {
            graph.children[node1.as_index()].insert(node);
            graph.parents[node.as_index()].insert(node1);
            graph.ancestry[node1.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::Overlap(node),
            });
        }
        for node in [node5, node6] {
            graph.children[node2.as_index()].insert(node);
            graph.parents[node.as_index()].insert(node2);
            graph.ancestry[node2.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::Overlap(node),
            });
        }

        for node in [node3, node4, node5, node6] {
            graph.ancestry[node.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::ToSelf,
            });
        }

        graph.deaths.push(node4);
        graph.deaths.push(node6);
        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);

        assert_eq!(graph.ancestry[node0.as_index()].len(), 2);
        assert!(graph.ancestry[node1.as_index()].is_empty());
        for node in [node3, node5] {
            assert!(graph.ancestry[node0.as_index()].contains(&Ancestry {
                ancestry: AncestryType::Overlap(node),
                segment: Segment {
                    left: 0,
                    right: graph.genome_length()
                }
            }));
            assert_eq!(
                graph.parents[node.as_index()].len(),
                1,
                "{:?}",
                graph.parents[node.as_index()]
            );
            assert!(graph.parents[node.as_index()].contains(&node0));
        }
        assert_eq!(graph.children[node0.as_index()].len(), 2);

        for node in [node1, node2, node4, node6] {
            assert!(node_is_extinct(node, &graph))
        }
        // FIXME: super hack alert.
        // This is not a proper test involving births
        for node in [node3, node5] {
            graph.births.push(node);
        }
        let reachable = reachable_nodes(&graph).collect::<Vec<_>>();
        for node in [node1, node2, node4, node6] {
            assert!(!reachable.contains(&node));
        }
        for node in [node0, node3, node5] {
            assert!(reachable.contains(&node), "node {node:?} is not reachable ");
        }
    }

    #[test]
    fn test_output_node_state_topology0() {
        let graph_fixtures::Topology0 {
            node0,
            node1: _,
            node2,
            node3,
            node4,
            node5,
            mut graph,
        } = graph_fixtures::Topology0::new();

        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
        for node in [node3, node4, node5] {
            assert!(matches!(graph.status[node.as_index()], NodeStatus::Alive));
        }
        for node in [node0, node2] {
            assert!(!graph.ancestry[node.as_index()].is_empty());
            assert!(
                matches!(graph.status[node.as_index()], NodeStatus::Ancestor),
                "{:?} -> {:?}",
                node,
                graph.status[node.as_index()]
            );
        }
    }

    #[test]
    fn test_output_node_state_topology1() {
        let graph_fixtures::Topology1 {
            node0: _,
            node1: _,
            node2,
            node3,
            node4,
            mut graph,
        } = graph_fixtures::Topology1::new();

        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
        for node in [node3, node4] {
            assert!(matches!(graph.status[node.as_index()], NodeStatus::Alive));
        }
        assert!(!graph.ancestry[node2.as_index()].is_empty());
        assert!(
            matches!(graph.status[node2.as_index()], NodeStatus::Ancestor),
            "{:?} -> {:?}",
            node2,
            graph.status[node2.as_index()]
        );
    }

    #[test]
    #[should_panic]
    fn test_unconnected_birth_topology0() {
        let graph_fixtures::Topology0 {
            node0: _,
            node1: _,
            node2: _,
            node3: _,
            node4: _,
            node5: _,
            mut graph,
        } = graph_fixtures::Topology0::new();

        // This birth with not have a transmission
        // connecting it to a parent
        let _ = graph.add_birth(3).unwrap();

        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
    }
}

#[cfg(test)]
mod test_unary_nodes {
    use super::*;

    //                     0
    //                   -----
    //                   |   |
    //                   1   2
    //                 ----  ---
    //                 |  |  | |
    //                 3  4  5 6
    //
    //          If we kill node 4 and 6,
    //          nodes 1 and 2 become unary.
    //
    // With unary retention, the final topology is:
    //
    //                     0
    //                   -----
    //                   |   |
    //                   1   2
    //                   |   |
    //                   |   |
    //                   3   5
    #[test]
    fn test_subtree_propagation_2_with_unary_retention() {
        let graph_fixtures::Topology2 {
            node0,
            node1,
            node2,
            node3,
            node4,
            node5,
            node6,
            mut graph,
        } = graph_fixtures::Topology2::new();
        graph.status[node4.as_index()] = NodeStatus::Death;
        graph.status[node6.as_index()] = NodeStatus::Death;

        graph.deaths.push(node4);
        graph.deaths.push(node6);
        propagate_ancestry_changes(
            PropagationOptions::default().with_keep_unary_nodes(),
            &mut graph,
        );

        assert_eq!(graph.ancestry[node0.as_index()].len(), 2);
        assert_eq!(graph.ancestry[node1.as_index()].len(), 1_usize);
        assert_eq!(graph.ancestry[node2.as_index()].len(), 1);
        assert_eq!(graph.children[node0.as_index()].len(), 2);
        for node in [node1, node2] {
            assert!(graph.children[node0.as_index()].contains(&node));
            assert!(graph.ancestry[node0.as_index()].contains(&Ancestry {
                ancestry: AncestryType::Overlap(node),
                segment: Segment {
                    left: 0,
                    right: graph.genome_length()
                }
            }));
            assert_eq!(
                graph.parents[node.as_index()].len(),
                1,
                "{:?}",
                graph.parents[node.as_index()]
            );
            assert!(graph.parents[node.as_index()].contains(&node0));
        }
        assert!(graph.ancestry[node1.as_index()].contains(&Ancestry {
            segment: Segment {
                left: 0,
                right: graph.genome_length()
            },
            ancestry: AncestryType::Overlap(node3),
        }));
        assert!(graph.ancestry[node2.as_index()].contains(&Ancestry {
            segment: Segment {
                left: 0,
                right: graph.genome_length()
            },
            ancestry: AncestryType::Overlap(node5),
        }));

        for node in [node4, node6] {
            assert!(node_is_extinct(node, &graph))
        }
        // FIXME: super hack alert.
        // This is not a proper test involving births
        for node in [node3, node5] {
            graph.births.push(node);
        }
        let reachable = reachable_nodes(&graph).collect::<Vec<_>>();
        for node in [node4, node6] {
            assert!(!reachable.contains(&node));
        }
        for node in [node0, node1, node2, node3, node5] {
            assert!(reachable.contains(&node), "node {node:?} is not reachable ");
        }
    }

    //                     0
    //                   -----
    //                   |   |
    //                   1   2
    //                 ----  ---
    //                 |  |  | |
    //                 3  4  5 6
    //
    //          If we kill node 5,
    //          node 2 becomes unary.
    //
    // With unary retention, the final topology is:
    //
    //                     0
    //                   -----
    //                   |   |
    //                   1   2
    //                 ---   |
    //                 | |   |
    //                 4 3   6
    #[test]
    fn test_subtree_propagation_with_assymetric_unary_retention() {
        let graph_fixtures::Topology2 {
            node0,
            node1,
            node2,
            node3,
            node4,
            node5,
            node6,
            mut graph,
        } = graph_fixtures::Topology2::new();
        graph.status[node5.as_index()] = NodeStatus::Death;

        graph.deaths.push(node5);
        propagate_ancestry_changes(
            PropagationOptions::default().with_keep_unary_nodes(),
            &mut graph,
        );

        assert_eq!(graph.ancestry[node0.as_index()].len(), 2);
        assert_eq!(graph.ancestry[node1.as_index()].len(), 2_usize);
        assert_eq!(graph.ancestry[node2.as_index()].len(), 1);
        assert_eq!(graph.children[node0.as_index()].len(), 2);
        for node in [node1, node2] {
            assert!(graph.children[node0.as_index()].contains(&node));
            assert!(graph.ancestry[node0.as_index()].contains(&Ancestry {
                ancestry: AncestryType::Overlap(node),
                segment: Segment {
                    left: 0,
                    right: graph.genome_length()
                }
            }));
            assert_eq!(
                graph.parents[node.as_index()].len(),
                1,
                "{:?}",
                graph.parents[node.as_index()]
            );
            assert!(graph.parents[node.as_index()].contains(&node0));
        }
        for node in [node3, node4] {
            assert!(graph.children[node1.as_index()].contains(&node));
            assert!(graph.ancestry[node1.as_index()].contains(&Ancestry {
                ancestry: AncestryType::Overlap(node),
                segment: Segment {
                    left: 0,
                    right: graph.genome_length()
                }
            }));
            assert_eq!(
                graph.parents[node.as_index()].len(),
                1,
                "{:?}",
                graph.parents[node.as_index()]
            );
            assert!(graph.parents[node.as_index()].contains(&node1));
        }
        assert!(graph.ancestry[node2.as_index()].contains(&Ancestry {
            segment: Segment {
                left: 0,
                right: graph.genome_length()
            },
            ancestry: AncestryType::Overlap(node6),
        }));

        assert!(node_is_extinct(node5, &graph));
        // FIXME: super hack alert.
        // This is not a proper test involving births
        for node in [node4, node3, node6] {
            graph.births.push(node);
        }
        let reachable = reachable_nodes(&graph).collect::<Vec<_>>();
        assert!(!reachable.contains(&node5));
        for node in [node0, node1, node2, node3, node4] {
            assert!(reachable.contains(&node), "node {node:?} is not reachable ");
        }
    }
}

#[cfg(test)]
mod test_internal_samples {
    use super::*;

    //                     0
    //                   -----
    //                   |   |
    //                   1   2
    //                 ----  ---
    //                 |  |  | |
    //                 3  4  5 6
    //
    //          If we kill node 5,
    //          node 2 becomes unary.
    //
    // With node2 as an internal sample, the final topology is:
    //
    //                     0
    //                   -----
    //                   |   |
    //                   1   2
    //                 ---   |
    //                 | |   |
    //                 4 3   6
    #[test]
    fn test_subtree_propagation_with_internal_sample() {
        let graph_fixtures::Topology2 {
            node0,
            node1,
            node2,
            node3,
            node4,
            node5,
            node6,
            mut graph,
        } = graph_fixtures::Topology2::new();
        graph.status[node2.as_index()] = NodeStatus::Sample;
        graph.status[node5.as_index()] = NodeStatus::Death;

        graph.deaths.push(node5);
        // With node2 marked as an internal sample, we
        // should get the same results as the previous test
        // with the default options.
        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);

        assert_eq!(graph.ancestry[node0.as_index()].len(), 2);
        assert_eq!(graph.ancestry[node1.as_index()].len(), 2_usize);
        assert_eq!(
            graph.ancestry[node2.as_index()].len(),
            1,
            "{:?}",
            graph.ancestry[node2.as_index()]
        );
        assert_eq!(graph.children[node0.as_index()].len(), 2);
        for node in [node1, node2] {
            assert!(graph.children[node0.as_index()].contains(&node));
            assert!(graph.ancestry[node0.as_index()].contains(&Ancestry {
                ancestry: AncestryType::Overlap(node),
                segment: Segment {
                    left: 0,
                    right: graph.genome_length()
                }
            }));
            assert_eq!(
                graph.parents[node.as_index()].len(),
                1,
                "{:?}",
                graph.parents[node.as_index()]
            );
            assert!(graph.parents[node.as_index()].contains(&node0));
        }
        for node in [node3, node4] {
            assert!(graph.children[node1.as_index()].contains(&node));
            assert!(graph.ancestry[node1.as_index()].contains(&Ancestry {
                ancestry: AncestryType::Overlap(node),
                segment: Segment {
                    left: 0,
                    right: graph.genome_length()
                }
            }));
            assert_eq!(
                graph.parents[node.as_index()].len(),
                1,
                "{:?}",
                graph.parents[node.as_index()]
            );
            assert!(graph.parents[node.as_index()].contains(&node1));
        }
        assert!(graph.ancestry[node2.as_index()].contains(&Ancestry {
            segment: Segment {
                left: 0,
                right: graph.genome_length()
            },
            ancestry: AncestryType::Overlap(node6),
        }));

        assert!(node_is_extinct(node5, &graph));
        // FIXME: super hack alert.
        // This is not a proper test involving births
        for node in [node4, node3, node6] {
            graph.births.push(node);
        }
        let reachable = reachable_nodes(&graph).collect::<Vec<_>>();
        assert!(!reachable.contains(&node5));
        for node in [node0, node1, node2, node3, node4] {
            assert!(reachable.contains(&node), "node {node:?} is not reachable ");
        }
    }

    // Tree 1: [pos0, pos1)
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
    // Tree 2: [pos2, pos3)
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
    //
    // If node 1 is an internal sample, the final topology will be:
    //
    // Tree 1:
    //
    //           1   <- Death
    //           |   <- Death
    //           |
    //         -----
    //         |   |
    //         3   4  <- Birth
    //
    // Tree 2:
    //
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
    //
    //  ALSO, node1 will have ancestry over the ENTIRE GENOME.
    #[test]
    fn test_ancestry_completeness_of_internal_samples() {
        let mut graph = Graph::new(100).unwrap();
        let pos0 = graph.genome_length() / 8;
        let pos1 = graph.genome_length() / 4;
        let pos2 = 2 * graph.genome_length() / 3;
        let pos3 = 3 * graph.genome_length() / 4;
        let node0 = graph.add_node(NodeStatus::Ancestor, 0);
        let node1 = graph.add_node(NodeStatus::Death, 1);
        let node2 = graph.add_node(NodeStatus::Death, 2);
        let node3 = graph.add_birth(3).unwrap();
        let node4 = graph.add_birth(3).unwrap();

        graph.status[node1.as_index()] = NodeStatus::Sample;

        for node in [node1, node2] {
            graph.parents[node.as_index()].insert(node0);
            //record the deaths!!
            graph.deaths.push(node);
        }

        graph.record_transmission(pos0, pos1, node1, node3).unwrap();
        graph.record_transmission(pos0, pos1, node1, node4).unwrap();
        graph.record_transmission(pos2, pos3, node2, node3).unwrap();
        graph.record_transmission(pos2, pos3, node2, node4).unwrap();

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

        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);

        assert_eq!(graph.ancestry[node0.as_index()].len(), 2);
        assert!(graph.ancestry[node0.as_index()].contains(&Ancestry {
            segment: Segment {
                left: pos2,
                right: pos3
            },
            ancestry: AncestryType::Overlap(node2)
        }));
        assert!(graph.ancestry[node0.as_index()].contains(&Ancestry {
            segment: Segment {
                left: pos2,
                right: pos3
            },
            ancestry: AncestryType::Overlap(node1)
        }));

        assert_eq!(graph.ancestry[node2.as_index()].len(), 2);

        for node in [node3, node4] {
            assert!(graph.ancestry[node1.as_index()].contains(&Ancestry {
                segment: Segment {
                    left: pos0,
                    right: pos1
                },
                ancestry: AncestryType::Overlap(node),
            }));
            assert!(graph.ancestry[node2.as_index()].contains(&Ancestry {
                segment: Segment {
                    left: pos2,
                    right: pos3
                },
                ancestry: AncestryType::Overlap(node),
            }));
        }

        // Better than the next thing...
        let mut unique_segments = graph.ancestry[node1.as_index()]
            .iter()
            .map(|a| a.segment)
            .collect::<Vec<_>>();
        unique_segments.sort_unstable();
        unique_segments.dedup();
        let sum: i64 = unique_segments.iter().map(|s| s.right() - s.left()).sum();

        assert_eq!(
            sum,
            graph.genome_length,
            "{:?}",
            graph.ancestry[node1.as_index()]
        );
    }

    ///        0
    ///        |
    ///     -------
    ///     |   | |
    ///     |   | |
    ///     |   1 2  <- Deaths
    ///     3          <- births
    ///
    /// If 0 is an internal sample,
    /// the topology simplifies to:
    ///
    ///       0
    ///       |
    ///       |
    ///       |
    ///       3
    ///
    ///  Node 0 should be Unary(3)
    #[test]
    fn test_dangling_deaths() {
        let mut graph = Graph::new(100).unwrap();
        let node0 = graph.add_node(NodeStatus::Ancestor, 0);
        let node1 = graph.add_node(NodeStatus::Death, 1);
        let node2 = graph.add_node(NodeStatus::Death, 1);
        let node3 = graph.add_node(NodeStatus::Birth, 2);

        graph.status[node0.as_index()] = NodeStatus::Sample;

        for node in [node1, node2] {
            graph.deaths.push(node);
            graph.parents[node.as_index()].insert(node0);
            graph.ancestry[node.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::ToSelf,
            });
            graph.ancestry[node0.as_index()].push(Ancestry {
                segment: Segment {
                    left: 0,
                    right: graph.genome_length,
                },
                ancestry: AncestryType::Overlap(node),
            });
            graph.children[node0.as_index()].insert(node);
        }
        graph
            .record_transmission(0, graph.genome_length(), node0, node3)
            .unwrap();
        graph.births.push(node3);

        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
        for node in [node1, node2] {
            assert!(
                !reachable_nodes(&graph).any(|n| n == node),
                "failing node = {node:?}"
            );
            assert!(graph.free_nodes.contains(&node.0));
        }
        for node in [node0, node3] {
            assert!(
                reachable_nodes(&graph).any(|n| n == node),
                "failing node = {node:?}"
            );
            assert_eq!(
                graph.ancestry[node.as_index()].len(),
                1,
                "failing node = {node:?}, {:?}",
                graph.ancestry[node.as_index()]
            );
        }
        graph.births.clear();
        let _ = graph.add_birth(4).unwrap();
    }
}
