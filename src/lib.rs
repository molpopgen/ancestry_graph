use nohash::BuildNoHashHasher;
use std::collections::HashMap;
use std::collections::HashSet;

// NOTE: for design purposes -- delete later.
mod overlapper_experiments;

mod flags;

use flags::PropagationOptions;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum OverlapState {
    ToSelf,
    ToChild,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ChangeState {
    Birth,
    Loss,
    Overlap,
}

trait SegmentState {}

impl SegmentState for OverlapState {}
impl SegmentState for ChangeState {}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct AncestrySegment<T: SegmentState> {
    segment: Segment,
    node: Node,
    state: T,
}

impl AncestrySegment<OverlapState> {
    fn new_to_self(segment: Segment, node: Node) -> Self {
        Self::new(segment, node, OverlapState::ToSelf)
    }
    fn new_to_child(segment: Segment, node: Node) -> Self {
        Self::new(segment, node, OverlapState::ToChild)
    }

    fn maps_to_child(&self) -> bool {
        matches!(self.state, OverlapState::ToChild)
    }
}

impl AncestrySegment<ChangeState> {
    fn new_loss(segment: Segment, node: Node) -> Self {
        Self::new(segment, node, ChangeState::Loss)
    }

    fn new_overlap(segment: Segment, node: Node) -> Self {
        Self::new(segment, node, ChangeState::Overlap)
    }

    fn new_birth(segment: Segment, node: Node) -> Self {
        Self::new(segment, node, ChangeState::Birth)
    }

    fn is_loss(&self) -> bool {
        matches!(self.state, ChangeState::Loss)
    }

    fn is_birth(&self) -> bool {
        matches!(self.state, ChangeState::Birth)
    }
}

impl<T: SegmentState> AncestrySegment<T> {
    fn new(segment: Segment, node: Node, state: T) -> Self {
        Self {
            segment,
            node,
            state,
        }
    }

    //fn overlaps(&self, other: &Self) -> bool {
    //    self.right > other.left && other.right > self.left
    //}

    fn left(&self) -> i64 {
        self.segment.left()
    }

    fn right(&self) -> i64 {
        self.segment.right()
    }

    fn identical_segment(&self, other: &Self) -> bool {
        self.left() == other.left() && self.right() == other.right()
    }
}

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

#[derive(Clone, Copy, Debug)]
struct CandidateChange {
    source: Node,
    change: AncestrySegment<ChangeState>,
}

/// TODO: could be a newtype?
type NodeHash = HashSet<Node, BuildNoHashHasher<usize>>;
type AncestryChanges = HashMap<Node, Vec<CandidateChange>, BuildNoHashHasher<usize>>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Segment {
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

#[derive(Copy, Clone, Debug)]
struct Transmission {
    parent: Node,
    child: Node,
    left: i64,
    right: i64,
}

impl From<Transmission> for Segment {
    fn from(value: Transmission) -> Self {
        Self {
            left: value.left,
            right: value.right,
        }
    }
}

#[derive(Debug)]
pub struct Graph {
    current_time: i64,
    num_births: usize,
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
    ancestry: Vec<Vec<AncestrySegment<OverlapState>>>,
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
        let deaths = vec![];
        let free_nodes = Vec::new();
        Some(Self {
            current_time: 0,
            num_births: 0,
            status,
            birth_time,
            parents,
            children,
            transmissions,
            ancestry,
            deaths,
            free_nodes,
            genome_length,
        })
    }

    // TODO: remove code duplication to with_capacity if we keep this fn
    pub fn with_initial_nodes(num_nodes: usize, genome_length: i64) -> Option<(Self, Vec<Node>)> {
        let status = vec![NodeStatus::Ancestor; num_nodes];
        let birth_time = vec![Some(0); num_nodes];
        let parents = vec![NodeHash::with_hasher(BuildNoHashHasher::default()); num_nodes];
        let children = vec![NodeHash::with_hasher(BuildNoHashHasher::default()); num_nodes];
        let transmissions = vec![];
        let mut ancestry = vec![];
        let seg = Segment::new(0, genome_length)?;
        for i in 0..num_nodes {
            let anc = AncestrySegment::new_to_self(seg, Node(i));
            ancestry.push(vec![anc]);
        }
        let deaths = vec![];
        let free_nodes = Vec::new();
        let graph = Self {
            current_time: 0,
            num_births: 0,
            status,
            birth_time,
            parents,
            children,
            transmissions,
            ancestry,
            deaths,
            free_nodes,
            genome_length,
        };
        let nodes = graph
            .birth_time
            .iter()
            .enumerate()
            .filter(|(_, t)| t.is_some())
            .map(|(i, _)| Node(i))
            .collect::<Vec<_>>();
        Some((graph, nodes))
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

    fn add_node(&mut self, status: NodeStatus, birth_time: i64) -> Node {
        match self.free_nodes.pop() {
            Some(index) => {
                assert!(self.children[index].is_empty());
                assert!(self.parents[index].is_empty());
                assert!(self.ancestry[index].is_empty());
                assert!(matches!(self.status[index], NodeStatus::Extinct));
                self.birth_time[index] = Some(birth_time);
                self.status[index] = status;
                if matches!(status, NodeStatus::Birth) {
                    self.add_ancestry_to_self(
                        Segment::new(0, self.genome_length).unwrap(),
                        Node(index),
                    )
                }
                Node(index)
            }
            None => {
                self.birth_time.push(Some(birth_time));
                self.status.push(status);
                self.parents
                    .push(NodeHash::with_hasher(BuildNoHashHasher::default()));
                self.children
                    .push(NodeHash::with_hasher(BuildNoHashHasher::default()));
                let index = self.birth_time.len() - 1;
                match status {
                    NodeStatus::Birth => self.ancestry.push(vec![AncestrySegment::new_to_self(
                        Segment::new(0, self.genome_length).unwrap(),
                        Node(index),
                    )]),
                    _ => self.ancestry.push(vec![]),
                }
                Node(index)
            }
        }
    }

    pub fn add_birth(&mut self, birth_time: i64) -> Result<Node, ()> {
        if birth_time != self.current_time {
            return Err(());
        }
        let rv = self.add_node(NodeStatus::Birth, birth_time);
        debug_assert_eq!(self.birth_time[rv.as_index()], Some(birth_time));
        self.num_births += 1;
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
    // TODO: validate left/right
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

    fn add_ancestry_to_self(&mut self, segment: Segment, node: Node) {
        let new_ancestry = AncestrySegment::new_to_self(segment, node);
        debug_assert!(!self.ancestry[node.as_index()].contains(&new_ancestry));
        self.ancestry[node.as_index()].push(new_ancestry);
    }

    fn add_ancestry_to_self_from_raw(&mut self, left: i64, right: i64, node: Node) {
        self.add_ancestry_to_self(Segment::new(left, right).unwrap(), node)
    }

    fn add_ancestry_to_child(&mut self, segment: Segment, parent: Node, child: Node) {
        let new_ancestry = AncestrySegment::new_to_child(segment, child);
        debug_assert!(!self.ancestry[parent.as_index()].contains(&new_ancestry));
        self.ancestry[parent.as_index()].push(new_ancestry)
    }

    fn add_ancestry_to_child_from_raw(&mut self, left: i64, right: i64, parent: Node, child: Node) {
        self.add_ancestry_to_child(Segment::new(left, right).unwrap(), parent, child)
    }

    fn has_ancestry_to_child_raw(
        &mut self,
        left: i64,
        right: i64,
        parent: Node,
        child: Node,
    ) -> bool {
        self.ancestry[parent.as_index()].contains(&AncestrySegment {
            segment: Segment { left, right },
            node: child,
            state: OverlapState::ToChild,
        })
    }

    fn has_ancestry_to_self_raw(&mut self, left: i64, right: i64, node: Node) -> bool {
        self.ancestry[node.as_index()].contains(&AncestrySegment {
            segment: Segment { left, right },
            node,
            state: OverlapState::ToSelf,
        })
    }

    pub fn current_time(&self) -> i64 {
        self.current_time
    }

    pub fn advance_time(&mut self) -> Option<i64> {
        self.advance_time_by(1)
    }

    fn advance_time_by(&mut self, time_delta: i64) -> Option<i64> {
        if time_delta > 0 {
            match self.current_time.checked_add(time_delta) {
                Some(time) => {
                    self.current_time = time;
                    Some(time)
                }
                None => None,
            }
        } else {
            None
        }
    }

    pub fn mark_node_death(&mut self, node: Node) {
        self.status[node.as_index()] = NodeStatus::Death;
        self.deaths.push(node);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum AncestryOverlap {
    Parental(AncestrySegment<OverlapState>),
    Change(AncestrySegment<ChangeState>),
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
    current_parental_node: &AncestrySegment<OverlapState>,
    parental_node_ancestry: &[AncestrySegment<OverlapState>],
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
    parental_node_ancestry: &[AncestrySegment<OverlapState>],
    ancestry_changes: &[CandidateChange],
) -> Vec<AncestryOverlap> {
    assert!(
        ancestry_changes
            .windows(2)
            .all(|w| w[0].change.left() <= w[1].change.left()),
        "{ancestry_changes:?}"
    );
    assert!(!ancestry_changes.is_empty());
    let mut queue = vec![];

    #[cfg(debug_assertions)]
    println!("parental_node_ancestry = {parental_node_ancestry:?}");
    #[cfg(debug_assertions)]
    println!("ancestry_changes = {ancestry_changes:?}");
    let mut d = 0_usize;

    let mut last_left: Option<i64> = None;
    while d < parental_node_ancestry.len() {
        // Take the current node here to minimize bounds checks
        let current_parental_segment = &parental_node_ancestry[d];
        // Ensure that input ancestry are correctly sorted
        // w/o having to do an additional pass over the input
        if let Some(left) = last_left {
            // TODO: decide if this is an assert or an Err path?
            assert!(left <= current_parental_segment.left());
        }
        last_left = Some(current_parental_segment.left());
        // Proposed update:
        // Get rid of the call below
        // Compare current to all:
        // 1. if parental node matches change "source",
        //    queue source, unless it is a loss.
        // 2. Else, queue parental.
        //    - unless it is matched by a loss
        //
        // Should allow:
        //
        // 1. Fixing our major bug.
        // 2. Deleting all this caching of what nodes
        //    we have output to!
        //
        // NOTE: a key difference from tskit
        // is that tskit compared CURRENT EDGES
        // to CHILD ANCESTRY, wich "automagically"
        // filters out the kinds of events that are
        // problematic
        // HERE, we are sending ALL ancestry changes up,
        // which causes the labelling issue.
        // Can we do this more efficiently?
        // It seems that one option is to LABEL AN ANCESTRY
        // SEGMENT WITH ITS PARENT, RATHER THAN RELYING ON
        // "HERE ARE ALL THE PARENTS OF A NODE"
        // todo!("see comments above");
        let update = queue_identical_parental_segments(
            current_parental_segment,
            // Another bounds check...
            &parental_node_ancestry[d..],
            &mut queue,
        );
        for ac in ancestry_changes.iter() {
            if ac.change.right() > current_parental_segment.left()
                && current_parental_segment.right() > ac.change.left()
            {
                let left = std::cmp::max(ac.change.left(), current_parental_segment.left());
                let right = std::cmp::min(ac.change.right(), current_parental_segment.right());
                if ac.change.is_birth() {
                    queue.push(AncestryOverlap::Change(AncestrySegment::new(
                        Segment::new(left, right).unwrap(),
                        ac.change.node,
                        ac.change.state,
                    )));
                } else {
                    for p in &parental_node_ancestry[d..d + update] {
                        if ac.source == p.node {
                            queue.push(AncestryOverlap::Change(AncestrySegment::new(
                                Segment::new(left, right).unwrap(),
                                ac.change.node,
                                ac.change.state,
                            )));
                        }
                    }
                }
            }
            if ac.change.left() >= current_parental_segment.right() {
                break;
            }
        }
        d += update;
    }

    #[cfg(debug_assertions)]
    println!("queue = {queue:?}");

    // TODO: should be an error?.
    // But, an error/assert means that,
    // internally, we MUST not send
    // "no changes" up to parents.
    assert!(!queue.is_empty());

    debug_assert!(queue.windows(2).all(|w| w[0].left() <= w[1].left()));

    queue
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

fn push_ancestry_changes_to_parent<I: Iterator<Item = AncestrySegment<ChangeState>>>(
    parent: Node,
    source: Node,
    ancestry_changes: I,
    ancestry_changes_to_process: &mut AncestryChanges,
) {
    match ancestry_changes_to_process.get_mut(&parent) {
        Some(changes) => {
            changes.extend(ancestry_changes.map(|change| CandidateChange { source, change }));
        }
        None => {
            let changes =
                Vec::from_iter(ancestry_changes.map(|change| CandidateChange { source, change }));
            ancestry_changes_to_process.insert(parent, changes);
        }
    }
}

fn node_is_extinct(node: Node, graph: &Graph) -> bool {
    let index = node.as_index();
    matches!(graph.status[index], NodeStatus::Extinct)
        && graph.birth_time[index].is_none()
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
        let mut queued_nodes: NodeHash = HashSet::with_hasher(BuildNoHashHasher::default());
        for node in graph.birth_time.iter().enumerate().filter_map(|(i, bt)| {
            if bt.is_some() {
                Some(Node(i))
            } else {
                None
            }
        }) {
            if !queued_nodes.contains(&node) {
                node_queue.push(QueuedNode {
                    node,
                    birth_time: graph.birth_time[node.as_index()].unwrap(),
                });
                queued_nodes.insert(node);
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
                assert!(self.queued_nodes.remove(&node.node));
                for parent in &self.graph.parents[node.node.as_index()] {
                    if !self.queued_nodes.contains(parent) {
                        self.queued_nodes.insert(*parent);
                        self.node_queue.push(QueuedNode {
                            node: *parent,
                            birth_time: self.graph.birth_time[parent.as_index()].unwrap(),
                        });
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
    ancestry_changes_to_process: &mut AncestryChanges,
    graph: &mut Graph,
) {
    assert!(!matches!(
        graph.status[queued_parent.node.as_index()],
        NodeStatus::Birth
    ));
    let mut cached_changes = vec![];
    match ancestry_changes_to_process.get_mut(&queued_parent.node) {
        Some(ancestry_changes) => {
            ancestry_changes.sort_unstable_by_key(|ac| ac.change.left());
            let mut overlapper = AncestryOverlapper::new(
                queued_parent.node,
                graph.status[queued_parent.node.as_index()],
                &graph.ancestry[queued_parent.node.as_index()],
                ancestry_changes,
            );
            // Clear parental ancestry
            graph.ancestry[queued_parent.node.as_index()].clear();
            let mut previous_right: i64 = 0;
            while let Some(overlaps) = overlapper.calculate_next_overlap_set(options) {
                #[cfg(debug_assertions)]
                println!(
                    "COORDS: {previous_right} -> ({}, {}), {}",
                    overlaps.left,
                    overlaps.right,
                    matches!(parent_status, NodeStatus::Sample)
                );

                if previous_right != overlaps.left
                    && (matches!(parent_status, NodeStatus::Sample)
                        || matches!(parent_status, NodeStatus::Alive))
                {
                    #[cfg(debug_assertions)]
                    println!("FILLING IN GAP on {} {}", overlaps.left, overlaps.right);
                    graph.add_ancestry_to_self_from_raw(
                        previous_right,
                        overlaps.left,
                        queued_parent.node,
                    )
                }

                // There is some ugliness here: sample nodes are getting
                // ancestry changes marked as not None, which is befuddling
                // all of the logic below.
                if let Some(ancestry_change) = overlaps.parental_ancestry_change {
                    #[cfg(debug_assertions)]
                    println!("change detected for {queued_parent:?} is {ancestry_change:?}");
                    cached_changes.push(ancestry_change);
                } else {
                    #[cfg(debug_assertions)]
                    println!(
                        "no ancestry change detected for {queued_parent:?} on [{}, {})",
                        overlaps.left, overlaps.right
                    );
                }
                // Output the new ancestry for the parent
                match overlaps.parental_ancestry_change {
                    Some(x) if x.is_loss() => {
                        #[cfg(debug_assertions)]
                        println!(
                            "parents of lost node = {:?}",
                            graph.parents[queued_parent.node.as_index()]
                        );

                        #[cfg(debug_assertions)]
                        println!(
                            "children of lost node = {:?}",
                            graph.children[queued_parent.node.as_index()]
                        );
                        cached_changes.extend(
                            overlaps
                                .overlaps()
                                .iter()
                                .map(|a| AncestrySegment::new_overlap(a.segment, a.node)),
                        );
                    }
                    _ => {
                        if (matches!(parent_status, NodeStatus::Sample)
                            || matches!(parent_status, NodeStatus::Alive))
                            && overlaps.overlaps.is_empty()
                        {
                            #[cfg(debug_assertions)]
                            println!("EMPTY on {} {}", overlaps.left, overlaps.right);
                            graph.add_ancestry_to_self_from_raw(
                                overlaps.left,
                                overlaps.right,
                                queued_parent.node,
                            );
                        } else {
                            for &a in overlaps.overlaps() {
                                #[cfg(debug_assertions)]
                                println!(
                                    "adding new ancestry {a:?} to {queued_parent:?}|{}",
                                    overlaps.overlaps.len()
                                );
                                graph.add_ancestry_to_child(a.segment, queued_parent.node, a.node);
                                graph.parents[a.node.as_index()].insert(queued_parent.node);
                                graph.children[queued_parent.node.as_index()].insert(a.node);
                            }
                        }
                    }
                }
                previous_right = overlaps.right;
                #[cfg(debug_assertions)]
                println!("last right = {previous_right}");
            }
            if (matches!(parent_status, NodeStatus::Sample)
                || matches!(parent_status, NodeStatus::Alive))
                && previous_right != graph.genome_length()
            {
                graph.add_ancestry_to_self_from_raw(
                    previous_right,
                    graph.genome_length(),
                    queued_parent.node,
                );
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
        None => panic!(),
    }

    if !cached_changes.is_empty() {
        for parent in graph.parents(queued_parent.node) {
            #[cfg(debug_assertions)]
            println!(
                "{:?} sending {:?} to {parent:?}",
                queued_parent.node, cached_changes
            );
            update_internal_stuff(*parent, hashed_nodes, parent_queue, graph);
            push_ancestry_changes_to_parent(
                *parent,
                queued_parent.node,
                cached_changes.iter().cloned(),
                ancestry_changes_to_process,
            )
        }
    }

    // NOTE: major perf implications
    for c in graph.children[queued_parent.node.as_index()].iter() {
        if !graph.ancestry[queued_parent.node.as_index()]
            .iter()
            .any(|a| &a.node == c)
        {
            graph.parents[c.as_index()].remove(&queued_parent.node);
        }
    }

    // NOTE: major perf implications
    graph.children[queued_parent.node.as_index()].retain(|&child| {
        graph.ancestry[queued_parent.node.as_index()]
            .iter()
            .any(|a| a.node == child || matches!(a.state, OverlapState::ToSelf))
    });

    #[cfg(debug_assertions)]
    println!(
        "final ancestry len of {:?} = {}",
        queued_parent.node,
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

// NOTE: with another name, this type
// is probably good for reuse!
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct OutputSegment {
    segment: Segment,
    node: Node,
}

#[derive(Debug)]
struct Overlaps<'overlapper> {
    left: i64,
    right: i64,
    parent: Node,
    parental_ancestry_change: Option<AncestrySegment<ChangeState>>,
    overlaps: &'overlapper [OutputSegment],
}

impl<'overlapper> Overlaps<'overlapper> {
    fn new(
        left: i64,
        right: i64,
        parent: Node,
        overlaps: &'overlapper [OutputSegment],
        parental_ancestry_change: Option<AncestrySegment<ChangeState>>,
    ) -> Self {
        Self {
            left,
            right,
            parent,
            parental_ancestry_change,
            overlaps,
        }
    }

    fn overlaps(&self) -> &[OutputSegment] {
        self.overlaps
    }
}

fn output_overlaps(
    options: PropagationOptions,
    segment: Segment,
    parent: Node,
    parent_status: NodeStatus,
    parental_overlaps: &[AncestrySegment<OverlapState>],
    parental_nodes_lost: &[Node],
    change_overlaps: &[AncestrySegment<ChangeState>],
    output_ancestry: &mut Vec<OutputSegment>,
) -> Option<AncestrySegment<ChangeState>> {
    output_ancestry.clear();
    for co in change_overlaps {
        if !co.is_loss() {
            output_ancestry.push(OutputSegment {
                segment,
                node: co.node,
            })
        }
    }

    for po in parental_overlaps {
        if po.maps_to_child() && !parental_nodes_lost.contains(&po.node) {
            #[cfg(debug_assertions)]
            println!("ONODE = {:?}", po.node);
            output_ancestry.push(OutputSegment {
                segment,
                node: po.node,
            });
        } else {
            #[cfg(debug_assertions)]
            println!("LNODE = {:?}", po.node);
        }
    }

    let node_is_sample =
        matches!(parent_status, NodeStatus::Sample) || matches!(parent_status, NodeStatus::Alive);

    #[cfg(debug_assertions)]
    println!("the overlaps are {output_ancestry:?}");
    let change_type = match output_ancestry.len() {
        0 => {
            if node_is_sample {
                None
            } else {
                Some(ChangeState::Loss)
            }
        }
        1 => {
            if options.keep_unary_nodes() || node_is_sample {
                None
            } else {
                Some(ChangeState::Loss)
            }
        }
        _ => None,
    };

    change_type.map(|change_type| AncestrySegment::new(segment, parent, change_type))
}

struct AncestryOverlapper {
    queue: Vec<AncestryOverlap>,
    num_overlaps: usize,
    current_overlap: usize,
    parent: Node,
    parent_status: NodeStatus,
    left: i64,
    right: i64,
    overlaps: Vec<OutputSegment>,
    parental_overlaps: Vec<AncestrySegment<OverlapState>>,
    change_overlaps: Vec<AncestrySegment<ChangeState>>,
    parental_nodes_lost: Vec<Node>,
}

impl AncestryOverlapper {
    fn new(
        parent: Node,
        parent_status: NodeStatus,
        parental_node_ancestry: &[AncestrySegment<OverlapState>],
        ancestry_changes: &[CandidateChange],
    ) -> Self {
        let mut queue = generate_overlap_queue(parent, parental_node_ancestry, ancestry_changes);
        let num_overlaps = queue.len();
        // Add sentinel
        queue.push(AncestryOverlap::Parental(AncestrySegment::new_to_self(
            Segment::sentinel(),
            parent,
        )));
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
            parental_overlaps: vec![],
            change_overlaps: vec![],
            parental_nodes_lost: vec![],
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
        self.change_overlaps.retain(|x| {
            if x.right() > self.left {
                if x.is_loss() && !self.parental_nodes_lost.contains(&x.node) {
                    self.parental_nodes_lost.push(x.node)
                }
                true
            } else {
                false
            }
        });
    }

    fn make_overlaps(&mut self, options: PropagationOptions) -> Overlaps {
        #[cfg(debug_assertions)]
        println!(
            "LOST = {:?}, {} {}",
            self.parental_nodes_lost, self.left, self.right
        );
        let ancestry_change = output_overlaps(
            options,
            Segment::new(self.left, self.right).unwrap(),
            self.parent,
            self.parent_status,
            &self.parental_overlaps,
            &self.parental_nodes_lost,
            &self.change_overlaps,
            &mut self.overlaps,
        );
        #[cfg(debug_assertions)]
        println!("self.overlaps = {:?} | {ancestry_change:?}", self.overlaps);
        Overlaps::new(
            self.left,
            self.right,
            self.parent,
            &self.overlaps,
            ancestry_change,
        )
    }

    fn calculate_next_overlap_set(&mut self, options: PropagationOptions) -> Option<Overlaps> {
        self.parental_nodes_lost.clear();
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
                        AncestryOverlap::Change(o) => {
                            if o.is_loss() && !self.parental_nodes_lost.contains(&o.node) {
                                self.parental_nodes_lost.push(o.node);
                            }
                            self.change_overlaps.push(o)
                        }
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
}

// Returns the last ancestor visited.
fn propagate_ancestry_changes(options: PropagationOptions, graph: &mut Graph) -> Option<Node> {
    #[cfg(debug_assertions)]
    println!(
        "the input options are {options:?}, {}",
        options.keep_unary_nodes()
    );
    let mut hashed_nodes: NodeHash = NodeHash::with_hasher(BuildNoHashHasher::default());
    let mut parent_queue: std::collections::BinaryHeap<QueuedNode> =
        std::collections::BinaryHeap::new();
    let mut ancestry_changes_to_process =
        AncestryChanges::with_hasher(BuildNoHashHasher::default());

    let mut num_births_visited = 0_usize;
    for &tranmission in graph.transmissions.iter() {
        if matches!(
            graph.status[tranmission.child.as_index()],
            NodeStatus::Birth
        ) {
            graph.status[tranmission.child.as_index()] = NodeStatus::Alive;
            num_births_visited += 1;
        }
        let change = AncestrySegment::new_birth(tranmission.into(), tranmission.child);
        for parent in graph.parents(tranmission.child) {
            if parent == &tranmission.parent {
                #[cfg(debug_assertions)]
                println!(
                    "updating transmission from {:?} to {:?} on [{}, {}) to parent {parent:?}, and the actual parent is {:?}",
                    parent, tranmission.child, tranmission.left, tranmission.right, tranmission.parent
                );
                update_internal_stuff(*parent, &mut hashed_nodes, &mut parent_queue, graph);
                push_ancestry_changes_to_parent(
                    *parent,
                    tranmission.child,
                    [change].into_iter(),
                    &mut ancestry_changes_to_process,
                );
            }
        }
    }
    graph.transmissions.clear();
    assert_eq!(graph.num_births, num_births_visited);

    for death in graph.deaths.iter() {
        update_internal_stuff(*death, &mut hashed_nodes, &mut parent_queue, graph);
        // Dangling death
        if graph.children[death.as_index()].is_empty() {
            #[cfg(debug_assertions)]
            println!(
                "converting segments {:?} to losses for {death:?}",
                graph.ancestry[death.as_index()]
            );
            for anc in graph.ancestry[death.as_index()].iter() {
                let seg = AncestrySegment::new_loss(anc.segment, *death);
                assert!(seg.is_loss());
                match ancestry_changes_to_process.get_mut(death) {
                    Some(changes) => changes.push(CandidateChange {
                        source: *death,
                        change: seg,
                    }),
                    None => {
                        ancestry_changes_to_process.insert(
                            *death,
                            vec![CandidateChange {
                                source: *death,
                                change: seg,
                            }],
                        );
                    }
                }
            }
        }
    }
    graph.deaths.clear();

    // for q in parent_queue.iter() {
    //     println!("{q:?} -> {:?}", ancestry_changes_to_process.get(&q.node));
    // }

    let mut rv = None;
    while let Some(queued_parent) = parent_queue.pop() {
        rv = Some(queued_parent.node);
        #[cfg(debug_assertions)]
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

    assert!(parent_queue.is_empty());
    assert!(hashed_nodes.is_empty());
    assert!(ancestry_changes_to_process.is_empty());
    assert!(graph.transmissions.is_empty());
    graph.num_births = 0;
    assert!(graph.deaths.is_empty());
    rv
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
            graph.current_time = 3;
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
                graph.ancestry[node0.as_index()].push(AncestrySegment::new_to_child(
                    Segment {
                        left: 0,
                        right: graph.genome_length,
                    },
                    node,
                ));
                graph.ancestry[node.as_index()].push(AncestrySegment::new_to_self(
                    Segment {
                        left: 0,
                        right: graph.genome_length,
                    },
                    node,
                ));
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
            graph.current_time = 3;
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
                graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node0, node);
                graph.add_ancestry_to_self_from_raw(0, graph.genome_length(), node);
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
                graph.add_ancestry_to_child_from_raw(0, graph.genome_length, node0, node);
            }
            for node in [node3, node4] {
                graph.children[node1.as_index()].insert(node);
                graph.parents[node.as_index()].insert(node1);
                graph.add_ancestry_to_child_from_raw(0, graph.genome_length, node1, node);
            }
            for node in [node5, node6] {
                graph.children[node2.as_index()].insert(node);
                graph.parents[node.as_index()].insert(node2);
                graph.add_ancestry_to_child_from_raw(0, graph.genome_length, node2, node);
            }

            for node in [node3, node4, node5, node6] {
                graph.add_ancestry_to_self_from_raw(0, graph.genome_length(), node);
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

    //       0
    //    --------
    //    |      |
    //    1*     2
    //    |      |
    //    3      4  <- Deaths
    //
    //    *Sample nodes
    pub struct Topology3 {
        pub node0: Node,
        pub node1: Node,
        pub node2: Node,
        pub node3: Node,
        pub node4: Node,
        pub graph: Graph,
    }

    impl Topology3 {
        pub fn new() -> Self {
            let mut graph = Graph::new(100).unwrap();
            let node0 = graph.add_node(NodeStatus::Ancestor, 0);
            let node1 = graph.add_node(NodeStatus::Sample, 1);
            let node2 = graph.add_node(NodeStatus::Ancestor, 1);
            let node3 = graph.add_node(NodeStatus::Death, 2);
            let node4 = graph.add_node(NodeStatus::Death, 2);

            for node in [node1, node2] {
                graph.children[node0.as_index()].insert(node);
                graph.parents[node.as_index()].insert(node0);
                graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node0, node)
            }

            graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node2, node4);

            graph.children[node2.as_index()].insert(node4);
            graph.parents[node4.as_index()].insert(node2);
            graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node1, node3);
            graph.children[node1.as_index()].insert(node3);
            graph.parents[node3.as_index()].insert(node1);

            for node in [node3, node4] {
                graph.deaths.push(node);
                graph.add_ancestry_to_self_from_raw(0, graph.genome_length(), node);
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

    //       0
    //    --------
    //    |      |
    //    1      2  <- ALIVE
    //    |      |
    //    3      4  <- Deaths
    //
    //    Alive should be treated like "sample"
    pub struct Topology4 {
        pub node0: Node,
        pub node1: Node,
        pub node2: Node,
        pub node3: Node,
        pub node4: Node,
        pub graph: Graph,
    }

    impl Topology4 {
        pub fn new() -> Self {
            let mut graph = Graph::new(100).unwrap();
            let node0 = graph.add_node(NodeStatus::Ancestor, 0);
            let node1 = graph.add_node(NodeStatus::Alive, 1);
            let node2 = graph.add_node(NodeStatus::Alive, 1);
            let node3 = graph.add_node(NodeStatus::Death, 2);
            let node4 = graph.add_node(NodeStatus::Death, 2);

            for node in [node1, node2] {
                graph.children[node0.as_index()].insert(node);
                graph.parents[node.as_index()].insert(node0);
                graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node0, node);
            }

            graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node2, node4);
            graph.children[node2.as_index()].insert(node4);
            graph.parents[node4.as_index()].insert(node2);
            graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node1, node3);
            graph.children[node1.as_index()].insert(node3);
            graph.parents[node3.as_index()].insert(node1);

            for node in [node3, node4] {
                graph.deaths.push(node);
                graph.add_ancestry_to_self_from_raw(0, graph.genome_length(), node);
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

    // Tree 1:
    //
    //       0
    //    --------
    //    |      |
    //    1      2  <- ALIVE
    //    |      |
    //    3      4  <- Deaths
    //
    //  Tree 2:
    //
    //       0
    //    --------
    //    |      |
    //    1      2  <- ALIVE
    //    |      |
    //    |      |
    //    5      6  <- Births
    //
    //  Alive should be treated like "sample"
    //
    //  NOTE: testing this is showing us that it is currently hard
    //  to fill in sample ancestry "after the fact"
    pub struct Topology5 {
        pub node0: Node,
        pub node1: Node,
        pub node2: Node,
        pub node3: Node,
        pub node4: Node,
        pub node5: Node,
        pub node6: Node,
        pub inner_seg1: Segment,
        pub inner_seg2: Segment,
        pub graph: Graph,
    }

    impl Topology5 {
        pub fn new() -> Self {
            let mut graph = Graph::new(100).unwrap();
            let node0 = graph.add_node(NodeStatus::Ancestor, 0);
            let node1 = graph.add_node(NodeStatus::Alive, 1);
            let node2 = graph.add_node(NodeStatus::Alive, 1);
            let node3 = graph.add_node(NodeStatus::Death, 2);
            let node4 = graph.add_node(NodeStatus::Death, 2);
            graph.current_time = 3; // FAKE IT
            let node5 = graph.add_birth(3).unwrap();
            let node6 = graph.add_birth(3).unwrap();

            let inner_seg1 = Segment::new(10, 20).unwrap();
            let inner_seg2 = Segment::new(40, 43).unwrap();

            graph
                .record_transmission(inner_seg2.left, inner_seg2.right, node1, node5)
                .unwrap();
            graph
                .record_transmission(inner_seg2.left, inner_seg2.right, node2, node6)
                .unwrap();

            for node in [node1, node2] {
                graph.children[node0.as_index()].insert(node);
                graph.parents[node.as_index()].insert(node0);

                for seg in [inner_seg2, inner_seg1] {
                    graph.add_ancestry_to_child_from_raw(seg.left, seg.right, node0, node);
                }
            }

            // NOTE: w/o complete ancestry here, we have
            // test failure later.
            // Thus, it is currently difficult to make a node
            // a sample node after its initial birth time.
            graph.add_ancestry_to_self_from_raw(0, inner_seg1.left, node1);
            graph.add_ancestry_to_child_from_raw(inner_seg1.left, inner_seg1.right, node1, node3);
            graph.add_ancestry_to_self_from_raw(inner_seg1.right, graph.genome_length(), node1);

            graph.add_ancestry_to_self_from_raw(0, inner_seg1.left, node2);
            graph.add_ancestry_to_child_from_raw(inner_seg1.left, inner_seg1.right, node2, node4);
            graph.add_ancestry_to_self_from_raw(inner_seg1.right, graph.genome_length(), node2);

            for node in [node3, node4] {
                graph.deaths.push(node);
                graph.add_ancestry_to_self_from_raw(0, graph.genome_length(), node);
            }
            graph.parents[node3.as_index()].insert(node1);
            graph.parents[node4.as_index()].insert(node2);
            graph.children[node1.as_index()].insert(node3);
            graph.children[node2.as_index()].insert(node4);

            Self {
                node0,
                node1,
                node2,
                node3,
                node4,
                node5,
                node6,
                inner_seg1,
                inner_seg2,
                graph,
            }
        }
    }

    //     0      <- Death
    //   -----
    //   |   |
    //   1   2    <- Birth
    //
    //   Note: see impl of new for details
    pub struct Topology6 {
        pub node0: Node,
        pub node1: Node,
        pub node2: Node,
        pub graph: Graph,
    }

    impl Topology6 {
        pub fn new() -> Self {
            let mut graph = Graph::new(100).unwrap();
            let node0 = graph.add_node(NodeStatus::Death, 0);
            graph.current_time = 1; // FAKE IT
            let node1 = graph.add_birth(1).unwrap();
            let node2 = graph.add_birth(1).unwrap();
            graph.add_ancestry_to_self_from_raw(0, graph.genome_length(), node0);
            graph.record_transmission(0, 50, node0, node1).unwrap();
            graph.record_transmission(0, 100, node0, node2).unwrap();
            graph.deaths.push(node0);

            Self {
                node0,
                node1,
                node2,
                graph,
            }
        }
    }

    ///     0
    ///   -----
    ///   |   |
    ///   1   2 <- deaths
    ///
    /// With 0 as a sample, there should be no output overlaps
    /// and no ancestry change
    pub struct Topology7 {
        pub node0: Node,
        pub node1: Node,
        pub node2: Node,
        pub graph: Graph,
    }

    impl Topology7 {
        pub fn new() -> Self {
            let mut graph = Graph::new(100).unwrap();
            let node0 = graph.add_node(NodeStatus::Sample, 0);
            let node1 = graph.add_node(NodeStatus::Death, 0);
            let node2 = graph.add_node(NodeStatus::Death, 0);

            for node in [node1, node2] {
                graph.parents[node.as_index()].insert(node0);
                graph.children[node0.as_index()].insert(node);
                graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node0, node);
                graph.add_ancestry_to_self_from_raw(0, graph.genome_length(), node);
                graph.deaths.push(node);
            }
            Self {
                node0,
                node1,
                node2,
                graph,
            }
        }
    }

    // 3.00|      0            |    0            | 0                 |                 0 |
    //     |    +-+-+          |  +-+-+          | |                 |                 | |
    // 2.00| 3  1   2          |  1   2    3     | |                 |                 3 |
    //     | | +++ +++         | +++ +++   |     | |                 |                 | |
    // 1.00| 4 | | | |         | | | | |   4     | 4                 | 4               | |
    //     |   | | | |         | | | | | +-++    |                   |                 | |
    // 0.00|   5 6 7 8 9 10 11 | 5 6 7 8 9 10 11 |   5 6 7 8 9 10 11 |   5 6 7 8 9 10 11 |
    //     0                  10                50                  75                  100
    //
    // After standard simplification:
    //
    // {10: 0, 8: 1, 9: 2, 7: 4, 0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 6: 11}
    // 3.00|   10          |   10          |               |
    //     |  +-+-+        |  +-+-+        |               |
    // 2.00|  8   9        |  8   9        |               |
    //     | +++ +++       | +++ +++       |               |
    // 1.00| | | | |       | | | | |  7    |               |
    //     | | | | |       | | | | | +++   |               |
    // 0.00| 0 1 2 3 4 5 6 | 0 1 2 3 4 5 6 | 0 1 2 3 4 5 6 |
    //     0              10              50              100
    // Remap the nodes:
    // 3.00|    0          |    0          |               |
    //     |  +-+-+        |  +-+-+        |               |
    // 2.00|  1   2        |  1   2        |               |
    //     | +++ +++       | +++ +++       |               |
    // 1.00| | | | |       | | | | |  4    |               |
    //     | | | | |       | | | | | +++   |               |
    // 0.00| 0 1 2 3 4 5 6 | 0 1 2 3 4 5 6 | 0 1 2 3 4 5 6 | <- Mentally add 5 to each ;)
    //     0              10              50              100
    pub struct Topology8 {
        pub node0: Node,
        pub node1: Node,
        pub node2: Node,
        pub node3: Node,
        pub node4: Node,
        pub node5: Node,
        pub node6: Node,
        pub node7: Node,
        pub node8: Node,
        pub node9: Node,
        pub node10: Node,
        pub node11: Node,
        pub graph: Graph,
    }

    impl Topology8 {
        pub fn new() -> Self {
            let mut graph = Graph::new(100).unwrap();
            let node0 = graph.add_node(NodeStatus::Ancestor, 0);
            let node1 = graph.add_node(NodeStatus::Death, 1);
            let node2 = graph.add_node(NodeStatus::Death, 1);
            let node3 = graph.add_node(NodeStatus::Death, 1);
            let node4 = graph.add_node(NodeStatus::Death, 2);
            graph.current_time = 3;
            let node5 = graph.add_birth(3).unwrap();
            let node6 = graph.add_birth(3).unwrap();
            let node7 = graph.add_birth(3).unwrap();
            let node8 = graph.add_birth(3).unwrap();
            let node9 = graph.add_birth(3).unwrap();
            let node10 = graph.add_birth(3).unwrap();
            let node11 = graph.add_birth(3).unwrap();

            let seg1 = Segment::new(0, 50).unwrap();
            let seg2 = Segment::new(50, 75).unwrap();
            let seg3 = Segment::new(75, 100).unwrap();

            graph.add_ancestry_to_child(seg1, node0, node1);
            graph.add_ancestry_to_child(seg1, node0, node2);
            graph.add_ancestry_to_child(seg2, node0, node4);
            graph.add_ancestry_to_child(seg3, node0, node3);

            for node in [node1, node2, node3, node4] {
                graph.children[node0.as_index()].insert(node);
                graph.parents[node.as_index()].insert(node0);
            }
            graph.parents[node4.as_index()].insert(node3);

            // graph.add_ancestry_to_child(seg3, node3, node4);

            graph.add_ancestry_to_child(
                Segment::new(0, graph.genome_length()).unwrap(),
                node3,
                node4,
            );
            graph.add_ancestry_to_self(Segment::new(0, graph.genome_length()).unwrap(), node4);
            graph.add_ancestry_to_self(Segment::new(0, graph.genome_length()).unwrap(), node1);
            graph.add_ancestry_to_self(Segment::new(0, graph.genome_length()).unwrap(), node2);

            graph
                .record_transmission(seg1.left(), seg1.right(), node1, node5)
                .unwrap();
            graph
                .record_transmission(seg1.left(), seg1.right(), node1, node6)
                .unwrap();
            graph
                .record_transmission(seg1.left(), seg1.right(), node2, node7)
                .unwrap();
            graph
                .record_transmission(seg1.left(), seg1.right(), node2, node8)
                .unwrap();
            graph
                .record_transmission(seg3.left(), seg3.right(), node3, node11)
                .unwrap();

            graph
                .record_transmission(seg1.left() + 10, seg1.right(), node4, node9)
                .unwrap();
            graph
                .record_transmission(seg1.left() + 10, seg1.right(), node4, node10)
                .unwrap();

            for node in [node1, node2, node3, node4] {
                graph.deaths.push(node)
            }

            Self {
                node0,
                node1,
                node2,
                node3,
                node4,
                node5,
                node6,
                node7,
                node8,
                node9,
                node10,
                node11,
                graph,
            }
        }
    }
}

#[test]
fn design_test_0() {
    let (graph, _) = Graph::with_initial_nodes(10, 1000000).unwrap();
    assert_eq!(graph.genome_length(), 1000000);
}

#[test]
fn design_test_1() {
    let (graph, _) = Graph::with_initial_nodes(10, 1000000).unwrap();
    assert_eq!(graph.iter_nodes_with_ancestry().count(), 10);
}

#[test]
fn enforce_birth_at_current_time() {
    let mut graph = Graph::new(100).unwrap();
    let _ = graph.add_node(NodeStatus::Ancestor, 0);
    assert!(graph.add_birth(1).is_err());
}

#[cfg(test)]
mod test_standard_case {
    use super::*;

    #[test]
    fn test_simple_overlap() {
        let mut graph = Graph::new(100).unwrap();
        let parent = graph.add_node(NodeStatus::Ancestor, 0);
        graph.current_time = 1;
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
        graph.add_ancestry_to_self_from_raw(0, graph.genome_length(), parent);

        for c in [child0, child1] {
            assert_eq!(graph.status[c.as_index()], NodeStatus::Birth);
            assert!(graph.parents(c).any(|&p| p == parent));
        }

        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);

        assert_eq!(graph.ancestry[parent.as_index()].len(), 2);
        for c in [child0, child1] {
            assert!(graph.has_ancestry_to_child_raw(0, graph.genome_length(), parent, c));
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
        graph.current_time = 1;
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
            graph.add_ancestry_to_self_from_raw(0, graph.genome_length, parent);
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
            assert!(graph.has_ancestry_to_child_raw(0, graph.genome_length(), parent0, c));
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

        let last = propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
        assert_eq!(last, Some(node0));

        assert!(graph.ancestry[node1.as_index()].is_empty());
        assert!(graph.parents[node1.as_index()].is_empty());
        assert!(graph.birth_time[node1.as_index()].is_none());
        assert!(graph.free_nodes.contains(&node1.as_index()));

        assert_eq!(graph.ancestry[node2.as_index()].len(), 2);
        for child in [node3, node4] {
            assert!(graph.has_ancestry_to_child_raw(0, graph.genome_length(), node2, child));
        }

        // Node 0
        assert_eq!(graph.ancestry[node0.as_index()].len(), 2);
        for child in [node2, node5] {
            assert!(graph.has_ancestry_to_child_raw(0, graph.genome_length(), node0, child));
        }

        // Verify status changes
        for node in [node3, node4, node5] {
            assert!(matches!(graph.status[node.as_index()], NodeStatus::Alive));
        }
        for node in [node0, node2] {
            assert!(matches!(
                graph.status[node.as_index()],
                NodeStatus::Ancestor
            ));
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
            assert!(graph.has_ancestry_to_child_raw(0, graph.genome_length(), node2, child));
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
        graph.current_time = 3;
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

        // NOTE: we need to add "dummy" ancestry to the parents
        // to have a valid data structure for testing.
        for node in [node1, node2] {
            graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node0, node);
            graph.add_ancestry_to_self_from_raw(0, graph.genome_length(), node);
        }

        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);

        assert!(node_is_extinct(node0, &graph));

        // Node 0
        assert_eq!(graph.ancestry[node0.as_index()].len(), 0);
        for child in [node3, node4] {
            assert_eq!(graph.parents[child.as_index()].len(), 2);
            assert!(graph.has_ancestry_to_child_raw(0, crossover_pos, node1, child));
        }
        for child in [node3, node4] {
            assert_eq!(graph.parents[child.as_index()].len(), 2);
            assert!(graph.has_ancestry_to_child_raw(
                crossover_pos,
                graph.genome_length(),
                node2,
                child
            ));
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
        graph.current_time = 3;
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
            graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node0, node);
            graph.add_ancestry_to_self_from_raw(0, graph.genome_length(), node);
        }
        graph.add_ancestry_to_self_from_raw(0, graph.genome_length(), node2);

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
            assert!(graph.has_ancestry_to_child_raw(0, crossover_pos, node0, child));
            assert_eq!(graph.ancestry[child.as_index()].len(), 1);
            assert!(graph.has_ancestry_to_self_raw(0, graph.genome_length(), child));
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
            graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node0, node);
        }
        for node in [node2, node6] {
            graph.children[node1.as_index()].insert(node);
            graph.parents[node.as_index()].insert(node1);
            graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node1, node);
        }
        for node in [node4, node5] {
            graph.children[node2.as_index()].insert(node);
            graph.parents[node.as_index()].insert(node2);
            graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node2, node);
        }
        for node in [node3, node4, node5, node6] {
            graph.add_ancestry_to_self_from_raw(0, graph.genome_length(), node);
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
            graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node0, node);
        }
        for node in [node2, node3] {
            graph.children[node1.as_index()].insert(node);
            graph.parents[node.as_index()].insert(node1);
            graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node1, node);
        }

        for node in [node2, node3, node4] {
            graph.add_ancestry_to_self_from_raw(0, graph.genome_length(), node)
        }

        graph.deaths.push(node2);
        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);

        assert_eq!(graph.ancestry[node0.as_index()].len(), 2);
        assert!(graph.ancestry[node1.as_index()].is_empty());
        for node in [node3, node4] {
            assert!(graph.has_ancestry_to_child_raw(0, graph.genome_length(), node0, node));
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
            graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node0, node);
        }
        for node in [node3, node4] {
            graph.children[node1.as_index()].insert(node);
            graph.parents[node.as_index()].insert(node1);
            graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node1, node);
        }
        for node in [node5, node6] {
            graph.children[node2.as_index()].insert(node);
            graph.parents[node.as_index()].insert(node2);
            graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node2, node);
        }

        for node in [node3, node4, node5, node6] {
            graph.add_ancestry_to_self_from_raw(0, graph.genome_length(), node);
        }

        graph.deaths.push(node4);
        graph.deaths.push(node6);
        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);

        assert_eq!(graph.ancestry[node0.as_index()].len(), 2);
        assert!(graph.ancestry[node1.as_index()].is_empty());
        for node in [node3, node5] {
            assert!(graph.has_ancestry_to_child_raw(0, graph.genome_length(), node0, node));
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

    // NOTE: this is a case of total extinction.
    // This may be something interesting to detect.
    // The simulation is "done" when such a thing occurs...
    #[test]
    fn test_topology_loss_topology3() {
        let graph_fixtures::Topology3 {
            node0,
            node1,
            node2,
            node3,
            node4,
            mut graph,
        } = graph_fixtures::Topology3::new();

        // NOTE: modify the fixture so that NO NODES are samples
        graph.status[node1.as_index()] = NodeStatus::Ancestor;

        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);

        for node in [node0, node1, node2, node3, node4] {
            assert!(node_is_extinct(node, &graph));
        }
        assert_eq!(graph.free_nodes.len(), graph.birth_time.len());
    }

    #[test]
    fn test_topology4() {
        let graph_fixtures::Topology4 {
            node0,
            node1,
            node2,
            node3,
            node4,
            mut graph,
        } = graph_fixtures::Topology4::new();

        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);

        for node in [node3, node4] {
            assert!(node_is_extinct(node, &graph));
        }
        assert_eq!(graph.free_nodes.len(), 2);
        assert_eq!(graph.ancestry[node0.as_index()].len(), 2);
        assert_eq!(graph.children[node0.as_index()].len(), 2);
        for node in [node1, node2] {
            assert!(graph.children[node.as_index()].is_empty());
            assert!(graph.children[node0.as_index()].contains(&node));
            assert_eq!(graph.parents[node.as_index()].len(), 1);
            assert!(graph.parents[node.as_index()].contains(&node0));
            assert_eq!(graph.ancestry[node.as_index()].len(), 1);
            assert!(graph.has_ancestry_to_self_raw(0, graph.genome_length(), node));
        }
    }

    #[test]
    fn test_topology5() {
        let graph_fixtures::Topology5 {
            node0,
            node1,
            node2,
            node3,
            node4,
            node5,
            node6,
            inner_seg1,
            inner_seg2,
            mut graph,
        } = graph_fixtures::Topology5::new();
        for node in [node0, node1, node2, node3, node4, node5, node6] {
            println!("{node:?} => {:?}", graph.ancestry[node.as_index()]);
            println!("         => {:?}", graph.parents[node.as_index()]);
            println!("         => {:?}", graph.children[node.as_index()]);
        }
        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
        for node in [node3, node4] {
            assert!(node_is_extinct(node, &graph));
        }
        for node in [node0, node1, node2, node5, node6] {
            assert!(!node_is_extinct(node, &graph));
        }

        assert_eq!(graph.free_nodes.len(), 2);
        assert_eq!(graph.ancestry[node0.as_index()].len(), 4);
        assert_eq!(graph.children[node0.as_index()].len(), 2);
        for node in [node1, node2] {
            assert!(graph.children[node0.as_index()].contains(&node));
            assert_eq!(graph.parents[node.as_index()].len(), 1, "{node:?}");
            assert!(graph.parents[node.as_index()].contains(&node0));
            assert!(graph.has_ancestry_to_self_raw(inner_seg1.left, inner_seg1.right, node));
        }
        assert!(graph.has_ancestry_to_child_raw(inner_seg2.left, inner_seg2.right, node1, node5));
        assert!(graph.has_ancestry_to_child_raw(inner_seg2.left, inner_seg2.right, node2, node6));

        // test that nodes 1 and 2 have genome-wide andestry
        for node in [node1, node2] {
            let mut unique_segments = graph.ancestry[node.as_index()]
                .iter()
                .map(|a| a.segment)
                .collect::<Vec<_>>();
            unique_segments.sort_unstable();
            unique_segments.dedup();
            let sum: i64 = unique_segments.iter().map(|s| s.right() - s.left()).sum();
            assert_eq!(
                sum,
                graph.genome_length,
                "{node:?} => {:?}",
                graph.ancestry[node.as_index()]
            );
        }
    }

    #[test]
    fn test_topology6() {
        let graph_fixtures::Topology6 {
            node0,
            node1,
            node2,
            mut graph,
        } = graph_fixtures::Topology6::new();

        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
        for node in [node0, node1, node2] {
            assert!(reachable_nodes(&graph).any(|n| n == node));
        }
        assert_eq!(graph.ancestry[node0.as_index()].len(), 2);
        assert_eq!(graph.children[node0.as_index()].len(), 2);
        for node in [node1, node2] {
            assert!(graph.children[node0.as_index()].contains(&node));
            assert!(graph.parents[node.as_index()].contains(&node0));
            assert_eq!(graph.parents[node.as_index()].len(), 1);
            assert!(graph.has_ancestry_to_child_raw(0, 50, node0, node));
        }
    }

    #[test]
    fn test_topology7() {
        let graph_fixtures::Topology7 {
            node0,
            node1,
            node2,
            mut graph,
        } = graph_fixtures::Topology7::new();
        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
        for node in [node1, node2] {
            assert!(node_is_extinct(node, &graph));
        }
        assert!(graph.children[node0.as_index()].is_empty());
        assert_eq!(graph.ancestry[node0.as_index()].len(), 1);
        assert!(graph.has_ancestry_to_self_raw(0, graph.genome_length(), node0));
    }

    #[test]
    fn test_topology8() {
        let graph_fixtures::Topology8 {
            node0,
            node1,
            node2,
            node3,
            node4,
            node5,
            node6,
            node7,
            node8,
            node9,
            node10,
            node11,
            mut graph,
        } = graph_fixtures::Topology8::new();
        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
        assert!(node_is_extinct(node3, &graph));
        for node in [
            node0, node1, node2, node4, node5, node6, node7, node8, node9, node10, node11,
        ] {
            assert!(!node_is_extinct(node, &graph));
        }

        assert!(
            !graph.children[node0.as_index()].contains(&node4),
            "{:?} <=> {:?}",
            graph.children[node0.as_index()],
            graph.ancestry[node0.as_index()]
        );

        assert!(
            !graph.parents[node4.as_index()].contains(&node0),
            "{:?}",
            graph.parents[node4.as_index()],
        );

        assert_eq!(graph.ancestry[node4.as_index()].len(), 2);
        for node in [node9, node10] {
            assert!(graph.children[node4.as_index()].contains(&node));
            assert!(graph.parents[node.as_index()].contains(&node4));
            assert!(graph.has_ancestry_to_child_raw(10, 50, node4, node));
        }
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
            assert!(graph.has_ancestry_to_child_raw(0, graph.genome_length(), node0, node));
            assert_eq!(
                graph.parents[node.as_index()].len(),
                1,
                "{:?}",
                graph.parents[node.as_index()]
            );
            assert!(graph.parents[node.as_index()].contains(&node0));
        }
        assert!(graph.has_ancestry_to_child_raw(0, graph.genome_length(), node1, node3));
        assert!(graph.has_ancestry_to_child_raw(0, graph.genome_length(), node2, node5));

        for node in [node4, node6] {
            assert!(node_is_extinct(node, &graph))
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
        let last = propagate_ancestry_changes(
            PropagationOptions::default().with_keep_unary_nodes(),
            &mut graph,
        );
        assert!(last.is_some());
        assert_ne!(last, Some(node0));

        assert_eq!(graph.ancestry[node0.as_index()].len(), 2);
        assert_eq!(graph.ancestry[node1.as_index()].len(), 2_usize);
        assert_eq!(graph.ancestry[node2.as_index()].len(), 1);
        assert_eq!(graph.children[node0.as_index()].len(), 2);
        for node in [node1, node2] {
            assert!(graph.children[node0.as_index()].contains(&node));
            assert!(graph.has_ancestry_to_child_raw(0, graph.genome_length(), node0, node));
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
            assert!(graph.has_ancestry_to_child_raw(0, graph.genome_length(), node1, node));
            assert_eq!(
                graph.parents[node.as_index()].len(),
                1,
                "{:?}",
                graph.parents[node.as_index()]
            );
            assert!(graph.parents[node.as_index()].contains(&node1));
        }
        assert!(graph.has_ancestry_to_child_raw(0, graph.genome_length(), node2, node6));

        assert!(node_is_extinct(node5, &graph));
        let reachable = reachable_nodes(&graph).collect::<Vec<_>>();
        assert!(!reachable.contains(&node5));
        for node in [node0, node1, node2, node3, node4] {
            assert!(reachable.contains(&node), "node {node:?} is not reachable ");
        }
    }

    #[test]
    fn test_topology3() {
        let graph_fixtures::Topology3 {
            node0,
            node1,
            node2,
            node3,
            node4,
            mut graph,
        } = graph_fixtures::Topology3::new();

        let last = propagate_ancestry_changes(
            PropagationOptions::default().with_keep_unary_nodes(),
            &mut graph,
        );
        assert_eq!(last, Some(node0));

        for node in [node2, node3, node4] {
            assert!(node_is_extinct(node, &graph));
        }
        assert_eq!(graph.free_nodes.len(), 3);

        for node in [node0, node1] {
            assert_eq!(graph.ancestry[node.as_index()].len(), 1);
        }
        assert_eq!(graph.children[node0.as_index()].len(), 1);
        assert!(graph.children[node1.as_index()].is_empty());
        assert_eq!(graph.parents[node1.as_index()].len(), 1);
        assert!(graph.has_ancestry_to_self_raw(0, graph.genome_length(), node1));
        assert!(graph.has_ancestry_to_child_raw(0, graph.genome_length(), node0, node1));
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
    // NOTES:
    //
    // * should never visit node0
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
        let last_ancestor = propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
        assert!(last_ancestor.is_some());
        assert_ne!(last_ancestor.unwrap(), node0);

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
            graph.has_ancestry_to_child_raw(0, graph.genome_length(), node0, node);
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
            graph.has_ancestry_to_child_raw(0, graph.genome_length(), node1, node);
            assert_eq!(
                graph.parents[node.as_index()].len(),
                1,
                "{:?}",
                graph.parents[node.as_index()]
            );
            assert!(graph.parents[node.as_index()].contains(&node1));
        }
        graph.has_ancestry_to_child_raw(0, graph.genome_length(), node2, node6);

        assert!(node_is_extinct(node5, &graph));
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
    //
    // NOTES:
    //
    // This test has a subtletly to it:
    // 1. By making node0 an overlap on 1,2 for
    //    the whole genome, when we process changes coming
    //    from nodes3,4, we get end up with segments
    //    where there is no overlap to descendant segments.
    //    This scenario triggers the "if sample and no overlaps,
    //    then fill in the gap scenario"
    // 2. The test_topology5 has a different setup that creates
    //    "empty ancestry gaps" because there simply is no parent
    //    nor child segment relevant.
    //
    // The question becomes: why do we need this different logic?
    // It would seem that we are overly complicating things internally.
    // A priori, I'd have thought that the tskit way of doing things would
    // work here, too.
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
        graph.current_time = 3;
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
            graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node0, node);
            graph.add_ancestry_to_self_from_raw(0, graph.genome_length(), node);
        }

        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);

        assert_eq!(graph.ancestry[node0.as_index()].len(), 2);
        assert!(graph.has_ancestry_to_child_raw(pos2, pos3, node0, node2));
        assert!(graph.has_ancestry_to_child_raw(pos2, pos3, node0, node1));

        assert_eq!(graph.ancestry[node2.as_index()].len(), 2);

        for node in [node3, node4] {
            assert!(graph.has_ancestry_to_child_raw(pos0, pos1, node1, node));
            assert!(graph.has_ancestry_to_child_raw(pos2, pos3, node2, node));
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

        assert_eq!(
            graph.ancestry[node1.as_index()]
                .iter()
                .filter(|a| a.segment.left == pos0 && a.segment.right == pos1)
                .count(),
            2
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
        graph.current_time = 2;
        let node3 = graph.add_birth(2).unwrap();

        graph.status[node0.as_index()] = NodeStatus::Sample;

        for node in [node1, node2] {
            graph.deaths.push(node);
            graph.parents[node.as_index()].insert(node0);
            graph.add_ancestry_to_self_from_raw(0, graph.genome_length(), node);
            graph.add_ancestry_to_child_from_raw(0, graph.genome_length(), node0, node);
            graph.children[node0.as_index()].insert(node);
        }
        graph
            .record_transmission(0, graph.genome_length(), node0, node3)
            .unwrap();

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
        graph.current_time = 4;
        let _ = graph.add_birth(4).unwrap();
    }

    #[test]
    fn test_topology3() {
        let graph_fixtures::Topology3 {
            node0,
            node1,
            node2,
            node3,
            node4,
            mut graph,
        } = graph_fixtures::Topology3::new();

        propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
        assert_eq!(graph.num_births, 0);

        for node in [node0, node2, node3, node4] {
            assert!(node_is_extinct(node, &graph));
        }
        assert!(graph.ancestry[node0.as_index()].is_empty());
        assert!(graph.ancestry[node2.as_index()].is_empty());
        assert!(graph.ancestry[node3.as_index()].is_empty());
        assert!(graph.ancestry[node4.as_index()].is_empty());
        assert_eq!(graph.ancestry[node1.as_index()].len(), 1);
        assert!(graph.parents[node1.as_index()].is_empty());
        assert!(graph.children[node1.as_index()].is_empty());
        assert!(graph.has_ancestry_to_self_raw(0, graph.genome_length(), node1));
    }
}

#[cfg(test)]
mod public_api_design {
    use proptest::prelude::*;
    use rand::Rng;
    use rand::SeedableRng;

    use super::Graph;
    use super::Node;

    fn initialize_sim(initial_popsize: usize, genome_length: i64) -> (Graph, Vec<Node>) {
        let (graph, nodes) = Graph::with_initial_nodes(initial_popsize, genome_length).unwrap();
        assert_eq!(nodes.len(), initial_popsize);
        assert_eq!(graph.current_time(), 0);
        (graph, nodes)
    }

    // Standard WF with exactly 1 crossover per bith.
    // This is the same model as the tskit-c/-rust example.
    fn simulate(seed: u64, popsize: usize, genome_length: i64, num_generations: i64) -> Graph {
        assert!(genome_length >= 100); // Arbitrary
        let (mut graph, mut parents) = initialize_sim(popsize, genome_length);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let sample_parent = rand::distributions::Uniform::new(0, popsize);
        let sample_breakpoint = rand::distributions::Uniform::new(1, genome_length);
        let mut children = vec![];

        #[cfg(debug_assertions)]
        println!("CTIME {}", graph.current_time());
        for _ in 0..num_generations {
            children.clear();
            // Advance time
            graph.advance_time().unwrap();
            #[cfg(debug_assertions)]
            println!("CTIME {}", graph.current_time());
            // Mark parents as dead in the graph
            // TODO: mark_node_death needs testing in lib.rs!
            parents.iter().for_each(|&node| graph.mark_node_death(node));
            // Add births
            for _ in 0..popsize {
                let left_parent = parents[rng.sample(sample_parent)];
                let right_parent = parents[rng.sample(sample_parent)];
                let breakpoint = rng.sample(sample_breakpoint);
                // NOTE: we may not need the argument now?
                let child = graph.add_birth(graph.current_time()).unwrap();
                assert!(breakpoint > 0);
                assert!(breakpoint < graph.genome_length());
                graph
                    .record_transmission(0, breakpoint, left_parent, child)
                    .unwrap();
                graph
                    .record_transmission(breakpoint, graph.genome_length(), right_parent, child)
                    .unwrap();
                children.push(child);
            }
            // simplify
            super::propagate_ancestry_changes(super::PropagationOptions::default(), &mut graph);

            // Invariants that "should" be held
            // FIXME: this vector shouldn't exist...
            assert_eq!(graph.num_births, 0);

            std::mem::swap(&mut parents, &mut children);
        }

        graph
    }

    // NOTE: uses internal details
    fn validate_reachable(graph: &Graph) {
        let reachable = Vec::from_iter(super::reachable_nodes(graph));
        for &r in &reachable {
            assert!(!graph.ancestry[r.as_index()].is_empty());
            for p in graph.parents(r) {
                assert!(reachable.contains(p));
            }
            for c in graph.children[r.as_index()].iter() {
                assert!(reachable.contains(c));
            }
            // Assert no overlapping ancestry to the same node
            for (idx, i) in graph.ancestry[r.as_index()].iter().enumerate() {
                for j in graph.ancestry[r.as_index()].iter().skip(idx + 1) {
                    if i.node == j.node {
                        let overlap = i.right() > j.left() && j.right() > i.left();
                        assert!(!overlap, "{:?}, {:?}", i, j);
                    }
                }
            }

            // Assert overlap with every child
            for i in graph.ancestry[r.as_index()].iter() {
                if !matches!(i.state, super::OverlapState::ToSelf) {
                    let mut overlap = false;
                    for c in graph.children[r.as_index()].iter() {
                        for j in graph.ancestry[c.as_index()].iter() {
                            let does_overlap = i.right() > j.left() && j.right() > i.left();
                            if does_overlap {
                                overlap = true;
                                break;
                            }
                        }
                    }
                    assert!(overlap);
                }
            }

            // Assert that ALL parents contain r in their ancestry
            for p in graph.parents[r.as_index()].iter() {
                assert!(graph.ancestry[p.as_index()].iter().any(|a| a.node == r));
            }

            // Assert that ALL children are contained in an overlap
            for c in graph.children[r.as_index()].iter() {
                assert!(graph.ancestry[r.as_index()].iter().any(|a| &a.node == c));
            }
        }
        for (i, a) in graph.ancestry.iter().enumerate() {
            if !a.is_empty() {
                assert!(reachable.contains(&Node(i)));
            } else {
                assert!(!reachable.contains(&Node(i)));
            }
        }
    }

    // FIXME: we are adding redundant ancestry to nodes!!!
    #[test]
    fn failing_test_0() {
        // FAILED b/c our assertion about sortedness of input parental ancestry
        // was "total" but all we needed was "left-wise"
        // Now it fails for the same reason as failing_test_1
        // This second failure was due to not clearing the transmissions
        // array after processing.
        let graph = simulate(0, 4, 100, 100);
        validate_reachable(&graph)
    }

    #[test]
    fn failing_test_1() {
        let graph = simulate(0, 3, 100, 100);
        validate_reachable(&graph)
    }

    proptest! {
        #[test]
        fn test_2_individuals(seed in 0..u64::MAX) {
            let graph = simulate(seed, 2, 100, 10);
            validate_reachable(&graph)
        }
    }

    proptest! {
        #[test]
        fn test_3_individuals(seed in 0..u64::MAX) {
            let graph = simulate(seed, 3, 100, 10);
            validate_reachable(&graph)
        }
    }

    proptest! {
        #[test]
        fn test_5_individuals(seed in 0..u64::MAX) {
            let graph = simulate(seed, 5, 100, 10);
            validate_reachable(&graph)
        }
    }

    #[test]
    #[ignore]
    fn crude_perf_test() {
        let _ = simulate(65134451251234, 1000, 10000000, 500);
    }
}
