use nohash::BuildNoHashHasher;

use crate::Node;
use crate::NodeHash;
use crate::NodeStatus;
use crate::PropagationOptions;
use crate::QueuedNode;

// TODO: this should be in another module
// and made pub for internal use
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, PartialOrd, Ord, Eq)]
pub struct Index(usize);

#[derive(Default, Debug)]
struct NodeHeap {
    queued_nodes: NodeHash,
    node_queue: std::collections::BinaryHeap<QueuedNode>,
}

type UnarySegmentMap = std::collections::HashMap<Index, Index>;

impl NodeHeap {
    fn insert(&mut self, node: Node, birth_time: i64) {
        if !self.queued_nodes.contains(&node) {
            self.queued_nodes.insert(node);
            self.node_queue.push(QueuedNode { node, birth_time });
        }
    }

    fn pop(&mut self) -> Option<Node> {
        if let Some(qn) = self.node_queue.pop() {
            self.queued_nodes.remove(&qn.node);
            Some(qn.node)
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        assert_eq!(self.queued_nodes.len(), self.node_queue.len());
        self.queued_nodes.len()
    }

    fn is_empty(&self) -> bool {
        self.queued_nodes.is_empty()
    }
}

pub trait GenomicInterval {
    fn left(&self) -> i64;
    fn right(&self) -> i64;

    fn range(&self) -> (i64, i64) {
        (self.left(), self.right())
    }

    fn overlaps<T: GenomicInterval>(&self, other: &T) -> bool {
        self.right() > other.left() && other.right() > self.left()
    }
}

impl Index {
    #[inline(always)]
    fn sentinel() -> Self {
        Self(usize::MAX)
    }

    #[inline(always)]
    fn is_sentinel(&self) -> bool {
        self.0 == usize::MAX
    }

    #[inline(always)]
    fn into_option(self) -> Option<Self> {
        if self.is_sentinel() {
            None
        } else {
            Some(self)
        }
    }
}

// TODO: this should be in another module
// and made pub for internal use
#[derive(Debug)]
struct CursorList<T> {
    data: Vec<T>,
    next: Vec<usize>,
    free_list: Vec<usize>,
}

impl<T> CursorList<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            next: Vec::with_capacity(capacity),
            data: Vec::with_capacity(capacity),
            free_list: vec![],
        }
    }

    pub fn get(&self, at: Index) -> &T {
        &self.data[at.0]
    }

    pub fn get_mut(&mut self, at: Index) -> &mut T {
        &mut self.data[at.0]
    }

    pub fn new_index(&mut self, datum: T) -> Index
    where
        T: std::fmt::Debug,
    {
        if let Some(index) = self.free_list.pop() {
            debug_assert!(
                !self.free_list.contains(&index),
                "{index:?} in {:?}",
                self.free_list
            );
            self.next[index] = usize::MAX;
            let _ = std::mem::replace(&mut self.data[index], datum);
            Index(index)
        } else {
            self.next.push(Index::sentinel().0);
            self.data.push(datum);
            Index(self.data.len() - 1)
        }
    }

    fn setup_insertion(&mut self, at: Index, datum: T) -> (Index, Index)
    where
        T: std::fmt::Debug,
    {
        let new_index = self.new_index(datum);
        (new_index, at)
    }

    fn finalize_insertion(&mut self, next: Index, next_value: Index) {
        self.set_next(next, next_value);
    }

    fn set_next(&mut self, at: Index, value: Index) {
        self.next[at.0] = value.0
    }

    pub fn add_list(&mut self, datum: T) -> Index
    where
        T: std::fmt::Debug,
    {
        self.new_index(datum)
    }

    pub fn next(&self, at: Index) -> Option<Index> {
        if at.is_sentinel() {
            None
        } else {
            self.next_raw(at).into_option()
        }
    }

    fn next_raw(&self, at: Index) -> Index {
        Index(self.next[at.0])
    }

    pub fn insert_after(&mut self, at: Index, datum: T) -> Index
    where
        T: std::fmt::Debug,
    {
        let (new_index, index_at) = self.setup_insertion(at, datum);
        if let Some(next) = self.next(at) {
            self.set_next(new_index, next);
        }
        self.finalize_insertion(index_at, new_index);
        new_index
    }

    /// # Panics
    ///
    /// * If `at` is out of range
    pub fn excise_next(&mut self, at: Index) -> Index {
        let next = self.next_raw(at);
        self.next[at.0] = next.0;
        if !next.is_sentinel() {
            debug_assert!(!self.free_list.contains(&next.0));
            self.free_list.push(next.0);
        }
        next
    }

    /// Delete the entire list starting from `from`
    ///
    /// # Panics
    ///
    /// * If `from` is out of range
    fn eliminate(&mut self, from: Index) {
        if !from.is_sentinel() {
            debug_assert!(from.0 < self.next.len(), "{from:?} => {}", self.next.len());
            let mut next = self.excise_next(from);
            while !next.is_sentinel() {
                next = self.excise_next(next);
            }
            self.next[from.0] = usize::MAX;
            debug_assert!(!self.free_list.contains(&from.0));
            self.free_list.push(from.0)
        }
    }

    fn eliminate_and<F: FnMut(&T)>(&mut self, from: Index, mut f: F) {
        assert!(from.0 < self.next.len(), "{from:?} >= {}", self.next.len());
        f(self.get(from));
        let mut next = self.excise_next(from);
        while !next.is_sentinel() {
            f(self.get(next));
            next = self.excise_next(next);
        }
        self.next[from.0] = usize::MAX;
        debug_assert!(!self.free_list.contains(&from.0));
        self.free_list.push(from.0)
    }

    // Excise a node from a list.
    // The Index goes into the free list for
    // later recycling, making it a logic error
    // to use the value of `at` for further operations.
    //pub fn remove(&mut self, at: Index) {
    //    let prev = self.prev_raw(at);
    //    let next = self.next_raw(at);

    //    if !prev.is_sentinel() {
    //        self.set_next(prev, next);
    //    }
    //    if !next.is_sentinel() {
    //        self.set_prev(next, prev);
    //    }
    //    self.set_prev(at, Index::sentinel());
    //    self.set_next(at, Index::sentinel());
    //    self.free_list.push(at.0);
    //}
}

fn update_cursor_list<T>(
    at: usize,
    datum: T,
    head: &mut [Index],
    tail: &mut [Index],
    list: &mut CursorList<T>,
) where
    T: std::fmt::Debug,
{
    let current_tail = tail[at];
    if current_tail.is_sentinel() {
        let new_head = list.new_index(datum);
        head[at] = new_head;
        tail[at] = new_head;
    } else {
        // NOTE: insert_after does not terminate a list,
        // so we have to manually set the next value after
        // the tail to be a sentinel.
        let new_tail = list.insert_after(current_tail, datum);
        tail[at] = new_tail;
        list.next[tail[at].0] = Index::sentinel().0;
    }
}

type NodeAncestry = CursorList<AncestrySegment>;

// NOTE: Eq, PartialEq are only used in testing so far.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AncestrySegment {
    pub left: i64,
    pub right: i64,
    pub parent: Option<Node>,
    pub mapped_node: Node,
}

impl GenomicInterval for AncestrySegment {
    fn left(&self) -> i64 {
        self.left
    }
    fn right(&self) -> i64 {
        self.right
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edge {
    pub left: i64,
    pub right: i64,
    pub child: Node,
}

impl GenomicInterval for Edge {
    fn left(&self) -> i64 {
        self.left
    }
    fn right(&self) -> i64 {
        self.right
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AncestryIntersection {
    left: i64,
    right: i64,
    mapped_node: Node,
    child: Node,
    child_ancestry_segment: Index,
    coalescent: bool,
}

#[derive(Debug)]
struct AncestryOverlapper<'q> {
    queue: &'q [AncestryIntersection],
    num_overlaps: usize,
    current_overlap: usize,
    parent: Node,
    left: i64,
    right: i64,
    overlaps: Vec<AncestryIntersection>,
}

impl<'q> AncestryOverlapper<'q> {
    fn new(parent: Node, queue: &'q [AncestryIntersection]) -> Self {
        let num_overlaps = if queue.is_empty() { 0 } else { queue.len() - 1 };
        let right = if num_overlaps > 0 {
            queue[0].right
        } else {
            i64::MAX
        };
        let left = i64::MAX;
        Self {
            queue,
            num_overlaps,
            parent,
            left,
            right,
            current_overlap: 0,
            overlaps: vec![],
        }
    }

    fn calculate_next_overlap_set(&mut self) -> Option<(i64, i64, &mut [AncestryIntersection])> {
        // NOTE: this if statement hides from the compiler
        // that current_overlap is always < queue.len().
        // We should be able to check current_overlap + 1 <
        // queue.len() and have the later bounds check optimmized out.
        if self.current_overlap < self.num_overlaps {
            self.left = self.right;
            self.overlaps.retain(|o| o.right > self.left);
            if self.overlaps.is_empty() {
                self.left = self.queue[self.current_overlap].left;
            }
            // NOTE: we should be able to get this
            // from the "retain" step.
            // As is, we are going over (part of) overlaps
            // 2x.
            self.right = match self.overlaps.iter().map(|&overlap| overlap.right).min() {
                Some(r) => r,
                None => i64::MAX,
            };
            for segment in &self.queue[self.current_overlap..] {
                if segment.left == self.left {
                    self.current_overlap += 1;
                    self.right = std::cmp::min(self.right, segment.right);
                    self.overlaps.push(*segment)
                } else {
                    break;
                }
            }
            // NOTE: we can track the left value while
            // traversing the overlaps, setting it to MAX
            // initially, and dodge another bounds check
            self.right = std::cmp::min(self.right, self.queue[self.current_overlap].left);
            Some((self.left, self.right, &mut self.overlaps))
        } else {
            if !self.overlaps.is_empty() {
                self.left = self.right;
                // DUPLICATION
                self.overlaps.retain(|o| o.right > self.left);
            }
            if !self.overlaps.is_empty() {
                self.right = match self.overlaps.iter().map(|&overlap| overlap.right).min() {
                    Some(right) => right,
                    None => self.right,
                };
                self.overlaps.retain(|o| o.right > self.left);
                Some((self.left, self.right, &mut self.overlaps))
            } else {
                None
            }
        }
    }
}

// Some definitions:
// * A node w/no edges is extinct if its birth_time != current_time.
pub struct Graph {
    current_time: i64,
    genome_length: i64,
    birth_time: Vec<i64>,
    node_status: Vec<NodeStatus>,
    free_nodes: Vec<usize>,

    num_births: usize,
    // This effectively is just a place to hold
    // new births
    // cached_extant_nodes: Vec<Node>,
    node_heap: NodeHeap,

    // "Tables"
    // Arguably, these could be better encapsulated.
    edges: CursorList<Edge>,
    edge_head: Vec<Index>,
    edge_tail: Vec<Index>,
    ancestry: CursorList<AncestrySegment>,
    ancestry_head: Vec<Index>,
    ancestry_tail: Vec<Index>,
}

impl Graph {
    fn new(genome_length: i64) -> Option<Self> {
        if genome_length < 1 {
            return None;
        }
        let current_time = 0;
        let num_births = 0;
        // let cached_extant_nodes = vec![];
        let birth_time = vec![];
        let node_status = vec![];
        let free_nodes = vec![];
        let edge_head = vec![];
        let edge_tail = vec![];
        let ancestry_head = vec![];
        let ancestry_tail = vec![];
        let edges = CursorList::<Edge>::with_capacity(1000);
        let ancestry = CursorList::<AncestrySegment>::with_capacity(1000);
        let parents = NodeHash::with_hasher(BuildNoHashHasher::default());

        Some(Self {
            current_time,
            genome_length,
            birth_time,
            node_status,
            free_nodes,
            num_births,
            // cached_extant_nodes,
            edges,
            edge_head,
            edge_tail,
            ancestry,
            ancestry_head,
            ancestry_tail,
            node_heap: NodeHeap::default(),
        })
    }

    fn with_initial_nodes(num_nodes: usize, genome_length: i64) -> Option<(Self, Vec<Node>)> {
        let mut graph = Self::new(genome_length)?;
        let mut extant_nodes = vec![];
        for _ in 0..num_nodes {
            let n = graph.add_node(NodeStatus::Ancestor, 0);
            update_cursor_list(
                n.0,
                AncestrySegment {
                    left: 0,
                    right: genome_length,
                    mapped_node: n,
                    parent: None,
                },
                &mut graph.ancestry_head,
                &mut graph.ancestry_tail,
                &mut graph.ancestry,
            );
            extant_nodes.push(n);
        }
        Some((graph, extant_nodes))
    }

    fn add_node(&mut self, status: NodeStatus, birth_time: i64) -> Node {
        match self.free_nodes.pop() {
            Some(index) => {
                let head = self.ancestry_head[index];
                if !head.is_sentinel() {
                    self.ancestry.eliminate(head);
                    self.ancestry_head[index] = Index::sentinel();
                    self.ancestry_tail[index] = Index::sentinel();
                }
                assert!(self.ancestry_head[index].is_sentinel());
                assert!(self.ancestry_tail[index].is_sentinel());
                assert!(self.edge_head[index].is_sentinel());
                assert!(self.edge_tail[index].is_sentinel());
                self.birth_time[index] = birth_time;
                self.node_status[index] = status;
                Node(index)
            }
            None => {
                self.birth_time.push(birth_time);
                self.node_status.push(status);
                self.ancestry_head.push(Index::sentinel());
                self.ancestry_tail.push(Index::sentinel());
                self.edge_head.push(Index::sentinel());
                self.edge_tail.push(Index::sentinel());
                let index = self.birth_time.len() - 1;
                Node(index)
            }
        }
    }

    pub fn add_birth(&mut self, birth_time: i64) -> Result<Node, ()> {
        if birth_time != self.current_time {
            return Err(());
        }
        let rv = self.add_node(NodeStatus::Birth, birth_time);
        debug_assert_eq!(self.birth_time[rv.as_index()], birth_time);
        self.num_births += 1;
        // self.cached_extant_nodes.push(rv);
        Ok(rv)
    }

    pub fn advance_time(&mut self) -> Option<i64> {
        self.advance_time_by(1)
    }

    fn advance_time_by(&mut self, time_delta: i64) -> Option<i64> {
        if time_delta > 0 {
            self.current_time = self.current_time.checked_add(time_delta)?;
            Some(self.current_time)
        } else {
            None
        }
    }

    fn validate_parent_child_birth_time(&self, parent: Node, child: Node) -> Result<(), ()> {
        let ptime = self.birth_time.get(parent.as_index()).ok_or(())?;
        let ctime = self.birth_time.get(child.as_index()).ok_or(())?;
        if ctime > ptime {
            Ok(())
        } else {
            Err(())
        }
    }

    fn validate_record_transmission_input(
        &self,
        left: i64,
        right: i64,
        parent: Node,
        child: Node,
    ) -> Result<(), ()> {
        if right <= left {
            return Err(());
        }
        if right <= 0 {
            return Err(());
        }
        if left < 0 {
            return Err(());
        }
        self.validate_parent_child_birth_time(parent, child)?;
        let child_ancestry_tail = self.ancestry_tail[child.0];
        if !child_ancestry_tail.is_sentinel()
            && self.ancestry.get(child_ancestry_tail).right != left
        {
            return Err(());
        }
        Ok(())
    }

    fn record_transmission(
        &mut self,
        left: i64,
        right: i64,
        parent: Node,
        child: Node,
    ) -> Result<(), ()> {
        self.validate_record_transmission_input(left, right, parent, child)?;
        update_cursor_list(
            parent.0,
            Edge { left, right, child },
            &mut self.edge_head,
            &mut self.edge_tail,
            &mut self.edges,
        );
        update_cursor_list(
            child.0,
            AncestrySegment {
                left,
                right,
                parent: Some(parent),
                mapped_node: child,
            },
            &mut self.ancestry_head,
            &mut self.ancestry_tail,
            &mut self.ancestry,
        );
        //debug_assert!(self.ancestry.next(self.ancestry_tail[child.0]).is_none());
        self.node_heap
            .insert(parent, self.birth_time[parent.as_index()]);
        Ok(())
    }

    pub fn mark_node_death(&mut self, node: Node) {
        self.node_status[node.as_index()] = NodeStatus::Death;
        self.node_heap
            .insert(node, self.birth_time[node.as_index()]);
    }
}

fn ancestry_intersection(node: Node, graph: &Graph, queue: &mut Vec<AncestryIntersection>) {
    queue.clear();

    let mut current_edge = match graph.edge_head.get(node.as_index()) {
        Some(&index) if !index.is_sentinel() => Some(index),
        _ => None,
    };

    while let Some(edge_index) = current_edge {
        let edge_ref = graph.edges.get(edge_index);
        let mut child_ancestry = {
            let a = graph.ancestry_head[edge_ref.child.as_index()];
            if a.is_sentinel() {
                None
            } else {
                Some(a)
            }
        };
        //if child_ancestry.is_some() {
        //    assert!(graph
        //        .ancestry
        //        .next(graph.ancestry_tail[edge_ref.child.as_index()])
        //        .is_none())
        //}
        while let Some(child_ancestry_index) = child_ancestry {
            let anc_ref = graph.ancestry.get(child_ancestry_index);
            if edge_ref.overlaps(anc_ref) {
                let left = std::cmp::max(edge_ref.left, anc_ref.left);
                let right = std::cmp::min(edge_ref.right, anc_ref.right);
                queue.push(AncestryIntersection {
                    left,
                    right,
                    mapped_node: anc_ref.mapped_node,
                    child: edge_ref.child,
                    child_ancestry_segment: child_ancestry_index,
                    coalescent: false,
                });
            }
            child_ancestry = graph.ancestry.next(child_ancestry_index);
        }
        current_edge = graph.edges.next(edge_index);
    }
    queue.sort_unstable_by_key(|x| x.left);
    if !queue.is_empty() {
        queue.push(AncestryIntersection {
            left: i64::MAX,
            right: i64::MAX,
            mapped_node: Node(usize::MAX),
            child: Node(usize::MAX),
            child_ancestry_segment: Index::sentinel(),
            coalescent: false,
        });
    }
}

fn update_ancestry(
    left: i64,
    right: i64,
    mapped_node: Node,
    current_ancestry_index: Index,
    birth_time: &[i64],
    ancestry: &mut NodeAncestry,
    node_heap: &mut NodeHeap,
) -> Index {
    let mut seg_right = None;
    let current = *ancestry.get(current_ancestry_index);
    let temp_left = std::cmp::max(current.left, left);
    let temp_right = std::cmp::min(current.right, right);
    let mut rv = ancestry.next_raw(current_ancestry_index);
    if current.left != temp_left {
        assert!(current.left < temp_left);
        if let Some(parent) = current.parent {
            node_heap.insert(parent, birth_time[parent.as_index()]);
        }
    }
    if current.right != temp_right {
        if let Some(parent) = current.parent {
            node_heap.insert(parent, birth_time[parent.as_index()]);
        }
        {
            let current = ancestry.get_mut(current_ancestry_index);
            current.left = temp_right;
        }
        seg_right = Some(AncestrySegment {
            left: temp_right,
            ..current
        });
    }

    if mapped_node != current.mapped_node {
        // This is a change from "coalescent" to "unary" on this
        // segment, which is a change we must propagate.
        if let Some(parent) = current.parent {
            node_heap.insert(parent, birth_time[parent.as_index()]);
        }
    }

    let out_seg = AncestrySegment {
        left,
        right,
        mapped_node,
        parent: None,
    };
    // TODO: API fn to replace.
    *ancestry.get_mut(current_ancestry_index) = out_seg;

    if let Some(right_seg) = seg_right {
        let next = ancestry.next_raw(current_ancestry_index);
        let new_index = ancestry.new_index(right_seg);
        ancestry.next[current_ancestry_index.0] = new_index.0;
        ancestry.next[new_index.0] = next.0;
        rv = new_index;
    }
    rv
}

fn record_total_loss_of_ancestry(queued_parent: Node, graph: &mut Graph) {
    graph
        .edges
        .eliminate(graph.edge_head[queued_parent.as_index()]);
    graph.edge_head[queued_parent.as_index()] = Index::sentinel();
    graph.edge_tail[queued_parent.as_index()] = Index::sentinel();

    graph.ancestry.eliminate_and(
        graph.ancestry_head[queued_parent.as_index()],
        |a: &AncestrySegment| {
            if let Some(parent) = a.parent {
                graph
                    .node_heap
                    .insert(parent, graph.birth_time[parent.as_index()]);
            }
        },
    );
    graph.ancestry_head[queued_parent.as_index()] = Index::sentinel();
    graph.ancestry_tail[queued_parent.as_index()] = Index::sentinel();
    assert!(!graph.free_nodes.contains(&queued_parent.as_index()));
    graph.free_nodes.push(queued_parent.as_index());
}

fn process_queued_node(
    options: PropagationOptions,
    queued_parent: Node,
    parent_status: NodeStatus,
    graph: &mut Graph,
    queue: &[AncestryIntersection],
    temp_edges: &mut Vec<Edge>,
    unary_segment_map: &mut UnarySegmentMap,
) {
    let mut ahead = graph.ancestry_head[queued_parent.as_index()];
    let mut last_ancestry_index = ahead;

    let mut overlapper = AncestryOverlapper::new(queued_parent, queue);

    let mut overlaps = overlapper.calculate_next_overlap_set();

    while !ahead.is_sentinel() {
        if let Some((left, right, ref mut current_overlaps)) = overlaps {
            let (current_left, current_right) = {
                let current = graph.ancestry.get(ahead);
                (current.left, current.right)
            };
            if current_right > left && right > current_left {
                let mapped_node;
                let mut unary_segment = None;
                if current_overlaps.len() == 1 {
                    mapped_node = current_overlaps[0].mapped_node;
                    if !current_overlaps[0].coalescent {
                        let aseg_index = current_overlaps[0].child_ancestry_segment;
                        if let Some(un) = unary_segment_map.get(&aseg_index) {
                            unary_segment = Some(*un);
                            unary_segment_map.remove(&aseg_index);
                        } else {
                            unary_segment = Some(aseg_index);
                        }
                    }
                    if let Some(parent) = graph.ancestry.get(ahead).parent {
                        graph
                            .node_heap
                            .insert(parent, graph.birth_time[parent.as_index()]);
                    }
                } else {
                    mapped_node = queued_parent;
                    for o in current_overlaps.iter_mut() {
                        if let Some(un) = unary_segment_map.get(&o.child_ancestry_segment) {
                            graph.ancestry.data[un.0].parent = Some(queued_parent);
                            unary_segment_map.remove(&o.child_ancestry_segment);
                        }
                        temp_edges.push(Edge {
                            left,
                            right,
                            child: o.mapped_node,
                        });
                        graph.ancestry.data[o.child_ancestry_segment.0].parent =
                            Some(queued_parent);
                        o.coalescent = true;
                    }
                }
                last_ancestry_index = ahead;
                ahead = update_ancestry(
                    left,
                    right,
                    mapped_node,
                    ahead,
                    &graph.birth_time,
                    &mut graph.ancestry,
                    &mut graph.node_heap,
                );
                if let Some(useg) = unary_segment {
                    debug_assert!(!unary_segment_map.contains_key(&last_ancestry_index));
                    unary_segment_map.insert(last_ancestry_index, useg);
                }
                if !last_ancestry_index.is_sentinel() {
                    if let Some(useg) = unary_segment_map.get(&last_ancestry_index) {
                        graph.ancestry.data[useg.0].parent = None;
                    }
                }
                overlaps = overlapper.calculate_next_overlap_set();
            } else if last_ancestry_index == ahead {
                let next = graph.ancestry.next_raw(ahead);
                if !next.is_sentinel() {
                    if let Some(parent) = graph.ancestry.get(next).parent {
                        graph
                            .node_heap
                            .insert(parent, graph.birth_time[parent.as_index()])
                    }
                    graph.ancestry.data.swap(ahead.0, next.0);
                    graph.ancestry.next[ahead.0] = graph.ancestry.next[next.0];
                    debug_assert!(!graph.ancestry.free_list.contains(&next.0));
                    graph.ancestry.free_list.push(next.0);
                } else {
                    last_ancestry_index = ahead;
                    ahead = next;
                }
            } else {
                debug_assert!(
                    !graph.ancestry.free_list.contains(&ahead.0),
                    "{ahead:?} in {:?}",
                    graph.ancestry.free_list
                );
                // Will panic if ahead is sentinel, which is desired b/c
                // it'll let us know when we get // test coverrage here.
                if let Some(parent) = graph.ancestry.get(ahead).parent {
                    graph
                        .node_heap
                        .insert(parent, graph.birth_time[parent.as_index()])
                }
                ahead = graph.ancestry.excise_next(last_ancestry_index);
                let next = graph.ancestry.next_raw(ahead);
                graph.ancestry.next[last_ancestry_index.0] = next.0;
                ahead = next;
            }
        } else {
            break;
        }
    }

    if !ahead.is_sentinel() {
        let mut z = graph.ancestry.next(last_ancestry_index);
        // TODO: each of these is a right overhang
        // that we need to reckon with.
        while let Some(index) = z {
            if let Some(parent) = graph.ancestry.get(index).parent {
                graph
                    .node_heap
                    .insert(parent, graph.birth_time[parent.as_index()])
            }

            z = graph.ancestry.next(index);
            graph.ancestry.next[index.0] = usize::MAX;
            debug_assert!(!graph.ancestry.free_list.contains(&index.0));
            graph.ancestry.free_list.push(index.0);
        }
        graph.ancestry.next[last_ancestry_index.0] = usize::MAX;
    }

    let mut ahead = graph.ancestry_head[queued_parent.as_index()];
    while !ahead.is_sentinel() {
        if ahead == last_ancestry_index {
            break;
        }

        ahead = graph.ancestry.next_raw(ahead);
    }

    if temp_edges.is_empty() {
        let mut e = graph.edge_head[queued_parent.as_index()];
        while !e.is_sentinel() {
            let next = graph.edges.next_raw(e);
            graph.edges.next[e.0] = usize::MAX;
            debug_assert!(!graph.edges.free_list.contains(&e.0));
            graph.edges.free_list.push(e.0);
            e = next;
        }
        graph.edge_head[queued_parent.as_index()] = Index::sentinel();
        graph.edge_tail[queued_parent.as_index()] = Index::sentinel();
        assert!(!graph.free_nodes.contains(&queued_parent.as_index()));
        graph.free_nodes.push(queued_parent.as_index());
    } else {
        #[cfg(debug_assertions)]
        {
            let mut e = graph.edge_head[queued_parent.as_index()];
            let mut v = vec![];
            while !e.is_sentinel() {
                assert!(!v.contains(&e));
                v.push(e);
                e = graph.edges.next_raw(e);
            }
        }
        let mut e = graph.edge_head[queued_parent.as_index()];
        let mut last_e = e;
        for edge in temp_edges.iter() {
            if !e.is_sentinel() {
                *graph.edges.get_mut(e) = *edge;
                last_e = e;
                e = graph.edges.next_raw(e);
            } else {
                last_e = graph.edges.insert_after(last_e, *edge);
            }
        }
        // Recycle extraneous edges
        if !graph.edges.next_raw(last_e).is_sentinel() {
            let mut z = graph.edges.next(last_e);
            while let Some(index) = z {
                z = graph.edges.next(index);
                graph.edges.next[index.0] = usize::MAX;
                debug_assert!(
                    !graph.edges.free_list.contains(&index.0),
                    "{index:?} in {:?}",
                    graph.edges.free_list
                );
                graph.edges.free_list.push(index.0);
            }
            graph.edges.next[last_e.0] = usize::MAX;
        }
        graph.edge_tail[queued_parent.as_index()] = last_e;
    }

    graph.ancestry_tail[queued_parent.as_index()] = last_ancestry_index;
    debug_assert!(graph
        .ancestry
        .next(graph.ancestry_tail[queued_parent.as_index()])
        .is_none());

    #[cfg(debug_assertions)]
    {
        if !graph.edge_tail[queued_parent.as_index()].is_sentinel() {
            assert!(graph
                .edges
                .next(graph.edge_tail[queued_parent.as_index()])
                .is_none());
            assert!(!graph.edge_head[queued_parent.as_index()].is_sentinel());
        } else {
            assert!(graph.free_nodes.contains(&queued_parent.as_index()))
        }
        if !graph.ancestry_tail[queued_parent.as_index()].is_sentinel() {
            assert!(graph
                .ancestry
                .next(graph.ancestry_tail[queued_parent.as_index()])
                .is_none());
            assert!(!graph.ancestry_head[queued_parent.as_index()].is_sentinel());
        }
    }
}

// returns the value of the last (oldest, ignoring ties) Node processed
// The return value has no meaning beyond testing and should eventually
// be deleted.
fn propagate_ancestry_changes(options: PropagationOptions, graph: &mut Graph) -> Option<Node> {
    let mut temp_edges = vec![];

    let mut queue = vec![];
    let mut rv = None;
    let mut unary_segment_map = UnarySegmentMap::default();
    while let Some(queued_node) = graph.node_heap.pop() {
        rv = Some(queued_node);
        ancestry_intersection(queued_node, graph, &mut queue);
        if queue.is_empty() {
            // There are no overlaps with children.
            // The current node loses all ancestry
            // and any parents are added to the node heap.
            record_total_loss_of_ancestry(queued_node, graph);
        } else {
            process_queued_node(
                options,
                queued_node,
                graph.node_status[queued_node.as_index()],
                graph,
                &queue,
                &mut temp_edges,
                &mut unary_segment_map,
            );
        }
        // Clean up for next loop
        queue.clear();
        temp_edges.clear();
    }

    // TODO: this should be some "post simplify cleanup" step
    graph.num_births = 0;

    debug_assert!(graph.node_heap.is_empty());
    rv
}

#[cfg(test)]
fn haploid_wf(seed: u64, popsize: usize, genome_length: i64, num_generations: i64) -> Graph {
    use rand::Rng;
    use rand::SeedableRng;
    let (mut graph, mut parents) = Graph::with_initial_nodes(popsize, genome_length).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let sample_parent = rand::distributions::Uniform::new(0, popsize);
    let sample_breakpoint = rand::distributions::Uniform::new(1, genome_length);
    let mut children: Vec<Node> = vec![];

    for gen in 0..num_generations {
        //println!(
        //    "{gen}, {}, {}, {}|{}",
        //    graph.node_status.len(),
        //    graph.free_nodes.len(),
        //    graph
        //        .birth_time
        //        .iter()
        //        .enumerate()
        //        .filter(|(i, _)| { !graph.edge_head[*i].is_sentinel() })
        //        .count(),
        //    graph
        //        .birth_time
        //        .iter()
        //        .enumerate()
        //        .filter(|(i, _)| { !graph.ancestry_head[*i].is_sentinel() })
        //        .count()
        //);
        children.clear();
        // Advance time
        graph.advance_time().unwrap();
        // Mark parents as dead in the graph
        // TODO: mark_node_death needs testing in lib.rs!
        parents.iter().for_each(|&node| graph.mark_node_death(node));
        // Add births
        for _ in 0..popsize {
            let left_parent = parents[rng.sample(sample_parent)];
            let right_parent = parents[rng.sample(sample_parent)];
            let breakpoint = rng.sample(sample_breakpoint);
            let child = graph.add_birth(graph.current_time).unwrap();
            // NOTE: we may not need the argument now?
            assert!(breakpoint > 0);
            assert!(breakpoint < graph.genome_length);
            graph
                .record_transmission(0, breakpoint, left_parent, child)
                .unwrap();
            graph
                .record_transmission(breakpoint, graph.genome_length, right_parent, child)
                .unwrap();
            children.push(child);
        }
        // simplify
        propagate_ancestry_changes(super::PropagationOptions::default(), &mut graph);

        // Invariants that "should" be held
        // FIXME: this vector shouldn't exist...
        assert_eq!(graph.num_births, 0);

        std::mem::swap(&mut parents, &mut children);
    }

    graph
}

#[cfg(test)]
mod sim_test {
    use super::haploid_wf;
    use proptest::prelude::*;

    #[test]
    fn foo_test_2_individuals() {
        let graph = haploid_wf(709617927814905890, 2, 100, 100);
        //validate_reachable(&graph)
    }

    #[test]
    fn foo_foo_test_2_individuals() {
        let graph = haploid_wf(5402151571545481800, 2, 100, 100);
        //validate_reachable(&graph)
    }

    proptest! {
        #[test]
        fn test_2_individuals(seed in 0..u64::MAX) {
            let graph = haploid_wf(seed, 2, 100, 10);
            //validate_reachable(&graph)
        }
    }

    proptest! {
        #[test]
        fn test_3_individuals(seed in 0..u64::MAX) {
            let graph = haploid_wf(seed, 3, 100, 10);
            //validate_reachable(&graph)
        }
    }

    proptest! {
        #[test]
        fn test_5_individuals(seed in 0..u64::MAX) {
            let graph = haploid_wf(seed, 5, 100, 10);
            //validate_reachable(&graph)
        }
    }

    #[test]
    fn test_1000_individuals_fixed() {
        let graph = haploid_wf(1235125152, 1000, 100000000, 1000);
        println!("{} {}", graph.edges.data.len(), graph.ancestry.data.len());
        let mut mean_e = 0;
        let mut mean_a = 0;
        let mut n = 0;
        for i in 0..graph.birth_time.len() {
            if !graph.ancestry_head[i].is_sentinel() {
                let mut a = graph.ancestry_head[i];
                while !a.is_sentinel() {
                    mean_a += 1;
                    a = graph.ancestry.next_raw(a);
                }
                let mut e = graph.edge_head[i];
                while !e.is_sentinel() {
                    mean_e += 1;
                    e = graph.edges.next_raw(e);
                }
                n += 1;
            }
        }
        println!(
            "{}, {} | {n}",
            (mean_a as f64) / (n as f64),
            (mean_e as f64) / (n as f64)
        );
        //validate_reachable(&graph)
    }
}

#[cfg(test)]
mod test_utils {
    use super::*;

    #[must_use]
    pub(super) fn setup_ancestry(
        input_ancestry: &[Vec<(i64, i64, Node)>],
        ancestry: &mut CursorList<AncestrySegment>,
    ) -> (Vec<Index>, Vec<Index>) {
        let mut ancestry_head = vec![];
        let mut ancestry_tail = vec![];

        for inner in input_ancestry {
            if !inner.is_empty() {
                let head = ancestry.new_index(AncestrySegment {
                    left: inner[0].0,
                    right: inner[0].1,
                    parent: None, // FIXME
                    mapped_node: inner[0].2,
                });
                ancestry_head.push(head);
                let mut last = head;
                for (left, right, mapped_node) in inner.iter().skip(1) {
                    last = ancestry.insert_after(
                        last,
                        AncestrySegment {
                            left: *left,
                            right: *right,
                            parent: None, // FIXME
                            mapped_node: *mapped_node,
                        },
                    );
                }
                ancestry_tail.push(last);
            }
        }

        (ancestry_head, ancestry_tail)
    }

    pub(super) fn extract<T>(
        at: usize,
        head: &[Index],
        tail: &[Index],
        list: &CursorList<T>,
    ) -> Vec<(T, Index)>
    where
        T: Copy,
    {
        let mut rv = vec![];
        let mut cursor = {
            let h = head[at];
            if h.is_sentinel() {
                None
            } else {
                Some(h)
            }
        };
        while let Some(index) = cursor {
            rv.push((*list.get(index), index));
            cursor = list.next(index);
            if cursor.is_none() {
                assert_eq!(tail[at], index);
            }
        }

        rv
    }

    pub(super) fn naive_ancestry_intersection(
        parent: Node,
        graph: &Graph,
    ) -> Vec<AncestryIntersection> {
        let mut rv = vec![];
        let edges = extract(
            parent.as_index(),
            &graph.edge_head,
            &graph.edge_tail,
            &graph.edges,
        );

        for (edge, edge_index) in edges {
            let child_ancestry = extract(
                edge.child.as_index(),
                &graph.ancestry_head,
                &graph.ancestry_tail,
                &graph.ancestry,
            )
            .into_iter()
            .map(|(a, _)| a)
            .collect::<Vec<_>>();
            for (i, a) in child_ancestry.into_iter().enumerate() {
                if a.right > edge.left && edge.right > a.left {
                    let left = std::cmp::max(a.left, edge.left);
                    let right = std::cmp::min(a.right, edge.right);
                    rv.push(AncestryIntersection {
                        left,
                        right,
                        mapped_node: a.mapped_node,
                        child: edge.child,
                        child_ancestry_segment: Index(i),
                        coalescent: false,
                    });
                }
            }
        }
        rv.sort_unstable_by_key(|f| f.left);
        rv
    }

    fn tail_is_tail<T>(tail: &[Index], list: &CursorList<T>) {
        for &t in tail {
            assert!(list.next(t).is_none())
        }
    }

    pub(super) fn extract_ancestry(node: Node, graph: &Graph) -> Vec<AncestrySegment> {
        tail_is_tail(&graph.ancestry_tail, &graph.ancestry);
        test_utils::extract(
            node.as_index(),
            &graph.ancestry_head,
            &graph.ancestry_tail,
            &graph.ancestry,
        )
        .into_iter()
        .map(|(a, _)| a)
        .collect::<Vec<_>>()
    }

    pub(super) fn extract_edges(node: Node, graph: &Graph) -> Vec<Edge> {
        tail_is_tail(&graph.edge_tail, &graph.edges);
        test_utils::extract(
            node.as_index(),
            &graph.edge_head,
            &graph.edge_tail,
            &graph.edges,
        )
        .into_iter()
        .map(|(e, _)| e)
        .collect::<Vec<_>>()
    }

    fn initialize_list<T, I, F>(
        input: Vec<Vec<I>>,
        f: F,
        head: &mut [Index],
        tail: &mut [Index],
        list: &mut CursorList<T>,
    ) where
        F: Fn(I) -> T,
        I: Copy,
        T: std::fmt::Debug,
        I: std::fmt::Debug,
    {
        for (i, e) in input.iter().enumerate() {
            assert!(head[i].is_sentinel());
            assert!(tail[i].is_sentinel());
            if !e.is_empty() {
                let mut index = list.new_index(f(e[0]));
                head[i] = index;
                for &j in &e[1..] {
                    index = list.insert_after(index, f(j))
                }
                tail[i] = index;
            }
        }
    }

    pub(super) fn setup_graph(
        num_nodes: usize,
        genome_length: i64,
        num_births: usize,
        initial_birth_times: Vec<i64>,
        // left, right, child
        initial_edges: Vec<Vec<(i64, i64, usize)>>,
        // left, right, parent, mapped_node
        initial_ancestry: Vec<Vec<(i64, i64, Option<usize>, usize)>>,
        // left, right, parent, child
        transmissions: Vec<(i64, i64, usize, usize)>,
    ) -> (Graph, Vec<Node>) {
        let max_time = *initial_birth_times.iter().max().unwrap();
        //assert_eq!(initial_birth_times.len(), initial_edges.len());
        //assert_eq!(initial_birth_times.len(), initial_ancestry.len());

        let mut graph = Graph::with_initial_nodes(num_nodes, genome_length)
            .unwrap()
            .0;

        graph.birth_time = initial_birth_times;

        initialize_list(
            initial_edges,
            |e: (i64, i64, usize)| Edge {
                left: e.0,
                right: e.1,
                child: Node(e.2),
            },
            &mut graph.edge_head,
            &mut graph.edge_tail,
            &mut graph.edges,
        );

        // HACK
        if !initial_ancestry.is_empty() {
            graph.ancestry_head.fill(Index::sentinel());
            graph.ancestry_tail.fill(Index::sentinel());
            graph.ancestry.next.fill(usize::MAX);
            graph.ancestry.data.clear();
            initialize_list(
                initial_ancestry,
                |e: (i64, i64, Option<usize>, usize)| {
                    let parent = e.2.map(Node);
                    AncestrySegment {
                        left: e.0,
                        right: e.1,
                        parent,
                        mapped_node: Node(e.3),
                    }
                },
                &mut graph.ancestry_head,
                &mut graph.ancestry_tail,
                &mut graph.ancestry,
            );
        }

        graph.advance_time_by(max_time + 1);
        let mut birth_nodes = vec![];
        for _ in 0..num_births {
            birth_nodes.push(graph.add_birth(graph.current_time).unwrap());
        }

        for t in transmissions {
            graph
                .record_transmission(t.0, t.1, Node(t.2), birth_nodes[t.3])
                .unwrap();
        }

        (graph, birth_nodes)
    }

    pub(super) fn validate_edges(node: usize, expected: &[(i64, i64, usize)], graph: &Graph) {
        let edges = extract_edges(Node(node), graph);
        assert_eq!(
            edges.len(),
            expected.len(),
            "unexpected number of edges for node {node}: {} != {}",
            edges.len(),
            expected.len()
        );
        for e in expected {
            let edge = Edge {
                left: e.0,
                right: e.1,
                child: Node(e.2),
            };
            assert!(
                edges.contains(&edge),
                "edges of node {node}: {edge:?} not in {edges:?}"
            );
        }
    }

    pub(super) fn validate_ancestry(
        node: usize,
        expected: &[(i64, i64, Option<usize>, usize)],
        graph: &Graph,
    ) {
        let ancestry = extract_ancestry(Node(node), graph);
        assert_eq!(
            ancestry.len(),
            expected.len(),
            "Node({node}): {ancestry:?} != {expected:?}"
        );
        for e in expected {
            let parent = e.2.map(Node);
            let seg = AncestrySegment {
                left: e.0,
                right: e.1,
                parent,
                mapped_node: Node(e.3),
            };
            assert!(
                ancestry.contains(&seg),
                "ancestry of node {node}: {seg:?} not in {ancestry:?}"
            );
        }
    }
}

#[cfg(test)]
mod graph_tests {
    use super::{test_utils::validate_ancestry, *};

    #[test]
    fn with_initial_nodes() {
        let g = Graph::with_initial_nodes(10, 10).unwrap().0;
        assert_eq!(g.edge_head.len(), 10);
        assert_eq!(g.edge_tail.len(), 10);
        assert_eq!(g.ancestry_head.len(), 10);
        assert_eq!(g.ancestry_tail.len(), 10);
        assert_eq!(g.birth_time.len(), 10);
        assert_eq!(g.node_status.len(), 10);
    }

    #[test]
    fn add_birth() {
        let mut g = Graph::with_initial_nodes(10, 10).unwrap().0;
        assert_eq!(g.ancestry_head.len(), 10);
        assert_eq!(g.ancestry_tail.len(), 10);
        assert_eq!(g.birth_time.len(), 10);
        assert_eq!(g.node_status.len(), 10);
        g.advance_time().unwrap();
        let _ = g.add_birth(1).unwrap();
        assert_eq!(g.ancestry_head.len(), 11);
        assert_eq!(g.ancestry_tail.len(), 11);
        assert_eq!(g.birth_time.len(), 11);
        assert_eq!(g.node_status.len(), 11);
    }

    #[test]
    fn record_transmission() {
        let mut g = Graph::with_initial_nodes(10, 10).unwrap().0;
        g.advance_time().unwrap();
        let birth = g.add_birth(1).unwrap();
        assert!(g.record_transmission(0, 5, Node(0), birth).is_ok());
        assert!(g.record_transmission(5, 10, Node(1), birth).is_ok());
        assert_eq!(g.node_heap.len(), 2);
    }

    #[test]
    fn record_invalid_transmission_with_gap() {
        let mut g = Graph::with_initial_nodes(10, 10).unwrap().0;
        g.advance_time().unwrap();
        let birth = g.add_birth(1).unwrap();
        assert!(g.record_transmission(0, 5, Node(0), birth).is_ok());
        assert!(g.record_transmission(6, 10, Node(0), birth).is_err());
        assert_eq!(g.node_heap.len(), 1);
    }

    #[test]
    fn record_invalid_transmission_with_overlap() {
        let mut g = Graph::with_initial_nodes(10, 10).unwrap().0;
        g.advance_time().unwrap();
        let birth = g.add_birth(1).unwrap();
        assert!(g.record_transmission(0, 5, Node(0), birth).is_ok());
        assert!(g.record_transmission(4, 10, Node(1), birth).is_err());
        // 1 b/c the previous recording Err'd
        assert_eq!(g.node_heap.len(), 1);
    }

    #[test]
    fn ancestry_intersection_test0() {
        let mut g = Graph::with_initial_nodes(10, 10).unwrap().0;
        g.advance_time().unwrap();
        let birth = g.add_birth(1).unwrap();
        assert!(g.record_transmission(0, 5, Node(0), birth).is_ok());
        assert!(g.record_transmission(5, 10, Node(1), birth).is_ok());
        for n in [0, 1] {
            let mut queue = vec![];
            ancestry_intersection(Node(n), &g, &mut queue);
            // Have to be careful re: the sentinel value
            assert_eq!(queue.len(), 2);
            // NOTE: we cannot compare the child ancestry index
            // here b/c our naive method generates it incorrectly,
            // b/c it pulls from another set of data types.
            for (i, j) in queue[0..1]
                .iter()
                .zip(test_utils::naive_ancestry_intersection(Node(n), &g).iter())
            {
                assert_eq!(i.left, j.left);
                assert_eq!(i.right, j.right);
                assert_eq!(i.mapped_node, j.mapped_node);
            }
        }
    }

    #[test]
    fn ancestry_intersection_test1() {
        let mut g = Graph::with_initial_nodes(1, 2).unwrap().0;
        g.advance_time().unwrap();
        let birth0 = g.add_birth(1).unwrap();
        let birth1 = g.add_birth(1).unwrap();
        assert!(g.record_transmission(0, 1, Node(0), birth0).is_ok());
        assert!(g.record_transmission(1, 2, Node(0), birth1).is_ok());
        let mut queue = vec![];
        ancestry_intersection(Node(0), &g, &mut queue);
        let mut num_iters = 0;
        let mut overlapper = AncestryOverlapper::new(Node(0), &queue);

        while overlapper.calculate_next_overlap_set().is_some() {
            num_iters += 1;
            if num_iters > 2 {
                panic!("there are only 2 overlaps")
            }
        }
        assert_eq!(num_iters, 2);
    }

    #[test]
    fn ancestry_intersection_test2() {
        let v = vec![
            super::AncestryIntersection {
                left: 0,
                right: 73,
                mapped_node: super::Node(3),
                child: super::Node(3),
                child_ancestry_segment: super::Index(4),
                coalescent: false,
            },
            super::AncestryIntersection {
                left: 45,
                right: 100,
                mapped_node: super::Node(2),
                child: super::Node(2),
                child_ancestry_segment: super::Index(3),
                coalescent: false,
            },
            super::AncestryIntersection {
                left: 9223372036854775807,
                right: 9223372036854775807,
                mapped_node: super::Node(18446744073709551615),
                child: super::Node(18446744073709551615),
                child_ancestry_segment: super::Index(18446744073709551615),
                coalescent: false,
            },
        ];

        let mut overlapper = super::AncestryOverlapper::new(super::Node(0), &v);
        let mut n = 0;
        while let Some((l, r, o)) = overlapper.calculate_next_overlap_set() {
            println!("v = {l} {r} {o:?}");
            n += 1;
        }
        assert_eq!(n, 3);
    }

    #[test]
    fn test_node_recycling() {
        let initial_ancestry = vec![vec![(0, 1, None, 0)]];
        let birth_times = vec![0];
        let (mut graph, _) =
            test_utils::setup_graph(1, 1, 0, birth_times, vec![], initial_ancestry, vec![]);
        validate_ancestry(0, &[(0, 1, None, 0)], &graph);
        graph.free_nodes.push(0);
        let node = graph.add_node(NodeStatus::Death, 0);
        assert_eq!(node.as_index(), 0);
        validate_ancestry(node.as_index(), &[], &graph);
        assert_eq!(graph.ancestry.free_list.len(), 1);
    }
}

#[cfg(test)]
mod propagation_tests {
    use super::*;
    use test_utils::*;

    #[test]
    fn propagation_test0() {
        let birth_times = vec![0; 10];
        let transmissions = vec![(0, 5, 0, 0), (5, 10, 1, 0)];
        let (mut graph, birth_nodes) =
            setup_graph(10, 10, 1, birth_times, vec![], vec![], transmissions);
        let _ = propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
        let anc = extract_ancestry(Node(1), &graph);
        assert_eq!(anc.len(), 1);
        assert!(anc.contains(&AncestrySegment {
            left: 5,
            right: 10,
            parent: None,
            mapped_node: birth_nodes[0]
        }));
        let anc = extract_ancestry(Node(0), &graph);
        assert_eq!(anc.len(), 1);
        assert!(anc.contains(&AncestrySegment {
            left: 0,
            right: 5,
            parent: None,
            mapped_node: birth_nodes[0]
        }));

        for node in [0, 1] {
            let edges = extract_edges(Node(node), &graph);
            assert!(edges.is_empty());
        }
    }

    #[test]
    fn propagation_test1() {
        let initial_birth_times = vec![0; 3];
        let num_births = 4;
        let transmissions = vec![
            (0, 5, 0, 0),
            (5, 10, 1, 0),
            (0, 5, 0, 1),
            (5, 10, 2, 1),
            (0, 10, 0, 2),
            (0, 10, 0, 3),
        ];
        let (mut graph, birth_nodes) = setup_graph(
            3,
            10,
            num_births,
            initial_birth_times,
            vec![],
            vec![],
            transmissions,
        );
        let _ = propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
        for (node, b) in [(1, birth_nodes[0]), (2, birth_nodes[1])] {
            let anc = extract_ancestry(Node(node), &graph);
            assert_eq!(anc.len(), 1);
            assert!(anc.contains(&AncestrySegment {
                left: 5,
                right: 10,
                parent: None,
                mapped_node: b
            }));
        }
        for node in [1, 2] {
            let edges = extract_edges(Node(node), &graph);
            assert!(edges.is_empty());
        }
        let anc = extract_ancestry(Node(0), &graph);
        assert_eq!(anc.len(), 2);
        assert!(anc.contains(&AncestrySegment {
            left: 0,
            right: 5,
            parent: None,
            mapped_node: Node(0)
        }));
        assert!(anc.contains(&AncestrySegment {
            left: 5,
            right: 10,
            parent: None,
            mapped_node: Node(0)
        }));
        let edges = extract_edges(Node(0), &graph);
        assert_eq!(edges.len(), 6);
        for &node in &birth_nodes {
            assert!(edges.contains(&Edge {
                left: 0,
                right: 5,
                child: node
            }));
        }
        // NOTE: will fail once we squash
        for node in [2, 3] {
            assert!(edges.contains(&Edge {
                left: 5,
                right: 10,
                child: birth_nodes[node]
            }));
        }
    }

    // Tree 1:
    //
    //  0
    // ---
    // | |
    // 1 3
    //
    // Tree 2:
    //
    //  0
    //  |
    //  |
    //  2
    //
    // Tree 3:
    //
    //  0
    // ---
    // | |
    // 1 3
    #[test]
    fn propagation_test2() {
        let initial_birth_times = vec![0, 1, 1, 1];
        let num_births = 0;
        let transmissions = vec![];
        let initial_edges = vec![
            vec![(0, 1, 1), (0, 1, 3), (1, 2, 2), (2, 3, 1), (2, 3, 3)],
            vec![],
            vec![],
            vec![],
        ];
        let initial_ancestry = vec![
            vec![(0, 1, None, 0), (1, 2, None, 2), (2, 3, None, 0)],
            vec![(0, 1, Some(0), 1), (2, 3, Some(0), 1)],
            vec![],
            vec![(0, 1, Some(0), 3), (2, 3, Some(0), 3)],
        ];
        let (mut graph, birth_nodes) = setup_graph(
            initial_birth_times.len(),
            3,
            num_births,
            initial_birth_times,
            initial_edges,
            initial_ancestry,
            transmissions,
        );
        graph.node_heap.insert(Node(0), graph.birth_time[0]);
        validate_ancestry(
            0,
            &[(0, 1, None, 0), (1, 2, None, 2), (2, 3, None, 0)],
            &graph,
        );
        let _ = propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
        validate_ancestry(0, &[(0, 1, None, 0), (2, 3, None, 0)], &graph);
        validate_edges(0, &[(0, 1, 1), (2, 3, 1), (0, 1, 3), (2, 3, 3)], &graph);
    }

    #[test]
    fn propagation_test3() {
        let initial_birth_times = vec![0, 0];
        let num_births = 2;
        let transmissions = vec![
            (0, 45, 1, 0),
            (45, 100, 0, 0),
            (0, 73, 0, 1),
            (73, 100, 1, 1),
        ];
        let initial_edges = vec![vec![], vec![]];
        let initial_ancestry = vec![vec![(0, 100, None, 0)], vec![(0, 100, None, 1)]];
        let (mut graph, birth_nodes) = setup_graph(
            initial_birth_times.len(),
            100,
            num_births,
            initial_birth_times,
            initial_edges,
            initial_ancestry,
            transmissions,
        );
        super::test_utils::validate_ancestry(
            2,
            &[(0, 45, Some(1), 2), (45, 100, Some(0), 2)],
            &graph,
        );
        super::test_utils::validate_ancestry(
            3,
            &[(0, 73, Some(0), 3), (73, 100, Some(1), 3)],
            &graph,
        );
        super::propagate_ancestry_changes(super::PropagationOptions::default(), &mut graph);

        // validate parent status
        super::test_utils::validate_ancestry(
            0,
            &[(0, 45, None, 3), (45, 73, None, 0), (73, 100, None, 2)],
            &graph,
        );
        super::test_utils::validate_edges(0, &[(45, 73, 2), (45, 73, 3)], &graph);
        super::test_utils::validate_ancestry(1, &[(0, 45, None, 2), (73, 100, None, 3)], &graph);
        super::test_utils::validate_edges(1, &[], &graph);
        assert!(graph.free_nodes.contains(&1));

        // validate child status

        super::test_utils::validate_ancestry(2, &[(0, 45, None, 2), (45, 100, Some(0), 2)], &graph);
        super::test_utils::validate_ancestry(3, &[(0, 73, Some(0), 3), (73, 100, None, 3)], &graph);

        assert_eq!(graph.free_nodes.len(), 1);

        assert_eq!(
            graph
                .birth_time
                .iter()
                .enumerate()
                .filter(|(i, _)| { !graph.ancestry_head[*i].is_sentinel() })
                .count(),
            graph.birth_time.len()
        );
        assert_eq!(
            graph
                .birth_time
                .iter()
                .enumerate()
                .filter(|(i, _)| { !graph.edge_head[*i].is_sentinel() })
                .count(),
            1
        );
        assert_eq!(graph.edges.free_list.len(), 2);
    }
}

#[cfg(test)]
mod multistep_tests {
    use super::*;
    use test_utils::*;

    //     0
    //   -----
    //   |   |
    //   |   1
    //   |  ---
    //   4  2 3
    //
    // 2-4 will be births
    #[test]
    fn test0() {
        let initial_edges = vec![vec![(0, 2, 1)], vec![]];
        let initial_ancestry = vec![vec![(0, 2, None, 1)], vec![(0, 2, Some(0), 1)]];
        let initial_birth_times = vec![0, 1];
        let num_births = 3;
        let transmissions = vec![(0, 2, 0, 2), (0, 2, 1, 0), (0, 2, 1, 1)];
        let (mut graph, birth_nodes) = setup_graph(
            3,
            2,
            num_births,
            initial_birth_times,
            initial_edges,
            initial_ancestry,
            transmissions,
        );
        let edges = extract_edges(Node(0), &graph);
        let ancestry = extract_ancestry(Node(1), &graph);
        let _ = propagate_ancestry_changes(PropagationOptions::default(), &mut graph);

        // node 1
        validate_edges(1, &[(0, 2, 2), (0, 2, 3)], &graph);
        validate_ancestry(1, &[(0, 2, Some(0), 1)], &graph);

        // node 0
        validate_edges(0, &[(0, 2, 1), (0, 2, 4)], &graph);
        validate_ancestry(0, &[(0, 2, None, 0)], &graph);
    }

    //     0
    //   -----
    //   |   |
    //   |   1
    //   |  ---
    //   4  2 3
    //
    // 3 will "die", resulting in (0,(2,4)) being the remaining topo.
    #[test]
    fn test1() {
        let initial_edges = vec![
            vec![(0, 2, 1), (0, 2, 4)],
            vec![(0, 2, 2), (0, 2, 3)],
            vec![],
            vec![],
            vec![],
        ];
        let initial_ancestry = vec![
            vec![(0, 2, None, 0)],
            vec![(0, 2, Some(0), 1)],
            vec![(0, 2, Some(1), 2)],
            vec![(0, 2, Some(1), 3)],
            vec![(0, 2, Some(0), 4)],
        ];
        let initial_birth_times = vec![0, 1, 2, 2, 2];
        let num_births = 0;
        let transmissions = vec![];
        let (mut graph, _) = setup_graph(
            5,
            2,
            num_births,
            initial_birth_times,
            initial_edges,
            initial_ancestry,
            transmissions,
        );
        // NOTE: This is an API limitation
        graph.ancestry.eliminate(graph.ancestry_head[3]);
        graph.ancestry_head[3] = Index::sentinel();
        graph.ancestry_tail[3] = Index::sentinel();
        // By "killing" node 3, we must enter its parents
        // into the queue
        graph.node_heap.insert(Node(1), graph.birth_time[1]);
        assert!(extract_ancestry(Node(3), &graph).is_empty());
        let last_node = propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
        assert_eq!(last_node, Some(Node(0)));

        // node 1
        validate_edges(1, &[], &graph);
        validate_ancestry(1, &[(0, 2, Some(0), 2)], &graph);

        // node 0
        validate_ancestry(0, &[(0, 2, None, 0)], &graph);
        validate_edges(0, &[(0, 2, 2), (0, 2, 4)], &graph);

        for node in [2, 4] {
            validate_ancestry(node, &[(0, 2, Some(0), node)], &graph)
        }
    }

    // Fundamentally the same as the previous test but with an extra level
    // of unary stuff.
    //
    //     0
    //   -----
    //   |   |
    //   |   1
    //   |   |
    //   |   2
    //   |   |
    //   |  ---
    //   5  3 4
    //
    // 3 will "die", resulting in (0,(5,4)) being the remaining topo.
    #[test]
    fn test2() {
        let initial_edges = vec![
            vec![(0, 2, 1), (0, 2, 5)],
            vec![(0, 2, 2)],
            vec![(0, 2, 3), (0, 2, 4)],
            vec![],
            vec![],
            vec![],
        ];
        let initial_ancestry = vec![
            vec![(0, 2, None, 0)],
            vec![(0, 2, Some(0), 1)],
            vec![(0, 2, Some(1), 2)],
            vec![(0, 2, Some(2), 3)],
            vec![(0, 2, Some(2), 4)],
            vec![(0, 2, Some(0), 5)],
        ];
        let initial_birth_times = vec![0, 1, 2, 3, 3, 3];
        let num_births = 0;
        let transmissions = vec![];
        let (mut graph, _) = setup_graph(
            6,
            2,
            num_births,
            initial_birth_times,
            initial_edges,
            initial_ancestry,
            transmissions,
        );
        // NOTE: This is an API limitation
        graph.ancestry.eliminate(graph.ancestry_head[3]);
        graph.ancestry_head[3] = Index::sentinel();
        graph.ancestry_tail[3] = Index::sentinel();
        assert!(extract_ancestry(Node(3), &graph).is_empty());
        // By "killing" node 3, we must enter its parents
        // into the queue
        graph.node_heap.insert(Node(2), graph.birth_time[2]);
        assert!(extract_ancestry(Node(3), &graph).is_empty());
        let last_node = propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
        assert_eq!(last_node, Some(Node(0)));

        // node 1
        validate_edges(1, &[], &graph);
        validate_ancestry(1, &[(0, 2, Some(0), 4)], &graph);

        // node 0
        validate_ancestry(0, &[(0, 2, None, 0)], &graph);
        validate_edges(0, &[(0, 2, 5), (0, 2, 4)], &graph);

        // Node 2
        // This expected result is purely an internal detail:
        // 2 is unary, and we will have no overlaps to it ancestral to 1,
        // so its parent field stops updating
        validate_ancestry(2, &[(0, 2, None, 4)], &graph);

        for node in [4, 5] {
            validate_ancestry(node, &[(0, 2, Some(0), node)], &graph)
        }
    }
    // Tree 0, span [0,1)
    //    0
    //  -----
    //  1   |
    //  |   2
    // ---  |
    // 3 4  5
    //
    // Tree 1, span [1,2)
    //    0
    //  -----
    //  1   |
    //  |   2
    //  |  ---
    //  5  3 4
    //
    // Node 5 loses all ancestry
    //
    // Node 0 will be unary on both trees, having no edges
    #[test]
    fn test3() {
        let initial_edges = vec![
            vec![(0, 2, 1), (0, 2, 2)],
            vec![(0, 1, 3), (0, 1, 4), (1, 2, 5)],
            vec![(1, 2, 3), (1, 2, 4), (0, 1, 5)],
            vec![],
            vec![],
            vec![],
        ];

        let initial_ancestry = vec![
            vec![(0, 2, None, 0)],
            vec![(0, 2, Some(0), 1)],
            vec![(0, 2, Some(0), 2)],
            // Births
            vec![(0, 1, Some(1), 3), (1, 2, Some(2), 3)],
            vec![(0, 1, Some(1), 4), (1, 2, Some(2), 4)],
            // Loss of ancestry for node 5
            vec![],
        ];
        let initial_birth_times = vec![0, 1, 2, 3, 3, 3];
        let num_births = 0;
        let transmissions = vec![];
        let (mut graph, _) = setup_graph(
            6,
            2,
            num_births,
            initial_birth_times,
            initial_edges,
            initial_ancestry,
            transmissions,
        );
        graph.node_heap.insert(Node(2), graph.birth_time[2]);
        graph.node_heap.insert(Node(1), graph.birth_time[1]);
        assert!(extract_ancestry(Node(5), &graph).is_empty());
        let last_node = propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
        assert_eq!(last_node, Some(Node(0)));

        // Node 1
        validate_ancestry(1, &[(0, 1, None, 1)], &graph);

        // Node 2
        validate_ancestry(2, &[(1, 2, None, 2)], &graph);

        // Node 3
        validate_ancestry(3, &[(0, 1, Some(1), 3), (1, 2, Some(2), 3)], &graph);

        // Node 4
        validate_ancestry(4, &[(0, 1, Some(1), 4), (1, 2, Some(2), 4)], &graph);

        // Node 0
        validate_ancestry(0, &[(0, 1, None, 1), (1, 2, None, 2)], &graph);

        // Node 0
        validate_edges(0, &[], &graph);

        // Node 1
        validate_edges(1, &[(0, 1, 3), (0, 1, 4)], &graph);

        // Node 2
        validate_edges(2, &[(1, 2, 3), (1, 2, 4)], &graph);
    }

    //     0
    //    ---
    //    | |
    //    1 2
    //
    //  Nodes 1 and 2 lose all ancestry, leaving 0 with no overlaps.
    #[test]
    fn test4() {
        let initial_edges = vec![vec![(0, 2, 1), (0, 2, 2)]];

        let initial_ancestry = vec![
            vec![(0, 2, None, 0)],
            // Losses
            vec![],
            vec![],
        ];
        let initial_birth_times = vec![0, 1, 1];
        let num_births = 0;
        let transmissions = vec![];
        let (mut graph, _) = setup_graph(
            3,
            2,
            num_births,
            initial_birth_times,
            initial_edges,
            initial_ancestry,
            transmissions,
        );
        graph.node_heap.insert(Node(0), graph.birth_time[0]);
        let last_node = propagate_ancestry_changes(PropagationOptions::default(), &mut graph);

        validate_edges(0, &[], &graph);
        validate_ancestry(0, &[], &graph);

        assert_eq!(graph.edges.free_list.len(), 2);
        assert_eq!(graph.ancestry.free_list.len(), 1);
    }

    // Tree 1 on [0,1):
    //
    //    0
    //   ---
    //   1 |
    //   | 2
    // --- ---
    // 3 4 5 6
    //
    // Tree 2 on [2,3)
    //    0
    //   ---
    //   1 |
    //   | 2
    // --- ---
    // 5 6 3 4
    //
    // Nodes 5,6 lose all ancestry, propagating
    // a state of no overlap to some parental nodes.
    #[test]
    fn test5() {
        let initial_edges = vec![
            vec![(0, 1, 1), (0, 1, 2), (2, 3, 1), (2, 3, 2)],
            vec![(0, 1, 3), (0, 1, 4), (2, 3, 5), (2, 3, 6)],
            vec![(2, 3, 3), (2, 3, 4), (0, 1, 5), (0, 1, 6)],
            vec![],
            vec![],
            vec![],
            vec![],
        ];

        let initial_ancestry = vec![
            vec![(0, 1, None, 0), (2, 3, None, 0)],
            vec![(0, 1, Some(0), 1), (2, 3, Some(0), 1)],
            vec![(0, 1, Some(0), 2), (2, 3, Some(0), 2)],
            vec![(0, 1, Some(1), 3), (2, 3, Some(2), 3)],
            vec![(0, 1, Some(1), 4), (2, 3, Some(2), 4)],
            vec![],
            vec![],
        ];
        let initial_birth_times = vec![0, 1, 2, 3, 3, 3, 6];
        let num_births = 0;
        let transmissions = vec![];
        let (mut graph, _) = setup_graph(
            initial_birth_times.len(),
            3,
            num_births,
            initial_birth_times,
            initial_edges,
            initial_ancestry,
            transmissions,
        );
        for node in [1, 2] {
            graph.node_heap.insert(Node(node), graph.birth_time[node]);
        }
        let last_node = propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
        assert_eq!(last_node, Some(Node(0)));

        validate_edges(0, &[], &graph);
        validate_ancestry(0, &[(0, 1, None, 1), (2, 3, None, 2)], &graph);

        validate_edges(1, &[(0, 1, 3), (0, 1, 4)], &graph);
        // TODO: is this None behavior consistent/desired?
        validate_ancestry(1, &[(0, 1, None, 1)], &graph);

        validate_edges(2, &[(2, 3, 3), (2, 3, 4)], &graph);
        // TODO: is this None behavior consistent/desired?
        validate_ancestry(2, &[(2, 3, None, 2)], &graph);

        validate_ancestry(3, &[(2, 3, Some(2), 3), (0, 1, Some(1), 3)], &graph);
        validate_ancestry(4, &[(2, 3, Some(2), 4), (0, 1, Some(1), 4)], &graph);
    }
}
