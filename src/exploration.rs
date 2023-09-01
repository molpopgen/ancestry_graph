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

    pub fn new_index(&mut self, datum: T) -> Index {
        if let Some(index) = self.free_list.pop() {
            let _ = std::mem::replace(&mut self.data[index], datum);
            println!("recycling {index}");
            Index(index)
        } else {
            self.next.push(Index::sentinel().0);
            self.data.push(datum);
            Index(self.data.len() - 1)
        }
    }

    fn setup_insertion(&mut self, at: Index, datum: T) -> (Index, Index) {
        let new_index = self.new_index(datum);
        (new_index, at)
    }

    fn finalize_insertion(&mut self, next: Index, next_value: Index) {
        self.set_next(next, next_value);
    }

    fn set_next(&mut self, at: Index, value: Index) {
        self.next[at.0] = value.0
    }

    pub fn add_list(&mut self, datum: T) -> Index {
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

    pub fn insert_after(&mut self, at: Index, datum: T) -> Index {
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
        assert!(from.0 < self.next.len());
        let mut next = self.excise_next(from);
        while !next.is_sentinel() {
            println!("next = {next:?}");
            next = self.excise_next(next);
        }
        self.next[from.0] = usize::MAX;
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
) {
    let current_tail = tail[at];
    if current_tail.is_sentinel() {
        let new_head = list.new_index(datum);
        head[at] = new_head;
        tail[at] = new_head;
    } else {
        let new_tail = list.insert_after(current_tail, datum);
        tail[at] = new_tail;
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
    ancestry_segment: Index,
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

    fn calculate_next_overlap_set(&mut self) -> Option<Overlaps<'_>> {
        // NOTE: this if statement hides from the compiler
        // that current_overlap is always < queue.len().
        // We should be able to check current_overlap + 1 <
        // queue.len() and have the later bounds check optimmized out.
        if self.current_overlap < self.num_overlaps {
            self.overlaps.retain(|o| o.right > self.left);
            if self.overlaps.is_empty() {
                self.left = self.queue[self.current_overlap].left;
            }
            let mut new_right = self.right;
            for segment in &self.queue[self.current_overlap..] {
                if segment.left == self.left {
                    self.current_overlap += 1;
                    new_right = std::cmp::min(new_right, segment.right);
                    self.overlaps.push(*segment)
                } else {
                    break;
                }
            }
            // NOTE: we can track the left value while
            // traversing the overlaps, setting it to MAX
            // initially, and dodge another bounds check
            self.right = new_right;
            self.right = std::cmp::min(self.right, self.queue[self.current_overlap].left);
            Some(Overlaps {
                left: self.left,
                right: self.right,
                overlaps: &self.overlaps,
            })
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
                Some(Overlaps {
                    left: self.left,
                    right: self.right,
                    overlaps: &self.overlaps,
                })
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
    overlaps: &'overlapper [AncestryIntersection],
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
    cached_extant_nodes: Vec<Node>,

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
        let cached_extant_nodes = vec![];
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
            cached_extant_nodes,
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
                todo!("need to free up pre-existing ancestry");
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
        self.cached_extant_nodes.push(rv);
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
    assert!(!graph.edge_head[node.as_index()].is_sentinel());
    let mut current_edge = Some(graph.edge_head[node.as_index()]);
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
        println!("{edge_ref:?}, {child_ancestry:?}");
        while let Some(child_ancestry_index) = child_ancestry {
            let anc_ref = graph.ancestry.get(child_ancestry_index);
            println!("{anc_ref:?}");
            if edge_ref.overlaps(anc_ref) {
                let left = std::cmp::max(edge_ref.left, anc_ref.left);
                let right = std::cmp::min(edge_ref.right, anc_ref.right);
                println!("overlap!!!");
                queue.push(AncestryIntersection {
                    left,
                    right,
                    mapped_node: anc_ref.mapped_node,
                    ancestry_segment: child_ancestry_index,
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
            ancestry_segment: Index::sentinel(),
        });
    }
}

fn update_ancestry(
    left: i64,
    right: i64,
    mapped_node: Node,
    current_ancestry_index: Index,
    child_ancestry_segment_index: Index,
    birth_time: &[i64],
    ancestry: &mut NodeAncestry,
    node_heap: &mut NodeHeap,
) -> Index {
    let mut seg_right = None;
    let current_ancestry_index = current_ancestry_index;
    let current = *ancestry.get(current_ancestry_index);
    let (current_left, current_right) = {
        let current = ancestry.get(current_ancestry_index);
        (current.left, current.right)
    };
    let temp_left = std::cmp::max(current_left, left);
    let temp_right = std::cmp::min(current_right, right);
    println!("{:?} {temp_left} {temp_right}", current_ancestry_index);
    let mut rv = ancestry.next_raw(current_ancestry_index);
    if current_left != temp_left {
        assert!(current_left < temp_left);
        println!("we have a left dangle on {current_left}, {temp_left}");
        if let Some(parent) = current.parent {
            node_heap.insert(parent, birth_time[parent.as_index()]);
        }
    }
    if current_right != temp_right {
        if let Some(parent) = current.parent {
            node_heap.insert(parent, birth_time[parent.as_index()]);
        }
        println!("right dangle: {current_right:?}, {temp_right}");
        {
            let current = ancestry.get_mut(current_ancestry_index);
            current.left = temp_right;
        }
        seg_right = Some(AncestrySegment {
            left: temp_right,
            right: current_right,
            parent: None, // FIXME
            mapped_node: ancestry.get(current_ancestry_index).mapped_node,
        });
    }

    //if mapped_node != current.mapped_node {
    //    // This is a change from "coalescent" to "unary" on this
    //    // segment, which is a change we must propagate.
    //    if let Some(parent) = current.parent {
    //        node_heap.insert(parent, birth_time[parent.as_index()]);
    //        ancestry.data[child_ancestry_segment_index.0].parent = current.parent;
    //    }
    //}

    let out_seg = AncestrySegment {
        left,
        right,
        parent: current.parent,
        mapped_node,
    };
    println!("out = {out_seg:?}");
    // TODO: API fn to replace.
    *ancestry.get_mut(current_ancestry_index) = out_seg;
    //if left == current_left && right == current_right {
    //    // perfect overlap, all we need to do is update
    //    // the mapped node...
    //    let current = ancestry.get_mut(current_ancestry_index);
    //    println!(
    //        "update mapping from {:?} to {:?}",
    //        current.mapped_node, out_seg.mapped_node
    //    );
    //    current.mapped_node = out_seg.mapped_node;
    //    // ... but if the mapping has changed, then this
    //    // segment is possibly a CHANGE TO UNARY that we
    //    // must record
    //} else {
    //    *ancestry.get_mut(current_ancestry_index) = out_seg;
    //    // rv = ancestry.next_raw(current_ancestry_index);
    //    //println!("insert out");
    //    //// We insert out_seg at current_ancestry_index
    //    //if current_ancestry_index == last_ancestry_index {
    //    //    println!("case A");
    //    //    // replace current with out_seg and insert the
    //    //    // current value next
    //    //    let current = *ancestry.get(current_ancestry_index);
    //    //    let next = ancestry.next_raw(current_ancestry_index);
    //    //    let new_index = ancestry.new_index(current);
    //    //    println!("new_index = {new_index:?}");
    //    //    ancestry.next[new_index.0] = next.0;
    //    //    let _ = std::mem::replace(&mut ancestry.data[current_ancestry_index.0], out_seg);
    //    //    ancestry.next[current_ancestry_index.0] = new_index.0;
    //    //    // Needed for handling right_seg below
    //    //    current_ancestry_index = new_index;
    //    //    rv = current_ancestry_index;
    //    //    println!("rv = {rv:?}");
    //    //} else {
    //    //    println!("case B");
    //    //    let next = ancestry.next_raw(last_ancestry_index);
    //    //    let new_index = ancestry.new_index(out_seg);
    //    //    ancestry.next[last_ancestry_index.0] = new_index.0;
    //    //    ancestry.next[new_index.0] = next.0;
    //    //    rv = new_index;
    //    //}
    //}

    if let Some(right_seg) = seg_right {
        println!("seg_right = {right_seg:?}");
        //if right_seg.left == current_left && right_seg.right == current_right {
        //    let current = ancestry.get_mut(current_ancestry_index);
        //    println!(
        //        "updating node from {:?} to {:?}",
        //        current.mapped_node, right_seg.mapped_node
        //    );
        //    // Could be an ancestry change!
        //    current.mapped_node = right_seg.mapped_node;
        //} else {
        println!("inserting seg_right");
        let next = ancestry.next_raw(current_ancestry_index);
        let new_index = ancestry.new_index(right_seg);
        println!("new_index = {new_index:?}");
        ancestry.next[current_ancestry_index.0] = new_index.0;
        ancestry.next[new_index.0] = next.0;
        rv = new_index;
        //}
    }
    rv
}

fn process_queued_node(
    options: PropagationOptions,
    queued_parent: Node,
    parent_status: NodeStatus,
    graph: &mut Graph,
    queue: &mut Vec<AncestryIntersection>,
    temp_edges: &mut Vec<Edge>,
) {
    let mut ahead = graph.ancestry_head[queued_parent.as_index()];
    while !ahead.is_sentinel() {
        println!("input {:?}", graph.ancestry.get(ahead));
        ahead = graph.ancestry.next_raw(ahead);
    }
    ancestry_intersection(queued_parent, graph, queue);
    println!("{queued_parent:?} => {queue:?}");
    let mut ahead = graph.ancestry_head[queued_parent.as_index()];
    let mut last_ancestry_index = ahead;

    println!("ahead = {ahead:?}");

    let mut overlapper = AncestryOverlapper::new(queued_parent, queue);
    println!("{overlapper:?}");

    let mut overlaps = overlapper.calculate_next_overlap_set();

    while !ahead.is_sentinel() {
        if let Some(ref current_overlaps) = overlaps {
            println!("current = {:?}", graph.ancestry.get(ahead));
            println!("overlaps = {current_overlaps:?}");
            let (current_left, current_right) = {
                let current = graph.ancestry.get(ahead);
                (current.left, current.right)
            };
            if current_right > current_overlaps.left && current_overlaps.right > current_left {
                let mapped_node;
                let aseg_index;
                if current_overlaps.overlaps.len() == 1 {
                    mapped_node = current_overlaps.overlaps[0].mapped_node;
                    aseg_index = current_overlaps.overlaps[0].ancestry_segment;
                } else {
                    mapped_node = queued_parent;
                    aseg_index = Index::sentinel();
                    for o in current_overlaps.overlaps {
                        println!("child node = {:?}", o.mapped_node);
                        temp_edges.push(Edge {
                            left: current_overlaps.left,
                            right: current_overlaps.right,
                            child: o.mapped_node,
                        });
                        println!("anc seg = {:?}", graph.ancestry.data[o.ancestry_segment.0]);
                        // graph.ancestry.data[o.ancestry_segment.0].parent = Some(queued_parent);
                    }
                }
                last_ancestry_index = ahead;
                ahead = update_ancestry(
                    current_overlaps.left,
                    current_overlaps.right,
                    mapped_node,
                    ahead,
                    aseg_index,
                    &graph.birth_time,
                    &mut graph.ancestry,
                    &mut graph.node_heap,
                );
                overlaps = overlapper.calculate_next_overlap_set();
            } else {
                panic!("no coverage until now!");
                if last_ancestry_index == ahead {
                    println!("gotta shift left");
                    let next = graph.ancestry.next_raw(ahead);
                    if !next.is_sentinel() {
                        graph.ancestry.data.swap(ahead.0, next.0);
                        graph.ancestry.next[ahead.0] = graph.ancestry.next[next.0];
                        graph.ancestry.free_list.push(next.0);
                    }
                    //last_ancestry_index = ahead;
                    println!("free list = {:?}", graph.ancestry.free_list);
                } else {
                    println!("gotta excise the current thing");
                    panic!("no coverage until now!");
                    ahead = graph.ancestry.excise_next(last_ancestry_index);
                    //let next = graph.ancestry.next_raw(ahead);
                    //println!("current = {:?}", graph.ancestry.get(ahead));
                    //println!("prev = {:?}", graph.ancestry.get(last_ancestry_index));
                    //println!("here");
                    //graph.ancestry.next[last_ancestry_index.0] = next.0;
                    //graph.ancestry.free_list.push(ahead.0);
                    //ahead = next;
                    //println!("{ahead:?}");
                }
                //last_ancestry_index = ahead;
                //ahead = graph.ancestry.next_raw(ahead)
            }
        } else {
            break;
        }
        println!("temp_edges = {temp_edges:?}");
    }

    println!(
        "done {ahead:?}, {last_ancestry_index:?}, {:?}",
        graph.ancestry.next_raw(last_ancestry_index)
    );

    if !ahead.is_sentinel() {
        let mut z = graph.ancestry.next(last_ancestry_index);
        // TODO: each of these is a right overhang
        // that we need to reckon with.
        while let Some(index) = z {
            println!("removing trailing segment {:?}", graph.ancestry.get(index));
            z = graph.ancestry.next(index);
            graph.ancestry.next[index.0] = usize::MAX;
            graph.ancestry.free_list.push(index.0);
        }
        graph.ancestry.next[last_ancestry_index.0] = usize::MAX;
    }

    let mut ahead = graph.ancestry_head[queued_parent.as_index()];
    while !ahead.is_sentinel() {
        println!("output {:?}", graph.ancestry.get(ahead));

        if ahead == last_ancestry_index {
            println!("breaking");
            break;
        }

        ahead = graph.ancestry.next_raw(ahead);
    }

    if temp_edges.is_empty() {
        let mut e = graph.edge_head[queued_parent.as_index()];
        while !e.is_sentinel() {
            println!("deleting edge {e:?}");
            let next = graph.edges.next_raw(e);
            graph.edges.next[e.0] = usize::MAX;
            graph.edges.free_list.push(e.0);
            e = next;
        }
        graph.edge_head[queued_parent.as_index()] = Index::sentinel();
        graph.edge_tail[queued_parent.as_index()] = Index::sentinel();
    } else {
        let mut e = graph.edge_head[queued_parent.as_index()];
        let mut last_e = e;
        for edge in temp_edges.iter() {
            println!("adding edge: {edge:?}, {e:?}, {last_e:?}");
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
                graph.edges.free_list.push(index.0);
            }
            graph.edges.next[last_e.0] = usize::MAX;
        }
        graph.edge_tail[queued_parent.as_index()] = last_e;
    }

    graph.ancestry_tail[queued_parent.as_index()] = last_ancestry_index;
    println!("{:?}", graph.ancestry.next_raw(last_ancestry_index));
}

// returns the value of the last (oldest, ignoring ties) Node processed
// The return value has no meaning beyond testing and should eventually
// be deleted.
fn propagate_ancestry_changes(options: PropagationOptions, graph: &mut Graph) -> Option<Node> {
    let mut temp_edges = vec![];

    let mut queue = vec![];
    let mut rv = None;
    while let Some(queued_node) = graph.node_heap.pop() {
        rv = Some(queued_node);
        process_queued_node(
            options,
            queued_node,
            graph.node_status[queued_node.as_index()],
            graph,
            &mut queue,
            &mut temp_edges,
        );
        // Clean up for next loop
        queue.clear();
        temp_edges.clear();
    }

    debug_assert!(graph.node_heap.is_empty());
    rv
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
                        ancestry_segment: Index(i),
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
                println!("added {:?} for node {i}", e[0]);
                let mut index = list.new_index(f(e[0]));
                head[i] = index;
                for &j in &e[1..] {
                    println!("added {:?} for node {i}", j);
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
            println!("{:?}", graph.ancestry);
            println!("{:?}", graph.ancestry_head);
            println!("{:?}", graph.ancestry_tail);
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
        println!("{:?}", graph.node_heap);

        (graph, birth_nodes)
    }

    pub(super) fn validate_edges(node: usize, expected: &[(i64, i64, usize)], graph: &Graph) {
        let edges = extract_edges(Node(node), graph);
        assert_eq!(edges.len(), expected.len());
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
        assert_eq!(ancestry.len(), expected.len());
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
    use super::*;

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
        println!("{birth_nodes:?}");
        for (node, b) in [(1, birth_nodes[0]), (2, birth_nodes[1])] {
            let anc = extract_ancestry(Node(node), &graph);
            assert_eq!(anc.len(), 1);
            println!("{node:?} -> {anc:?}");
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
        println!("input edges for node 0: {:?}", edges);
        let ancestry = extract_ancestry(Node(1), &graph);
        println!("input ancestry for node 1: {:?}", ancestry);
        let _ = propagate_ancestry_changes(PropagationOptions::default(), &mut graph);

        println!("{:?}", graph.ancestry_head);
        println!("{:?}", graph.ancestry);

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
        graph.node_heap.insert(Node(2), graph.birth_time[1]);
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
        validate_ancestry(2, &[(0, 2, Some(0), 4)], &graph);

        for node in [4, 5] {
            validate_ancestry(node, &[(0, 2, Some(0), node)], &graph)
        }
    }
}
