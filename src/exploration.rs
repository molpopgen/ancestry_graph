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
        self.next_raw(at).into_option()
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
}

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
    // NOTE: this is redundant
    // with what we will do during propagation,
    // we the graph should contain information
    // about the queue.
    parents: crate::NodeHash,
    // NOTE: this is redundant with what we
    // need to accomplish.
    deaths: Vec<Node>,

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
        let deaths = vec![];

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
            parents,
            deaths,
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
        self.parents.insert(parent);
        Ok(())
    }

    pub fn mark_node_death(&mut self, node: Node) {
        self.node_status[node.as_index()] = NodeStatus::Death;
        self.deaths.push(node);
    }
}

fn ancestry_intersection(node: Node, graph: &Graph, queue: &mut Vec<AncestryIntersection>) {
    queue.clear();
    assert!(!graph.edge_head[node.as_index()].is_sentinel());
    let mut current_edge = Some(graph.edge_head[node.as_index()]);
    //let mut current_ancestry = Some(graph.ancestry_head[node.as_index()]);

    while let Some(edge_index) = current_edge {
        let edge_ref = graph.edges.get(edge_index);
        // while let Some(aseg) = current_ancestry {
        //     let anc_ref = graph.ancestry.get(aseg);
        //     if anc_ref.overlaps(edge_ref) {
        //         break;
        //     }
        //     current_ancestry = graph.ancestry.next(aseg);
        // }
        let mut child_ancestry = {
            let a = graph.ancestry_head[edge_ref.child.as_index()];
            if a.is_sentinel() {
                None
            } else {
                Some(a)
            }
        };
        while let Some(child_ancestry_index) = child_ancestry {
            let anc_ref = graph.ancestry.get(child_ancestry_index);
            if edge_ref.overlaps(anc_ref) {
                let left = std::cmp::max(edge_ref.left, anc_ref.left);
                let right = std::cmp::min(edge_ref.right, anc_ref.right);
                queue.push(AncestryIntersection {
                    left,
                    right,
                    mapped_node: anc_ref.mapped_node,
                });
            }
            child_ancestry = graph.ancestry.next(child_ancestry_index);
        }
        current_edge = graph.edges.next(edge_index);
    }
    queue.sort_unstable_by_key(|x| x.left);
}

fn update_ancestry(
    left: i64,
    right: i64,
    mapped_node: Node,
    last_ancestry_index: Index,
    current_ancestry_index: Index,
    ancestry: &mut NodeAncestry,
) -> Index {
    let mut seg_right = None;
    let current_ancestry_index = current_ancestry_index;
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
    }
    if current_right != temp_right {
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
    let out_seg = AncestrySegment {
        left,
        right,
        parent: None, // FIXME
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

fn update_ancestry_design(
    node: Node,
    overlaps: &[(i64, i64, Node)],
    ancestry: &mut NodeAncestry,
    ancestry_head: &mut [Index],
    ancestry_tail: &mut [Index],
) {
    assert_eq!(ancestry_head.len(), ancestry_tail.len());
    let mut ahead = ancestry_head[node.as_index()];
    let mut last_ancestry_index = ahead;
    let mut current_overlap = 0_usize;
    let mut last_right = 0;
    while !ahead.is_sentinel() && current_overlap < overlaps.len() {
        let (left, right, mapped_node) = overlaps[current_overlap];
        println!("current input segment = {:?}", ancestry.get(ahead));
        if right > ancestry.get(ahead).left && ancestry.get(ahead).right > left {
            println!(
                "yes {left}, {right}, {:?}, {:?}",
                ancestry.get(ahead).left,
                ancestry.get(ahead).right
            );
            last_ancestry_index = ahead;
            ahead = update_ancestry(
                left,
                right,
                mapped_node,
                last_ancestry_index,
                ahead,
                ancestry,
            );
            println!("updated to {ahead:?} (overlap)");
            current_overlap += 1;
            last_right = right;
        } else {
            println!(
                "no for {ahead:?}: {left}, {right}, {:?} | vs {}, {}",
                ancestry.next(ahead),
                ancestry.get(ahead).left,
                ancestry.get(ahead).right,
            );
            // Goal here is to keep the output head the same.
            // as it was upon input
            if last_ancestry_index == ahead {
                println!("gotta shift left");
                let next = ancestry.next_raw(ahead);
                if !next.is_sentinel() {
                    ancestry.data.swap(ahead.0, next.0);
                    ancestry.next[ahead.0] = ancestry.next[next.0];
                    ancestry.free_list.push(next.0);
                }
                last_ancestry_index = ahead;
                println!("free list = {:?}", ancestry.free_list);
            } else {
                println!("gotta excise the current thing");
                let next = ancestry.next_raw(ahead);
                ancestry.next[last_ancestry_index.0] = next.0;
                ancestry.free_list.push(ahead.0);
                ahead = next;
            }
            println!("updated to {ahead:?} (non overlap)");
        }
    }
    println!("done: {:?} {:?} | {last_right}", last_ancestry_index, ahead,);
    if !ahead.is_sentinel() {
        let mut z = ancestry.next(last_ancestry_index);
        // TODO: each of these is a right overhang
        // that we need to reckon with.
        while let Some(index) = z {
            println!("removing trailing segment {:?}", ancestry.get(index));
            z = ancestry.next(index);
            ancestry.next[index.0] = usize::MAX;
            ancestry.free_list.push(index.0);
        }
        ancestry.next[last_ancestry_index.0] = usize::MAX;
    }
    ancestry_tail[node.0] = last_ancestry_index;
}

fn process_queued_node(
    options: PropagationOptions,
    queued_parent: Node,
    parent_status: NodeStatus,
    graph: &mut Graph,
    queue: &mut Vec<AncestryIntersection>,
) {
    ancestry_intersection(queued_parent, graph, queue);
    if !queue.is_empty() {
        queue.push(AncestryIntersection {
            left: i64::MAX,
            right: i64::MAX,
            mapped_node: Node(usize::MAX),
        });
    }
    println!("{queued_parent:?} => {queue:?}");
    todo!();
}

fn propagate_ancestry_changes(options: PropagationOptions, graph: &mut Graph) -> Option<Node> {
    // 1. Need to build our queue

    // We need some encapsulation here:
    let mut queued_nodes: NodeHash = NodeHash::with_hasher(BuildNoHashHasher::default());
    let mut node_queue: std::collections::BinaryHeap<QueuedNode> =
        std::collections::BinaryHeap::new();
    for node in graph.parents.iter() {
        if !queued_nodes.contains(node) {
            node_queue.push(QueuedNode {
                node: *node,
                birth_time: graph.birth_time[node.as_index()],
            });
            queued_nodes.insert(*node);
        }
    }
    // Repetition here shows why our Graph has the comments
    // about redundant data structures
    for node in graph.deaths.iter() {
        if !queued_nodes.contains(node) {
            node_queue.push(QueuedNode {
                node: *node,
                birth_time: graph.birth_time[node.as_index()],
            });
            queued_nodes.insert(*node);
        }
    }

    let mut queue = vec![];
    while let Some(queued_node) = node_queue.pop() {
        process_queued_node(
            options,
            queued_node.node,
            graph.node_status[queued_node.node.as_index()],
            graph,
            &mut queue,
        )
    }

    // TODO: should be a "cleanup" fn.
    graph.parents.clear();
    graph.deaths.clear();

    todo!()
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

    #[must_use]
    pub(super) fn run_ancestry_tests(
        input_ancestry: &[Vec<(i64, i64, Node)>],
        overlaps: &[(i64, i64, Node)],
    ) -> (NodeAncestry, Vec<Index>, Vec<Index>) {
        let mut ancestry = NodeAncestry::with_capacity(1000);
        let (mut ancestry_head, mut ancestry_tail) =
            test_utils::setup_ancestry(input_ancestry, &mut ancestry);
        for i in 0..ancestry.data.len() {
            println!("{i}: {:?} => {:?}", ancestry.data[i], ancestry.next[i])
        }
        update_ancestry_design(
            Node(0),
            overlaps,
            &mut ancestry,
            &mut ancestry_head,
            &mut ancestry_tail,
        );
        let mut extracted = vec![];
        let mut h = ancestry_head[0];
        while !h.is_sentinel() {
            let a = ancestry.get(h);
            println!("extracted {a:?}");
            extracted.push((a.left, a.right, a.mapped_node));
            let next = ancestry.next_raw(h);
            // Check that our tail is properly updated
            if next.is_sentinel() {
                assert_eq!(ancestry_tail[0], h);
            }
            h = next;
        }
        for i in &extracted {
            assert!(overlaps.contains(i), "{i:?}, {overlaps:?} != {extracted:?}");
        }
        for o in overlaps {
            assert!(
                extracted.contains(o),
                "{o:?}, {extracted:?} != {overlaps:?}"
            );
        }

        (ancestry, ancestry_head, ancestry_tail)
    }

    fn extract<T>(
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
            for a in child_ancestry {
                if a.right > edge.left && edge.right > a.left {
                    let left = std::cmp::max(a.left, edge.left);
                    let right = std::cmp::min(a.right, edge.right);
                    rv.push(AncestryIntersection {
                        left,
                        right,
                        mapped_node: a.mapped_node,
                    });
                }
            }
        }
        rv.sort_unstable_by_key(|f| f.left);
        rv
    }
}

// this is test3 from the python prototype
#[test]
fn test_list_updating_1() {
    let anc0 = vec![(0_i64, 2_i64, Node(0))];
    let anc1 = vec![(1_i64, 2_i64, Node(1))];
    let anc2 = vec![(0_i64, 1_i64, Node(2))];
    let input_ancestry = vec![anc0, anc1, anc2];

    // (left, right, mapped_node)
    // cribbed from manual calculation/the python prototype
    let overlaps = [(0_i64, 1_i64, Node(2)), (1, 2, Node(1))];
    let (ancestry, _, _) = test_utils::run_ancestry_tests(&input_ancestry, &overlaps);
    // FIXME: below should be audited carefully.
    // Adding in "seg_right" is creating new entries.
    //assert_eq!(ancestry.data.len(), 5);
    //assert_eq!(ancestry.next.len(), 5);
}

// this is test0 from the python prototype
#[test]
fn test_list_updating_2() {
    let input_ancestry = vec![
        vec![(0_i64, 3_i64, Node(0))],
        vec![(0_i64, 1_i64, Node(1)), (2_i64, 3_i64, Node(1))],
        vec![(0_i64, 1_i64, Node(2))],
    ];

    // (left, right, mapped_node)
    // cribbed from manual calculation/the python prototype
    let overlaps = [(0_i64, 1_i64, Node(0)), (2, 3, Node(1))];
    let (ancestry, _, _) = test_utils::run_ancestry_tests(&input_ancestry, &overlaps);
    // FIXME: below should be audited carefully.
    // Adding in "seg_right" is creating new entries.
    // assert_eq!(ancestry.data.len(), 6);
    // assert_eq!(ancestry.next.len(), 6);
}

// This is test5b_b_less_contrived from python prototype
#[test]
fn test_list_updating_3() {
    let input_ancestry = vec![
        vec![
            (0_i64, 1_i64, Node(1)),
            (1_i64, 8_i64, Node(0)),
            (8, 16, Node(1)),
        ],
        vec![(2, 8, Node(1)), (12, 14, Node(1))],
        vec![],
        vec![],
    ];
    let overlaps = vec![(2_i64, 8_i64, Node(1)), (12, 14, Node(1))];
    let (ancestry, _, _) = test_utils::run_ancestry_tests(&input_ancestry, &overlaps);
    for i in 0..ancestry.data.len() {
        println!("{i}: {:?} => {:?}", ancestry.data[i], ancestry.next[i])
    }
    println!("{:?}", ancestry.free_list)
}

#[test]
fn test_list_updating_3b() {
    let input_ancestry = vec![
        vec![(1_i64, 8_i64, Node(0)), (8, 14, Node(1)), (14, 16, Node(1))],
        vec![(2, 8, Node(1)), (12, 14, Node(1))],
        vec![],
        vec![],
    ];
    let overlaps = vec![(2_i64, 8_i64, Node(1)), (12, 14, Node(1))];
    let (ancestry, _, _) = test_utils::run_ancestry_tests(&input_ancestry, &overlaps);
    for i in 0..ancestry.data.len() {
        println!("{i}: {:?} => {:?}", ancestry.data[i], ancestry.next[i])
    }
    println!("{:?}", ancestry.free_list)
}

#[test]
fn test_list_updating_4() {
    let input_ancestry = vec![vec![(0_i64, 5_i64, Node(0))], vec![], vec![], vec![]];
    let overlaps = vec![(1_i64, 2_i64, Node(0)), (3, 4, Node(1))];
    let (ancestry, _, _) = test_utils::run_ancestry_tests(&input_ancestry, &overlaps);
    for i in 0..ancestry.data.len() {
        println!("{i}: {:?} => {:?}", ancestry.data[i], ancestry.next[i])
    }
    println!("{:?}", ancestry.free_list)
}

#[cfg(test)]
mod design_property_tests {
    use super::*;
    use proptest::prelude::*;
    use rand::Rng;
    use rand::SeedableRng;

    proptest! {
        #[test]
        fn test_continguous_parent(seed in 0..u64::MAX) {
            let input_ancestry = vec![vec![(0_i64, 500_i64, Node(0))]];
            let next_distance = rand::distributions::Uniform::new(1_i64, 25);
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut last = rng.sample(next_distance);
            let mut overlaps = vec![];
            while last < 500 {
                let next = rng.sample(next_distance);
                if last + next < 500 {
                    overlaps.push((last, last + next, Node(1)));
                    last = last + next + rng.sample(next_distance);
                } else { break }
            }
            let (_, _, _) = test_utils::run_ancestry_tests(&input_ancestry, &overlaps);
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
        assert_eq!(g.parents.len(), 2);
    }

    #[test]
    fn record_invalid_transmission_with_gap() {
        let mut g = Graph::with_initial_nodes(10, 10).unwrap().0;
        g.advance_time().unwrap();
        let birth = g.add_birth(1).unwrap();
        assert!(g.record_transmission(0, 5, Node(0), birth).is_ok());
        assert!(g.record_transmission(6, 10, Node(0), birth).is_err());
        assert_eq!(g.parents.len(), 1);
    }

    #[test]
    fn record_invalid_transmission_with_overlap() {
        let mut g = Graph::with_initial_nodes(10, 10).unwrap().0;
        g.advance_time().unwrap();
        let birth = g.add_birth(1).unwrap();
        assert!(g.record_transmission(0, 5, Node(0), birth).is_ok());
        assert!(g.record_transmission(4, 10, Node(1), birth).is_err());
        assert_eq!(g.parents.len(), 1);
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
            assert_eq!(queue.len(), 1);
            assert_eq!(queue, test_utils::naive_ancestry_intersection(Node(n), &g));
        }
    }
}

#[cfg(test)]
mod propagation_tests {
    use super::*;

    #[test]
    fn propagation_test0() {
        let mut graph = Graph::with_initial_nodes(10, 10).unwrap().0;
        graph.advance_time().unwrap();
        let birth = graph.add_birth(1).unwrap();
        graph.record_transmission(0, 5, Node(0), birth).unwrap();
        graph.record_transmission(5, 10, Node(1), birth).unwrap();
        let _ = propagate_ancestry_changes(PropagationOptions::default(), &mut graph);
    }
}
