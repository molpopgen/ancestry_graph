use std::collections::HashMap;

use nohash::BuildNoHashHasher;

use crate::exploration::AncestrySegment;
use crate::exploration::Edge;
use crate::exploration::GenomicInterval;
use crate::Node;
use crate::NodeHash;
use crate::NodeStatus;
use crate::PropagationOptions;
use crate::QueuedNode;

#[derive(Default, Clone, Copy)]
struct Range {
    start: usize,
    stop: usize,
}

#[derive(Default)]
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
}

#[derive(Default)]
struct Ancestry {
    ancestry: Vec<AncestrySegment>,
    ranges: Vec<Range>,
}

impl Ancestry {
    fn with_initial_nodes(num_nodes: usize) -> Self {
        Self {
            ranges: vec![Range::default(); num_nodes],
            ..Default::default()
        }
    }
}

#[derive(Default)]
struct Edges {
    edges: Vec<Edge>,
    ranges: Vec<Range>,
}

impl Edges {
    fn with_initial_nodes(num_nodes: usize) -> Self {
        Self {
            ranges: vec![Range::default(); num_nodes],
            ..Default::default()
        }
    }
}

type NewParentEdges = HashMap<Node, Vec<Edge>, BuildNoHashHasher<usize>>;
type BirthAncestry = HashMap<Node, Vec<AncestrySegment>, BuildNoHashHasher<usize>>;

#[derive(Default)]
pub struct Graph {
    current_time: i64,
    genome_length: i64,
    birth_time: Vec<i64>,
    node_status: Vec<NodeStatus>,
    output_node_map: Vec<Option<Node>>,

    edges: Edges,
    ancestry: Ancestry,
    simplified_edges: Edges,
    simplified_ancestry: Ancestry,

    new_parent_edges: NewParentEdges,
    birth_ancestry: BirthAncestry,
}

impl Graph {
    pub fn new(genome_length: i64) -> Self {
        Self {
            genome_length,
            ..Default::default()
        }
    }

    pub fn with_initial_nodes(num_nodes: usize, genome_length: i64) -> Self {
        Self {
            genome_length,
            birth_time: vec![0; num_nodes],
            node_status: vec![NodeStatus::Ancestor; num_nodes],
            edges: Edges::with_initial_nodes(num_nodes),
            ancestry: Ancestry::with_initial_nodes(num_nodes),
            ..Default::default()
        }
    }

    pub fn add_birth(&mut self) -> Node {
        self.birth_time.push(self.current_time);
        self.node_status.push(NodeStatus::Birth);
        let node = Node(self.birth_time.len() - 1);
        let inserted = self.birth_ancestry.insert(node, vec![]);
        assert!(inserted.is_none());
        node
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

    fn record_transmission(
        &mut self,
        left: i64,
        right: i64,
        parent: Node,
        child: Node,
    ) -> Result<(), ()> {
        assert!(validate_birth_order(parent, child, &self.birth_time));
        update_birth_ancestry(left, right, parent, child, &mut self.birth_ancestry);
        add_parent_edge(left, right, parent, child, &mut self.new_parent_edges);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AncestryIntersection {
    left: i64,
    right: i64,
    mapped_node: Node,
    //edge_index: Index,
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

struct Overlaps<'overlapper> {
    left: i64,
    right: i64,
    overlaps: &'overlapper [AncestryIntersection],
}

fn update_ancestry_intersection(
    edge: &Edge,
    ancestry: &[AncestrySegment],
    queue: &mut Vec<AncestryIntersection>,
) {
    for aseg in ancestry {
        if edge.overlaps(aseg) {
            let left = std::cmp::max(edge.left(), aseg.left());
            let right = std::cmp::min(edge.right(), aseg.right());
            queue.push(AncestryIntersection {
                left,
                right,
                mapped_node: aseg.mapped_node,
            });
        }
    }
}

fn ancestry_intersection(
    node: Node,
    edges: &Edges,
    ancestry: &Ancestry,
    queue: &mut Vec<AncestryIntersection>,
) {
    let parent_edges = {
        let range = edges.ranges[node.as_index()];
        &edges.edges[range.start..range.stop]
    };

    for edge in parent_edges {
        let child_ancestry = {
            let range = ancestry.ranges[edge.child.as_index()];
            &ancestry.ancestry[range.start..range.stop]
        };
        update_ancestry_intersection(edge, child_ancestry, queue);
    }
}

fn finalize_ancestry_intersection(queue: &mut Vec<AncestryIntersection>) {
    queue.sort_unstable_by_key(|x| x.left);
    // Sentinel
    if !queue.is_empty() {
        queue.push(AncestryIntersection {
            left: i64::MAX,
            right: i64::MAX,
            mapped_node: Node(usize::MAX),
        })
    }
}

fn validate_birth_order(parent: Node, child: Node, birth_time: &[i64]) -> bool {
    birth_time[parent.as_index()] < birth_time[child.as_index()]
}

fn update_birth_ancestry(
    left: i64,
    right: i64,
    parent: Node,
    child: Node,
    birth_ancestry: &mut BirthAncestry,
) {
    if let Some(ancestry) = birth_ancestry.get_mut(&child) {
        if let Some(last) = ancestry.last() {
            if left != last.right() {
                panic!("transmitted segments not contiguous");
            }
            ancestry.push(AncestrySegment {
                left,
                right,
                parent: Some(parent),
                mapped_node: child,
            });
        }
    } else {
        panic!("individual is not a birth");
    }
}

fn add_parent_edge(
    left: i64,
    right: i64,
    parent: Node,
    child: Node,
    new_parent_edges: &mut NewParentEdges,
) {
    let new_edge = Edge { left, right, child };
    if let Some(edges) = new_parent_edges.get_mut(&parent) {
        edges.push(new_edge);
    } else {
        new_parent_edges.insert(parent, vec![new_edge]);
    }
}

fn update_ancestry(
    node: Node,
    left: i64,
    right: i64,
    mapped_node: Node,
    current_ancestry: &mut AncestrySegment,
    output_ancestry: &mut Ancestry,
) -> (usize, i32) {
    todo!()
}

fn process_node(
    node: Node,
    node_input_ancestry: &mut [AncestrySegment],
    queue: &[AncestryIntersection],
    temp_edges: &mut Vec<Edge>,
    output_ancestry: &mut Ancestry,
) {
    if queue.is_empty() {
        todo!("no overlaps -- this node is extinct!")
    }

    let mut overlapper = AncestryOverlapper::new(node, queue);
    let mut current_input_ancestry = 0_usize;
    let mut overlaps = overlapper.calculate_next_overlap_set();
    let mut num_ancestry_changes = 0;
    debug_assert!(overlaps.is_some());

    while current_input_ancestry < node_input_ancestry.len() {
        let a = &mut node_input_ancestry[current_input_ancestry];
        if let Some(overlaps) = overlapper.calculate_next_overlap_set() {
            if a.right > overlaps.left && overlaps.right > a.left {
                let mapped_node;
                if overlaps.overlaps.len() == 1 {
                    mapped_node = overlaps.overlaps[0].mapped_node;
                    todo!("unary");
                } else {
                    mapped_node = node;
                    // output un-squashed edges
                    for seg in overlaps.overlaps {
                        temp_edges.push(Edge {
                            left: seg.left,
                            right: seg.right,
                            child: seg.mapped_node,
                        })
                    }
                }

                // FIXME: wrong design.
                // We are less interested in counting
                // changes than sending them into our
                // node heap for future processing
                let (increment, num_changes) = update_ancestry(
                    node,
                    overlaps.left,
                    overlaps.right,
                    mapped_node,
                    a,
                    output_ancestry,
                );
                current_input_ancestry += increment;
                num_ancestry_changes += num_changes;
            } else {
                current_input_ancestry += 1;
            }
        } else {
            break;
        }
    }
    todo!()
}

#[test]
fn test_with_initial_nodes() {
    let g = Graph::with_initial_nodes(10, 20);
    assert_eq!(g.birth_time.len(), 10);
    assert_eq!(g.node_status.len(), 10);
    assert_eq!(g.edges.ranges.len(), 10);
    assert_eq!(g.ancestry.ranges.len(), 10);
}

#[test]
fn test_lifetime() {
    let q = vec![];
    let mut ao = AncestryOverlapper::new(Node(0), &q);
    while let Some(_) = ao.calculate_next_overlap_set() {}
}
