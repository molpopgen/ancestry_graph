use std::collections::HashMap;

use nohash::BuildNoHashHasher;

use crate::exploration::AncestrySegment;
use crate::exploration::Edge;
use crate::exploration::GenomicInterval;
use crate::Node;
use crate::NodeHash;
use crate::NodeStatus;
use crate::PropagationOptions;

#[derive(Default, Debug, Clone, Copy)]
struct Range {
    start: usize,
    stop: usize,
}

#[derive(Debug, Copy, Clone, PartialOrd, Ord)]
struct QueuedNode {
    birth_time: i64,
    node: Node,
}

impl PartialEq for QueuedNode {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
    }
}

impl Eq for QueuedNode {}

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

    fn is_empty(&self) -> bool {
        self.queued_nodes.is_empty()
    }
}

#[derive(Default, Debug)]
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

#[derive(Default, Debug)]
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

#[derive(Debug)]
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
    parent_edges: &[Edge],
    ancestry: &Ancestry,
    queue: &mut Vec<AncestryIntersection>,
) {
    for edge in parent_edges {
        let child_ancestry = {
            let range = ancestry.ranges[edge.child.as_index()];
            &ancestry.ancestry[range.start..range.stop]
        };
        update_ancestry_intersection(edge, child_ancestry, queue);
    }
}

fn ancestry_intersection_part_deux(
    node: Node,
    parent_edges: &[Edge],
    ancestry: &Ancestry,
    output_node_map: &[Option<Node>],
    queue: &mut Vec<AncestryIntersection>,
) {
    for edge in parent_edges {
        println!("edge = {edge:?}");
        let child_ancestry = {
            println!(
                "this fails if the child has not been lifted over to the output...{:?}",
                edge.child
            );
            let output_node = output_node_map[edge.child.as_index()].unwrap().as_index();
            let range = ancestry.ranges[output_node];
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
    is_unary: bool,
    mapped_node: Node,
    birth_time: &[i64],
    current_ancestry: &mut AncestrySegment,
    node_heap: &mut NodeHeap,
    temp_ancestry: &mut Vec<AncestrySegment>,
) -> usize {
    let temp_left = std::cmp::max(left, current_ancestry.left);
    let temp_right = std::cmp::max(right, current_ancestry.right);
    let mut increment = 0;
    println!(
        "update: {current_ancestry:?}, {temp_left}, {temp_right}, {node:?} <=> {mapped_node:?}"
    );

    // The second condition means a unary transmission that
    // needs propagating
    if current_ancestry.left != temp_left || is_unary {
        println!("adding change from {node:?}");
        if let Some(parent) = current_ancestry.parent {
            node_heap.insert(parent, birth_time[parent.as_index()])
        }
    }
    if current_ancestry.right != temp_right {
        println!("adding change from {node:?}, case 2");
        // DUPLICATION
        if let Some(parent) = current_ancestry.parent {
            node_heap.insert(parent, birth_time[parent.as_index()])
        }
        current_ancestry.left = temp_right;
    } else {
        increment = 1;
    }
    let output_segment = AncestrySegment {
        left,
        right,
        mapped_node,
        parent: current_ancestry.parent,
    };
    temp_ancestry.push(output_segment);

    increment
}

fn process_node(
    node: Node,
    node_input_ancestry: &mut [AncestrySegment],
    queue: &[AncestryIntersection],
    output_node_map: &mut [Option<Node>],
    next_output_node: usize,
    birth_time: &[i64],
    node_heap: &mut NodeHeap,
    temp_edges: &mut Vec<Edge>,
    temp_ancestry: &mut Vec<AncestrySegment>,
) -> usize {
    if queue.is_empty() {
        todo!("no overlaps -- node {node:?} is extinct!")
    }
    let mut output_node_id = output_node_map[node.as_index()];

    let mut overlapper = AncestryOverlapper::new(node, queue);
    let mut current_input_ancestry = 0_usize;
    let mut current_overlaps = overlapper.calculate_next_overlap_set();
    debug_assert!(current_overlaps.is_some());
    let mut rv = 0_usize;

    while current_input_ancestry < node_input_ancestry.len() {
        println!(
            "{current_input_ancestry:?}, {:?}, {current_overlaps:?}",
            node_input_ancestry.len()
        );
        let a = &mut node_input_ancestry[current_input_ancestry];
        let mut is_unary = false;
        if let Some(ref overlaps) = current_overlaps {
            if a.right > overlaps.left && overlaps.right > a.left {
                let mapped_node;
                if overlaps.overlaps.len() == 1 {
                    is_unary = true;
                    mapped_node = overlaps.overlaps[0].mapped_node;
                    println!(
                        "unary mapped node is {:?} -> {mapped_node:?}",
                        overlaps.overlaps[0].mapped_node
                    );
                    if output_node_id.is_none() {
                        output_node_map[node.as_index()] = Some(Node(next_output_node));
                        output_node_id = Some(Node(next_output_node));
                        rv += 1;
                    }
                    // TODO: if node is a sample, we have more work to
                    // do here.
                } else {
                    //mapped_node=node;
                    if let Some(output) = output_node_id {
                        mapped_node = output;
                    } else {
                        output_node_map[node.as_index()] = Some(Node(next_output_node));
                        mapped_node = Node(next_output_node);
                        rv += 1;
                    }
                    // output un-squashed edges
                    for seg in overlaps.overlaps {
                        temp_edges.push(Edge {
                            left: seg.left,
                            right: seg.right,
                            child: seg.mapped_node,
                        })
                    }
                }
                println!(
                    "mapped_node = {mapped_node:?}, {:?}",
                    output_node_map[mapped_node.as_index()]
                );

                current_input_ancestry += update_ancestry(
                    node,
                    overlaps.left,
                    overlaps.right,
                    is_unary,
                    mapped_node,
                    birth_time,
                    a,
                    node_heap,
                    temp_ancestry,
                );
                println!("current_input_ancestry = {current_input_ancestry:?}");
                current_overlaps = overlapper.calculate_next_overlap_set();
            } else {
                current_input_ancestry += 1;
            }
        } else {
            break;
        }
    }
    debug_assert!(current_overlaps.is_none());
    rv
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

#[test]
fn test_queue_node_ord() {
    let mut v = vec![
        QueuedNode {
            node: Node(1),
            birth_time: 0,
        },
        QueuedNode {
            node: Node(0),
            birth_time: 0,
        },
    ];
    v.sort_unstable();
    assert_eq!(v[0].node, Node(0));
    let mut v = vec![
        QueuedNode {
            node: Node(1),
            birth_time: 0,
        },
        QueuedNode {
            node: Node(0),
            birth_time: 1,
        },
    ];
    v.sort_unstable();
    assert_eq!(v[0].node, Node(1));
}

#[cfg(test)]
mod test_process_node {
    use super::*;

    //      0
    //     ---
    //     1 2
    //
    // 2 will die, leaving 0 as unary
    #[test]
    fn test_single_steps_1() {
        let birth_time = vec![0_i64, 1, 1];
        let mut output_node_map = vec![];
        for i in 0..birth_time.len() {
            output_node_map.push(Some(Node(i)));
        }
        let mut node_heap = NodeHeap::default();
        let mut temp_edges: Vec<Edge> = vec![];
        let mut temp_ancestry: Vec<AncestrySegment> = vec![];
        let mut node_input_ancestry = vec![AncestrySegment {
            left: 0,
            right: 2,
            mapped_node: Node(0),
            parent: None,
        }];
        let mut output_ancestry = Ancestry::default();
        let mut queue = vec![AncestryIntersection {
            left: 0,
            right: 2,
            mapped_node: Node(1),
        }];
        finalize_ancestry_intersection(&mut queue);
        let mut next_output_node = 0;

        process_node(
            Node(0),
            &mut node_input_ancestry,
            &queue,
            &mut output_node_map,
            next_output_node,
            &birth_time,
            &mut node_heap,
            &mut temp_edges,
            &mut temp_ancestry,
        );
        assert!(temp_edges.is_empty());
        assert!(node_heap.is_empty());
        assert_eq!(output_ancestry.ancestry.len(), 1);
        assert_eq!(
            output_ancestry.ancestry[0],
            AncestrySegment {
                left: 0,
                right: 2,
                mapped_node: Node(1),
                parent: None
            }
        );
    }

    //      0
    //      |
    //      1
    //     ---
    //     2 3
    //
    // 3 will die, leaving 1 as unary
    #[test]
    fn test_single_steps_2() {
        let birth_time = vec![0_i64, 1, 2, 2];
        let mut output_node_map = vec![];
        for i in 0..birth_time.len() {
            output_node_map.push(Some(Node(i)));
        }
        let mut node_heap = NodeHeap::default();
        let mut temp_edges: Vec<Edge> = vec![];
        let mut temp_ancestry: Vec<AncestrySegment> = vec![];
        let mut node_input_ancestry = vec![AncestrySegment {
            left: 0,
            right: 2,
            mapped_node: Node(1),
            parent: Some(Node(0)),
        }];
        let mut output_ancestry = Ancestry::default();
        let mut queue = vec![AncestryIntersection {
            left: 0,
            right: 2,
            mapped_node: Node(2),
        }];
        finalize_ancestry_intersection(&mut queue);
        let mut next_output_node = 0;

        process_node(
            Node(1),
            &mut node_input_ancestry,
            &queue,
            &mut output_node_map,
            next_output_node,
            &birth_time,
            &mut node_heap,
            &mut temp_edges,
            &mut temp_ancestry,
        );
        assert!(temp_edges.is_empty());
        assert_eq!(node_heap.len(), 1);
        assert!(node_heap.queued_nodes.contains(&Node(0)));
        assert_eq!(temp_ancestry.len(), 1);
        assert_eq!(
            temp_ancestry[0],
            AncestrySegment {
                left: 0,
                right: 2,
                mapped_node: Node(2),
                parent: Some(Node(0))
            }
        );
    }

    //      0
    //      |
    //      1
    //     ---
    //     2 3
    //
    // 2 and 3 are births
    #[test]
    fn test_single_steps_3() {
        let birth_time = vec![0_i64, 1, 2, 2];
        let mut output_node_map = vec![];
        for i in 0..birth_time.len() {
            output_node_map.push(Some(Node(i)));
        }
        let mut node_heap = NodeHeap::default();
        let mut temp_edges: Vec<Edge> = vec![];
        let mut temp_ancestry: Vec<AncestrySegment> = vec![];
        let mut node_input_ancestry = vec![AncestrySegment {
            left: 0,
            right: 2,
            mapped_node: Node(1),
            parent: Some(Node(0)),
        }];
        let mut output_ancestry = Ancestry::default();
        let mut queue = vec![
            AncestryIntersection {
                left: 0,
                right: 2,
                mapped_node: Node(2),
            },
            AncestryIntersection {
                left: 0,
                right: 2,
                mapped_node: Node(3),
            },
        ];
        finalize_ancestry_intersection(&mut queue);
        let mut next_output_node = 0;

        process_node(
            Node(1),
            &mut node_input_ancestry,
            &queue,
            &mut output_node_map,
            next_output_node,
            &birth_time,
            &mut node_heap,
            &mut temp_edges,
            &mut temp_ancestry,
        );
        assert!(node_heap.is_empty());
        assert_eq!(output_ancestry.ancestry.len(), 1);
        assert_eq!(
            output_ancestry.ancestry[0],
            AncestrySegment {
                left: 0,
                right: 2,
                mapped_node: Node(1),
                parent: Some(Node(0))
            }
        );
        assert_eq!(temp_edges.len(), 2);
    }
}

// These are design stage tests that should probably
// be deleted at some point.
#[cfg(test)]
mod multistep_tests {
    use super::*;

    fn setup_input_edges(raw: Vec<Vec<(i64, i64, usize)>>) -> Edges {
        let mut edges = Edges::default();
        for r in raw {
            let current = edges.edges.len();
            for (left, right, n) in r {
                edges.edges.push(Edge {
                    left,
                    right,
                    child: Node(n),
                });
            }
            edges.ranges.push(Range {
                start: current,
                stop: edges.edges.len(),
            });
        }
        edges
    }

    fn setup_input_ancestry(raw: Vec<Vec<(i64, i64, usize, Option<usize>)>>) -> Ancestry {
        let mut ancestry = Ancestry::default();
        for r in raw {
            let current = ancestry.ancestry.len();
            for (left, right, mapped_node, pnode) in r {
                let parent = if let Some(p) = pnode {
                    Some(Node(p))
                } else {
                    None
                };
                ancestry.ancestry.push(AncestrySegment {
                    left,
                    right,
                    parent,
                    mapped_node: Node(mapped_node),
                })
            }
            ancestry.ranges.push(Range {
                start: current,
                stop: ancestry.ancestry.len(),
            });
        }
        ancestry
    }

    fn validate_edges(
        node: usize,
        expected: Vec<(i64, i64, usize)>,
        output_node_map: &[Option<Node>],
        simplified_edges: &Edges,
    ) {
        let output_node = output_node_map[node].unwrap().as_index();
        assert!(
            output_node < simplified_edges.ranges.len(),
            "{node:?} -> {output_node:} out of range"
        );
        let range = simplified_edges.ranges[output_node];
        let edges = &simplified_edges.edges[range.start..range.stop];
        for (left, right, child) in expected {
            let child = output_node_map[child].unwrap();
            let edge = Edge { left, right, child };
            assert!(edges.contains(&edge), "{edge:?} not in {edges:?}");
        }
    }

    //     0
    //   -----
    //   |   |
    //   |   1
    //   |   |
    //   3   2
    //
    // 2 and 3 are births, leaving 1 as unary.
    #[test]
    fn test_multisteps_1() {
        // We have a problem:
        // When a node is extinct, we don't want to
        // propagate its ancestry to the output/simplified ancestry.
        // But if we, say, cache it and then not output it,
        // we may lose info needed to map output mutations,
        // UNLESS we take a queue from tskit and map mutations
        // immediately when we process a node?
        //
        // There is a way out:
        // the output node map tracks extant/extinct status + output id
        // During our "liftover" step (not implementer yet), we only
        // output up until that node, effectively erasing it one simplification
        // down the road.
        // Corollary: we could add the extinct nodes to the heap so that
        // we don't have to explicly search for them?
        let birth_time = vec![0_i64, 1, 2, 2];
        let raw_edges_0 = vec![(0, 2, 1)];
        let raw_edges_1 = vec![];
        let raw_edges = vec![raw_edges_0, raw_edges_1, vec![], vec![], vec![]];
        let raw_ancestry = vec![
            vec![(0, 2, 0, None)],
            vec![(0, 2, 1, Some(0))],
            vec![(0, 2, 2, Some(1))],
            vec![],
            vec![(0, 2, 4, Some(0))],
        ];
        let edges = setup_input_edges(raw_edges);
        let ancestry = setup_input_ancestry(raw_ancestry);
        let mut node_heap = NodeHeap::default();
        node_heap.insert(Node(1), birth_time[1]);

        let mut temp_edges = vec![];
        let mut temp_ancestry = vec![];
        let mut graph = Graph::new(2);
        graph.edges = edges;
        graph.ancestry = ancestry;
        graph.birth_time = birth_time;
        // Manually deal with the births
        graph.new_parent_edges.insert(
            Node(0),
            vec![Edge {
                left: 0,
                right: 2,
                child: Node(3),
            }],
        );
        graph.new_parent_edges.insert(
            Node(1),
            vec![Edge {
                left: 0,
                right: 2,
                child: Node(3),
            }],
        );
        graph.birth_ancestry.insert(
            Node(3),
            vec![AncestrySegment {
                left: 0,
                right: 2,
                mapped_node: Node(3),
                parent: Some(Node(0)),
            }],
        );
        graph.birth_ancestry.insert(
            Node(2),
            vec![AncestrySegment {
                left: 0,
                right: 2,
                mapped_node: Node(2),
                parent: Some(Node(1)),
            }],
        );
        let mut output_node_map = vec![None; graph.birth_time.len()];
        let mut next_output_node = 0;

        for (node, ancestry) in graph.birth_ancestry.iter() {
            output_node_map[node.as_index()] = Some(Node(next_output_node));
            println!("mapped {node:?} to {next_output_node}");
            next_output_node += 1;
            let current = graph.simplified_ancestry.ancestry.len();
            graph
                .simplified_ancestry
                .ancestry
                .extend_from_slice(ancestry.as_slice());
            graph.simplified_ancestry.ranges.push(Range {
                start: current,
                stop: graph.simplified_ancestry.ancestry.len(),
            });
            let current = graph.simplified_edges.edges.len();
            graph.simplified_edges.ranges.push(Range {
                start: current,
                stop: current,
            });
        }

        // "simplify"
        let mut queue = vec![];
        let last_processed_node: Option<Node> = None;
        println!("{output_node_map:?}");
        while let Some(node) = node_heap.pop() {
            println!("{node:?}");
            let range = graph.edges.ranges[node.as_index()];
            println!("range = {range:?}");
            if let Some(last) = last_processed_node {
                // liftover
                let last_range = graph.edges.ranges[last.as_index()];
                println!("{last_range:?} <=> {range:?}");
            }
            let parent_edges = &graph.edges.edges[range.start..range.stop];
            println!("parent edges = {parent_edges:?}");
            ancestry_intersection_part_deux(
                node,
                &parent_edges,
                &graph.simplified_ancestry,
                &output_node_map,
                &mut queue,
            );
            if let Some(edges) = graph.new_parent_edges.get(&node) {
                for edge in edges {
                    // the CHILD ancestry MUST be a birth
                    // and the child MUST already be in output_ancestry
                    let child_output_node = output_node_map[edge.child.as_index()];
                    if let Some(child) = child_output_node {
                        let range = graph.simplified_ancestry.ranges[child.as_index()];
                        let child_ancestry =
                            &graph.simplified_ancestry.ancestry[range.start..range.stop];
                        update_ancestry_intersection(edge, child_ancestry, &mut queue)
                    } else {
                        panic!("{:?} must have an output node", edge.child);
                    }
                }
            }
            finalize_ancestry_intersection(&mut queue);
            let range = graph.ancestry.ranges[node.as_index()];
            let node_input_ancestry = &mut graph.ancestry.ancestry[range.start..range.stop];
            next_output_node += process_node(
                node,
                node_input_ancestry,
                &queue,
                &mut output_node_map,
                next_output_node,
                &graph.birth_time,
                &mut node_heap,
                &mut temp_edges,
                &mut temp_ancestry,
            );
            println!("temp anc = {temp_ancestry:?}");
            println!("temp_edges = {temp_edges:?}");

            //if !temp_edges.is_empty() {
            //    assert!(output_node_map[node.as_index()].is_none());
            //    output_node_map[node.as_index()] = Some(Node(next_output_node));
            //    next_output_node += 1;
            //}
            if !temp_ancestry.is_empty() {
                let current = graph.simplified_edges.edges.len();
                graph
                    .simplified_ancestry
                    .ancestry
                    .extend_from_slice(&temp_ancestry);
                graph.simplified_ancestry.ranges.push(Range {
                    start: current,
                    stop: graph.simplified_ancestry.ancestry.len(),
                });
            }

            if !temp_edges.is_empty() {
                let current = graph.simplified_edges.edges.len();
                graph.simplified_edges.edges.extend_from_slice(&temp_edges);
                graph.simplified_edges.ranges.push(Range {
                    start: current,
                    stop: graph.simplified_edges.edges.len(),
                });
                // Don't output ancestry for extinct nodes...
                // This step causes the problem referred to above:
                // we need some other concept of "ancestry for extinct nodes"
                // on order to deal with this.
                let current_output_ancestry_len = graph.simplified_ancestry.ancestry.len();
                graph
                    .simplified_ancestry
                    .ancestry
                    .extend_from_slice(&temp_ancestry);
                graph.simplified_ancestry.ranges.push(Range {
                    start: current_output_ancestry_len,
                    stop: graph.simplified_ancestry.ancestry.len(),
                });
            } else {
                println!("extinct node {node:?} ancestry = {temp_ancestry:?}");
                if !temp_ancestry.is_empty() {
                    let current = graph.simplified_edges.edges.len();
                    graph.simplified_edges.ranges.push(Range {
                        start: current,
                        stop: current,
                    });
                }
            }

            queue.clear();
            temp_edges.clear();
            temp_ancestry.clear();
        }
        println!("{:?}", graph.simplified_edges);
        println!("{:?}", graph.simplified_ancestry);
        println!("{output_node_map:?}");

        // Node1 should have no output mapping
        assert!(output_node_map[1].is_some());

        // node 0
        let output_edges = vec![(0, 2, 2), (0, 2, 3)];
        validate_edges(0, output_edges, &output_node_map, &graph.simplified_edges);
    }
}
