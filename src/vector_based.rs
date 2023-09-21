use std::collections::HashMap;

use nohash::BuildNoHashHasher;

use crate::exploration::AncestrySegment;
use crate::exploration::Edge;
use crate::exploration::GenomicInterval;
use crate::Node;
use crate::NodeHash;
use crate::NodeStatus;
use crate::PropagationOptions;

// NOTE: we can lose this and halve the storage/
// traversal bandwidth by simply storing starts and
// noting that something is empty if start i ==
// start i + 1
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

#[derive(Debug)]
struct RangeTable<T> {
    data: Vec<T>,
    ranges: Vec<Range>,
}

impl<T> Default for RangeTable<T> {
    fn default() -> Self {
        Self {
            data: Vec::default(),
            ranges: Vec::default(),
        }
    }
}

type Ancestry = RangeTable<AncestrySegment>;

//#[derive(Default, Debug)]
//struct Ancestry {
//    ancestry: Vec<AncestrySegment>,
//    ranges: Vec<Range>,
//}

impl Ancestry {
    fn with_initial_nodes(num_nodes: usize) -> Self {
        Self {
            ranges: vec![Range::default(); num_nodes],
            data: vec![],
        }
    }
}

type Edges = RangeTable<Edge>;

impl Edges {
    fn with_initial_nodes(num_nodes: usize) -> Self {
        Self {
            ranges: vec![Range::default(); num_nodes],
            ..Default::default()
        }
    }
}

#[derive(Default)]
struct Nodes {
    birth_time: Vec<i64>,
    status: Vec<NodeStatus>,
}

type NewParentEdges = HashMap<Node, Vec<Edge>, BuildNoHashHasher<usize>>;
type BirthAncestry = HashMap<Node, Vec<AncestrySegment>, BuildNoHashHasher<usize>>;

#[derive(Default)]
pub struct Graph {
    current_time: i64,
    genome_length: i64,
    nodes: Nodes,
    output_node_map: Vec<Option<Node>>,

    node_heap: NodeHeap,
    edges: Edges,
    ancestry: Ancestry,
    simplified_edges: Edges,
    simplified_ancestry: Ancestry,
    simplified_nodes: Nodes,

    // This could also be an encapsulated edge table:
    // If we require that all birth edges be generated at once:
    // * We can cache results per parent...
    // * ... then lift them over to and edge table ...
    // * And track mapping of parent -> range index
    new_parent_edges: NewParentEdges,
    // TODO: this should go right into
    // the simplified ancestry:
    // * Track a mapping of birth node -> output node
    // * Do the validation in "real time"
    // * Map an output node for the birth.
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
            nodes: Nodes {
                birth_time: vec![0; num_nodes],
                status: vec![NodeStatus::Ancestor; num_nodes],
            },
            edges: Edges::with_initial_nodes(num_nodes),
            ancestry: Ancestry::with_initial_nodes(num_nodes),
            ..Default::default()
        }
    }

    pub fn add_birth(&mut self) -> Node {
        self.nodes.birth_time.push(self.current_time);
        self.nodes.status.push(NodeStatus::Birth);
        let node = Node(self.nodes.birth_time.len() - 1);
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
        assert!(validate_birth_order(parent, child, &self.nodes.birth_time));
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
    index: usize,
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
            self.left = self.right;
            self.overlaps.retain(|o| o.right > self.left);
            if self.overlaps.is_empty() {
                self.left = self.queue[self.current_overlap].left;
            }
            let mut new_right = i64::MAX;
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
    for (index, aseg) in ancestry.iter().enumerate() {
        if edge.overlaps(aseg) {
            let left = std::cmp::max(edge.left(), aseg.left());
            let right = std::cmp::min(edge.right(), aseg.right());
            queue.push(AncestryIntersection {
                left,
                right,
                mapped_node: aseg.mapped_node,
                index,
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
            &ancestry.data[range.start..range.stop]
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
        if let Some(output_node) = output_node_map[edge.child.as_index()] {
            println!("{ancestry:?}");
            let range = ancestry.ranges[output_node.as_index()];
            let child_ancestry = &ancestry.data[range.start..range.stop];
            println!(
                "ancestry of child {:?}->{output_node:?} = {child_ancestry:?}, range = {range:?}",
                edge.child
            );
            update_ancestry_intersection(edge, child_ancestry, queue);
        }
        //let child_ancestry = {
        //    println!(
        //        "this fails if the child has not been lifted over to the output...{:?}",
        //        edge.child
        //    );
        //    let output_node = output_node_map[edge.child.as_index()].unwrap().as_index();
        //    let range = ancestry.ranges[output_node];
        //    &ancestry.ancestry[range.start..range.stop]
        //};
        //update_ancestry_intersection(edge, child_ancestry, queue);
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
            index: usize::MAX,
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
        parent: None,
    };
    println!("output = {output_segment:?}");
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
    output_ancestry: &mut Ancestry,
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

    println!("node input anc = {node_input_ancestry:?}");
    while current_input_ancestry < node_input_ancestry.len() {
        println!(
            "{current_input_ancestry:?}, {:?}, {current_overlaps:?}",
            node_input_ancestry.len()
        );
        let a = &mut node_input_ancestry[current_input_ancestry];
        println!("{a:?}");
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
                    println!(
                        "output node mapping for parent of unary = {:?}",
                        output_node_id
                    );
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
                            // TODO: can re refactor out this unwrap?
                            //child: output_node_map[seg.mapped_node.as_index()].unwrap(),
                            child: seg.mapped_node,
                        });
                        // Assign the parent to the ancestry segment for the child node
                        let range = output_ancestry.ranges[seg.mapped_node.as_index()];
                        let aseg = &mut output_ancestry.data[range.start..range.stop][seg.index];
                        assert!(output_node_map[node.as_index()].is_some());
                        aseg.parent = output_node_map[node.as_index()];
                    }
                }
                println!("mapped_node = {mapped_node:?}",);

                println!("current_input_ancestry = {current_input_ancestry:?}");
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
                println!("updated to {current_overlaps:?}");
            } else {
                println!("here");
                if let Some(parent) = node_input_ancestry[current_input_ancestry].parent {
                    node_heap.insert(parent, birth_time[parent.as_index()])
                }
                current_input_ancestry += 1;
            }
        } else {
            break;
        }
    }

    // Any remaining input ancestry had no overlaps, and is therefore
    // lost. Add those segment parents to the heap.
    for i in node_input_ancestry[current_input_ancestry..].iter() {
        if let Some(parent) = i.parent {
            node_heap.insert(parent, birth_time[parent.as_index()])
        }
    }
    debug_assert!(current_overlaps.is_none());
    rv
}

fn setup_output_node_map(graph: &mut Graph) {
    graph.output_node_map.fill(None);
    graph
        .output_node_map
        .resize(graph.nodes.birth_time.len(), None);
}

fn liftover<T, F>(
    i: usize,
    j: usize,
    ranges: &[Range],
    input: &RangeTable<T>,
    output: &mut RangeTable<T>,
    f: &mut F,
) where
    F: FnMut(&T) -> T,
{
    let k = ranges[i].start;
    let l = if j < ranges.len() {
        ranges[j].start
    } else {
        ranges[j - 1].stop
    };
    let mut offset = output.data.len();
    output.data.extend(input.data[k..l].iter().map(f));
    output.ranges.extend(ranges[i..j].iter().map(|&r| {
        let delta = r.stop - r.start;
        let start = offset;
        let stop = start + delta;
        offset = stop;
        Range { start, stop }
    }));
}

// This function needs to:
// 1. Copy input ancestry ranges into the output.
// 2. The copied ancestry ranges need to be updated
//    with respect to their new coordinates in the output.
// 3. Copy input edge ranges into the output.
// 4. The copied edge ranges need to be updated
//    with respect to their new coordinates in the output.
// 5. For copied ancestry segments, remap their mapped_node field.
// 6. For copied edges, remap their child field.
// 7. Remap output nodes (at least in some cases?)
fn liftover_unchanged_node_data(
    node: Node,
    last: Option<Node>,
    mut next_output_node: usize,
    input_ancestry: &Ancestry,
    input_edges: &Edges,
    output_ancestry: &mut Ancestry,
    output_edges: &mut Edges,
    output_node_map: &mut [Option<Node>],
) -> usize {
    let (ancestry_ranges, edge_ranges) = {
        let x = if let Some(l) = last {
            assert!(node.as_index() > l.as_index());
            l.as_index() + 1
        } else {
            0
        };
        let ancestry_ranges = &input_ancestry.ranges[x..node.as_index()];
        let edge_ranges = &input_edges.ranges[x..node.as_index()];
        (ancestry_ranges, edge_ranges)
    };

    let mut start = 0;
    while start < ancestry_ranges.len() {
        let (end, next_start) = if let Some(i) = ancestry_ranges[start..]
            .iter()
            .position(|r| r.start == r.stop)
        {
            (start + i, start + i + 1)
        } else {
            (ancestry_ranges.len(), ancestry_ranges.len())
        };
        liftover(
            start,
            end,
            ancestry_ranges,
            input_ancestry,
            output_ancestry,
            &mut |&a| {
                let mapped_node = if let Some(mn) = output_node_map[a.mapped_node.as_index()] {
                    mn
                } else {
                    let rv = Node(next_output_node);
                    output_node_map[a.mapped_node.as_index()] = Some(rv);
                    next_output_node += 1;
                    rv
                };
                AncestrySegment {
                    parent: None,
                    mapped_node,
                    ..a
                }
            },
        );
        println!("node map now = {output_node_map:?}");
        println!("update edges");
        liftover(
            start,
            end,
            edge_ranges,
            input_edges,
            output_edges,
            &mut |&e| {
                println!(
                    "mapping {:?} to {:?}",
                    e.child,
                    output_node_map[e.child.as_index()]
                );
                Edge {
                    child: output_node_map[e.child.as_index()].unwrap(),
                    ..e
                }
            },
        );
        start = next_start;
    }
    next_output_node
}

fn propagate_ancestry_changes(graph: &mut Graph, next_output_node: Option<usize>) {
    let mut next_output_node = if let Some(x) = next_output_node { x } else { 0 };
    for (node, ancestry) in graph.birth_ancestry.iter_mut() {
        graph.output_node_map[node.as_index()] = Some(Node(next_output_node));
        println!("mapped birth {node:?} to output {next_output_node}");
        // Remap our birth node mapped ancestry.
        // In theory, we should be able to avoid this
        println!("input birth ancestry = {ancestry:?}");
        ancestry
            .iter_mut()
            .for_each(|a| a.mapped_node = Node(next_output_node));
        println!("remapped birth ancestry = {ancestry:?}");
        next_output_node += 1;
        let current = graph.simplified_ancestry.data.len();
        graph
            .simplified_ancestry
            .data
            .extend_from_slice(ancestry.as_slice());
        graph.simplified_ancestry.ranges.push(Range {
            start: current,
            stop: graph.simplified_ancestry.data.len(),
        });
        let current = graph.simplified_edges.data.len();
        graph.simplified_edges.ranges.push(Range {
            start: current,
            stop: current,
        });
    }
    let mut queue = vec![];
    let mut last_processed_node: Option<Node> = None;
    for parent in graph.new_parent_edges.keys() {
        graph
            .node_heap
            .insert(*parent, graph.nodes.birth_time[parent.as_index()]);
    }
    let mut temp_edges = vec![];
    let mut temp_ancestry = vec![];
    println!("{:?}", graph.output_node_map);
    while let Some(node) = graph.node_heap.pop() {
        println!(
            "{node:?}, birth time = {:?}",
            graph.nodes.birth_time[node.as_index()]
        );
        next_output_node = liftover_unchanged_node_data(
            node,
            last_processed_node,
            next_output_node,
            &graph.ancestry,
            &graph.edges,
            &mut graph.simplified_ancestry,
            &mut graph.simplified_edges,
            &mut graph.output_node_map,
        );
        let range = graph.edges.ranges[node.as_index()];
        let parent_edges = &graph.edges.data[range.start..range.stop];
        let range = graph.ancestry.ranges[node.as_index()];
        println!("parent edges = {parent_edges:?}");
        println!(
            "(input) parent ancestry = {:?}",
            &graph.ancestry.data[range.start..range.stop]
        );
        ancestry_intersection_part_deux(
            node,
            parent_edges,
            &graph.simplified_ancestry,
            &graph.output_node_map,
            &mut queue,
        );
        println!("q = {queue:?}");
        println!(
            "current node map = {:?} | {}",
            graph.output_node_map, next_output_node
        );
        if let Some(edges) = graph.new_parent_edges.get(&node) {
            println!("birth edges for {node:?} = {edges:?}");
            for edge in edges {
                // the CHILD ancestry MUST be a birth
                // and the child MUST already be in output_ancestry
                let child_output_node = graph.output_node_map[edge.child.as_index()];
                if let Some(child) = child_output_node {
                    println!(
                        "adding births for remapped child {:?} => {child:?}",
                        edge.child
                    );
                    let range = graph.simplified_ancestry.ranges[child.as_index()];
                    let child_ancestry = &graph.simplified_ancestry.data[range.start..range.stop];
                    println!("the child anc = {:?}", child_ancestry);
                    update_ancestry_intersection(edge, child_ancestry, &mut queue)
                } else {
                    panic!("{:?} must have an output node", edge.child);
                }
            }
        }
        finalize_ancestry_intersection(&mut queue);
        println!("final q = {queue:?}");
        let range = graph.ancestry.ranges[node.as_index()];
        let node_input_ancestry = &mut graph.ancestry.data[range.start..range.stop];
        next_output_node += process_node(
            node,
            node_input_ancestry,
            &queue,
            &mut graph.output_node_map,
            next_output_node,
            &graph.nodes.birth_time,
            &mut graph.simplified_ancestry,
            &mut graph.node_heap,
            &mut temp_edges,
            &mut temp_ancestry,
        );
        println!(
            "temp anc for {node:?} => {:?} = {temp_ancestry:?}",
            graph.output_node_map[node.as_index()]
        );
        println!("temp_edges = {temp_edges:?}");

        if !temp_ancestry.is_empty() {
            debug_assert!(!temp_ancestry.is_empty());
            graph.simplified_edges.data.extend_from_slice(&temp_edges);
            graph.simplified_edges.ranges.push(Range {
                start: graph.simplified_edges.data.len() - temp_edges.len(),
                stop: graph.simplified_edges.data.len(),
            });
            assert_eq!(
                graph.simplified_edges.ranges.len(),
                graph.output_node_map[node.as_index()].unwrap().as_index() + 1,
                "{:?}",
                graph.simplified_edges.ranges,
            );
            graph
                .simplified_ancestry
                .data
                .extend_from_slice(&temp_ancestry);
            graph.simplified_ancestry.ranges.push(Range {
                start: graph.simplified_ancestry.data.len() - temp_ancestry.len(),
                stop: graph.simplified_ancestry.data.len(),
            });
            assert_eq!(
                graph.simplified_ancestry.ranges.len(),
                graph.output_node_map[node.as_index()].unwrap().as_index() + 1
            );
        }

        queue.clear();
        temp_edges.clear();
        temp_ancestry.clear();
        last_processed_node = Some(node);
    }
    graph.new_parent_edges.clear();
    graph.birth_ancestry.clear();
}

#[test]
fn test_with_initial_nodes() {
    let g = Graph::with_initial_nodes(10, 20);
    assert_eq!(g.nodes.birth_time.len(), 10);
    assert_eq!(g.nodes.status.len(), 10);
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

// These are design stage tests that should probably
// be deleted at some point.
#[cfg(test)]
mod multistep_tests {
    use super::*;

    fn setup_input_edges(raw: Vec<Vec<(i64, i64, usize)>>) -> Edges {
        let mut edges = Edges::default();
        for r in raw {
            let current = edges.data.len();
            for (left, right, n) in r {
                edges.data.push(Edge {
                    left,
                    right,
                    child: Node(n),
                });
            }
            edges.ranges.push(Range {
                start: current,
                stop: edges.data.len(),
            });
        }
        edges
    }

    fn setup_input_ancestry(raw: Vec<Vec<(i64, i64, usize, Option<usize>)>>) -> Ancestry {
        let mut ancestry = Ancestry::default();
        for r in raw {
            let current = ancestry.data.len();
            for (left, right, mapped_node, pnode) in r {
                let parent = if let Some(p) = pnode {
                    Some(Node(p))
                } else {
                    None
                };
                ancestry.data.push(AncestrySegment {
                    left,
                    right,
                    parent,
                    mapped_node: Node(mapped_node),
                })
            }
            ancestry.ranges.push(Range {
                start: current,
                stop: ancestry.data.len(),
            });
        }
        ancestry
    }

    fn setup_graph(
        edges: Vec<Vec<(i64, i64, usize)>>,
        ancestry: Vec<Vec<(i64, i64, usize, Option<usize>)>>,
        birth_time: Vec<i64>,
    ) -> Graph {
        let edges = setup_input_edges(edges);
        let ancestry = setup_input_ancestry(ancestry);
        let status = vec![NodeStatus::Ancestor; birth_time.len()];
        Graph {
            edges,
            ancestry,
            nodes: Nodes { birth_time, status },
            ..Default::default()
        }
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
        let edges = &simplified_edges.data[range.start..range.stop];
        for (left, right, child) in expected {
            println!("child remaps to {:?}", output_node_map[child]);
            let child = output_node_map[child].unwrap();
            let edge = Edge { left, right, child };
            assert!(
                edges.contains(&edge),
                "node {node}: {edge:?} not in {edges:?}"
            );
        }
    }

    fn validate_ancestry(
        node: usize,
        expected: Vec<(i64, i64, Option<usize>, usize)>,
        output_node_map: &[Option<Node>],
        simplified_ancestry: &Ancestry,
    ) {
        let output_node = output_node_map[node].unwrap().as_index();
        assert!(
            output_node < simplified_ancestry.ranges.len(),
            "{node:?} -> {output_node:} out of range"
        );
        let range = simplified_ancestry.ranges[output_node];
        let ancestry = &simplified_ancestry.data[range.start..range.stop];
        for (left, right, parent, mapped_node) in expected {
            let parent = parent.map(|n| output_node_map[n].unwrap());
            let seg = AncestrySegment {
                left,
                right,
                parent,
                mapped_node: output_node_map[mapped_node].unwrap(),
            };
            assert!(ancestry.contains(&seg), "{seg:?} not in {ancestry:?}");
        }
    }

    //  Note: the PARENTAL nodes
    //  are indexed in order of birth time,
    //  PRESENT to PAST.
    //     1
    //   -----
    //   |   |
    //   |   0
    //   |   |
    //   3   2
    //
    // 2 and 3 are births, leaving 0 as unary.
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
        //
        // Update sept 7, 2023: all of this is fine.
        // We "just need" the type of liftover bookmark mentioned above.
        let birth_time = vec![1_i64, 0, 2, 2];
        let raw_edges_1 = vec![(0, 2, 0)];
        let raw_edges_0 = vec![];
        let raw_edges = vec![raw_edges_0, raw_edges_1];
        let raw_ancestry = vec![
            vec![(0, 2, 0, Some(1))],
            vec![(0, 2, 1, None)],
            //vec![(0, 2, 2, Some(0))],
            //vec![(0, 2, 3, Some(0))],
        ];
        let edges = setup_input_edges(raw_edges);
        let ancestry = setup_input_ancestry(raw_ancestry);

        let mut graph = Graph::new(2);
        graph.edges = edges;
        graph.ancestry = ancestry;
        graph.nodes.birth_time = birth_time;
        // Manually deal with the births
        graph.new_parent_edges.insert(
            Node(1),
            vec![Edge {
                left: 0,
                right: 2,
                child: Node(3),
            }],
        );
        graph.new_parent_edges.insert(
            Node(0),
            vec![Edge {
                left: 0,
                right: 2,
                child: Node(2),
            }],
        );
        graph.birth_ancestry.insert(
            Node(3),
            vec![AncestrySegment {
                left: 0,
                right: 2,
                mapped_node: Node(3),
                parent: Some(Node(1)),
            }],
        );
        graph.birth_ancestry.insert(
            Node(2),
            vec![AncestrySegment {
                left: 0,
                right: 2,
                mapped_node: Node(2),
                parent: Some(Node(0)),
            }],
        );

        setup_output_node_map(&mut graph);
        println!("{:?}", graph.output_node_map);
        propagate_ancestry_changes(&mut graph, None);
        println!("{:?}", graph.simplified_edges);
        println!("{:?}", graph.simplified_ancestry);
        println!("{:?}", graph.output_node_map);

        assert!(graph.output_node_map[1].is_some());
        validate_edges(0, vec![], &graph.output_node_map, &graph.simplified_edges);

        // node 1
        let output_edges = vec![(0, 2, 2), (0, 2, 3)];
        validate_edges(
            1,
            output_edges,
            &graph.output_node_map,
            &graph.simplified_edges,
        );

        assert!(graph.new_parent_edges.is_empty());
        assert!(graph.birth_ancestry.is_empty());
        assert!(graph.node_heap.is_empty());

        assert_eq!(
            graph.simplified_nodes.birth_time.len(),
            graph.output_node_map.iter().filter(|x| x.is_some()).count()
        );
    }

    // Tests propagation of parent status thru unary transmission
    //
    //     5
    //   -----
    //   |   |
    //   |   4
    //   |   |
    //   |   3
    //   |   |
    //   |  ---
    //   0  1 2
    //
    // 1 will "die", resulting in (5,(0,2)) being the remaining topo.
    #[test]
    fn test2() {
        let initial_edges = vec![
            vec![],
            vec![],
            vec![],
            vec![(0, 5, 1), (0, 5, 2)],
            vec![(0, 5, 3)],
            vec![(0, 5, 0), (0, 5, 4)],
        ];
        let initial_ancestry = vec![
            vec![(0, 5, 0, Some(5))],
            vec![], // This node has "lost all ancestry"
            vec![(0, 5, 2, Some(3))],
            vec![(0, 5, 3, Some(4))],
            vec![(0, 5, 4, Some(5))],
            vec![(0, 5, 5, None)],
        ];
        let initial_birth_times = vec![3, 3, 3, 2, 1, 0];
        let mut graph = setup_graph(initial_edges, initial_ancestry, initial_birth_times);
        graph.node_heap.insert(Node(3), graph.nodes.birth_time[3]);
        setup_output_node_map(&mut graph);
        propagate_ancestry_changes(&mut graph, None);

        for node in [3, 4] {
            assert!(graph.output_node_map[node].is_some());
            validate_edges(
                node,
                vec![],
                &graph.output_node_map,
                &graph.simplified_edges,
            );
            validate_ancestry(
                node,
                vec![(0, 5, None, 2)],
                &graph.output_node_map,
                &graph.simplified_ancestry,
            );
        }
        for node in [0, 2] {
            assert!(graph.output_node_map[node].is_some());
            validate_ancestry(
                node,
                vec![(0, 5, Some(5), node)],
                &graph.output_node_map,
                &graph.simplified_ancestry,
            )
        }
        validate_ancestry(
            5,
            vec![(0, 5, None, 5)],
            &graph.output_node_map,
            &graph.simplified_ancestry,
        );
        validate_edges(
            5,
            vec![(0, 5, 0), (0, 5, 2)],
            &graph.output_node_map,
            &graph.simplified_edges,
        );
        assert_eq!(
            graph.simplified_nodes.birth_time.len(),
            graph.output_node_map.iter().filter(|x| x.is_some()).count()
        );
    }

    // Tree 1 on [0,1):
    //
    //    6
    //   ---
    //   5 |
    //   | 4
    // --- ---
    // 0 1 2 3
    //
    // Tree 2 on [2,3)
    //    6
    //   ---
    //   5 |
    //   | 4
    // --- ---
    // 2 3 0 1
    //
    // Nodes 2,3 lose all ancestry, propagating
    // a state of no overlap to some parental nodes.
    #[test]
    fn test5() {
        let initial_edges = vec![
            vec![],
            vec![],
            vec![],
            vec![],
            vec![(2, 3, 0), (2, 3, 1), (0, 1, 2), (0, 1, 3)],
            vec![(0, 1, 0), (0, 1, 1), (2, 3, 2), (2, 3, 3)],
            vec![(0, 1, 4), (0, 1, 5), (2, 3, 4), (2, 3, 5)],
        ];

        let initial_ancestry = vec![
            vec![(0, 1, 0, Some(5)), (2, 3, 0, Some(4))],
            vec![(0, 1, 1, Some(5)), (2, 3, 1, Some(4))],
            vec![],
            vec![],
            vec![(0, 1, 4, Some(6)), (2, 3, 4, Some(6))],
            vec![(0, 1, 5, Some(6)), (2, 3, 5, Some(6))],
            vec![(0, 1, 6, None), (2, 3, 6, None)],
        ];
        let initial_birth_times = vec![3, 3, 3, 3, 2, 1, 0];
        let mut graph = setup_graph(initial_edges, initial_ancestry, initial_birth_times);
        for node in [5, 4] {
            graph
                .node_heap
                .insert(Node(node), graph.nodes.birth_time[node]);
        }
        setup_output_node_map(&mut graph);

        propagate_ancestry_changes(&mut graph, None);
        assert_eq!(
            graph.simplified_edges.ranges.len(),
            graph.simplified_ancestry.ranges.len()
        );
        println!("{:?}", graph.output_node_map);
        println!("{:?}", graph.simplified_edges);
        println!("{:?}", graph.simplified_ancestry);
        validate_edges(6, vec![], &graph.output_node_map, &graph.simplified_edges);
        println!("{:?}", graph.output_node_map);
        validate_ancestry(
            6,
            vec![(0, 1, None, 5), (2, 3, None, 4)],
            &graph.output_node_map,
            &graph.simplified_ancestry,
        );
        validate_edges(
            5,
            vec![(0, 1, 0), (0, 1, 1)],
            &graph.output_node_map,
            &graph.simplified_edges,
        );
        validate_ancestry(
            5,
            vec![(0, 1, None, 5)],
            &graph.output_node_map,
            &graph.simplified_ancestry,
        );
        validate_edges(
            4,
            vec![(2, 3, 0), (2, 3, 1)],
            &graph.output_node_map,
            &graph.simplified_edges,
        );
        validate_ancestry(
            4,
            vec![(2, 3, None, 4)],
            &graph.output_node_map,
            &graph.simplified_ancestry,
        );

        validate_ancestry(
            0,
            vec![(0, 1, Some(5), 0), (2, 3, Some(4), 0)],
            &graph.output_node_map,
            &graph.simplified_ancestry,
        );
        validate_ancestry(
            1,
            vec![(0, 1, Some(5), 1), (2, 3, Some(4), 1)],
            &graph.output_node_map,
            &graph.simplified_ancestry,
        );
    }

    //        9
    //     --------
    //     |   |  |
    //     8   |  |
    //     |   7  |
    //     |   |  6
    //  ----  --- ---
    //  0  1  2 3 4 5
    //
    //  5 dies, making 6 unary, forcing evaluation of node 9.
    //  This test stresses our liftover code for non-tip nodes.
    #[test]
    fn test6() {
        let initial_edges = vec![
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![(0, 5, 4), (0, 5, 5)],
            vec![(0, 5, 2), (0, 5, 3)],
            vec![(0, 5, 0), (0, 5, 1)],
            vec![(0, 5, 6), (0, 5, 7), (0, 5, 8)],
        ];

        let initial_ancestry = vec![
            vec![(0, 5, 0, Some(8))],
            vec![(0, 5, 1, Some(8))],
            vec![(0, 5, 2, Some(7))],
            vec![(0, 5, 3, Some(7))],
            vec![(0, 5, 4, Some(6))],
            vec![], // loss of ancestry
            vec![(0, 5, 6, Some(9))],
            vec![(0, 5, 7, Some(9))],
            vec![(0, 5, 8, Some(9))],
            vec![(0, 5, 9, None)],
        ];
        let initial_birth_times = vec![4, 4, 4, 4, 4, 4, 3, 2, 1, 0];
        assert_eq!(initial_edges.len(), initial_ancestry.len());
        assert_eq!(initial_ancestry.len(), initial_birth_times.len());
        let mut graph = setup_graph(initial_edges, initial_ancestry, initial_birth_times);
        graph.node_heap.insert(Node(6), graph.nodes.birth_time[6]);
        setup_output_node_map(&mut graph);
        propagate_ancestry_changes(&mut graph, None);

        assert!(graph.output_node_map[5].is_none());

        // All input nodes other than 5
        // should have an output mapping
        // NOTE: node 6 is unary, but has an output
        // mapping b/c we have to know that it is extinct, etc..
        for node in [0, 1, 2, 3, 4, 6, 7, 8, 9] {
            assert!(
                graph.output_node_map[node].is_some(),
                "{node} has no output mapping"
            );
        }

        validate_edges(6, vec![], &graph.output_node_map, &graph.simplified_edges);
        validate_edges(
            7,
            vec![(0, 5, 2), (0, 5, 3)],
            &graph.output_node_map,
            &graph.simplified_edges,
        );
        validate_edges(
            8,
            vec![(0, 5, 0), (0, 5, 1)],
            &graph.output_node_map,
            &graph.simplified_edges,
        );
        validate_edges(
            9,
            vec![(0, 5, 8), (0, 5, 7), (0, 5, 4)],
            &graph.output_node_map,
            &graph.simplified_edges,
        );

        validate_ancestry(
            9,
            vec![(0, 5, None, 9)],
            &graph.output_node_map,
            &graph.simplified_ancestry,
        );
        validate_ancestry(
            8,
            vec![(0, 5, Some(9), 8)],
            &graph.output_node_map,
            &graph.simplified_ancestry,
        );
        validate_ancestry(
            7,
            vec![(0, 5, Some(9), 7)],
            &graph.output_node_map,
            &graph.simplified_ancestry,
        );
        validate_ancestry(
            6,
            vec![(0, 5, None, 4)],
            &graph.output_node_map,
            &graph.simplified_ancestry,
        );
        validate_ancestry(
            4,
            vec![(0, 5, Some(9), 4)],
            &graph.output_node_map,
            &graph.simplified_ancestry,
        );
    }
}
