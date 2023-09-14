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

    node_heap: NodeHeap,
    edges: Edges,
    ancestry: Ancestry,
    simplified_edges: Edges,
    simplified_ancestry: Ancestry,

    // This could also be an encapsulated edge table:
    // If we require that all birth edges be generated at once:
    // * We can cache results per parent...
    // * ... then lift them over to and edge table ...
    // * And track mapping of parent -> range index
    new_parent_edges: NewParentEdges,
    // TODO: this should go right into
    // the simplified ancestry:
    // * Track a mapping of birth node -> output node
    // * Do the valiation in "real time"
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
        if let Some(output_node) = output_node_map[edge.child.as_index()] {
            let range = ancestry.ranges[output_node.as_index()];
            let child_ancestry = &ancestry.ancestry[range.start..range.stop];
            println!("ancestry of child {output_node:?} = {child_ancestry:?}");
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
                        })
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
    graph.output_node_map.resize(graph.birth_time.len(), None);
}

fn liftover_ancestry_since_last_node(
    node: Node,
    last: Node,
    mut next_output_node: usize,
    input_ancestry: &Ancestry,
    output_ancestry: &mut Ancestry,
    output_node_map: &mut [Option<Node>],
) -> usize {
    println!("since last");
    let mut start = last.as_index();
    let ranges = &input_ancestry.ranges[start..node.as_index()];
    println!(
        "the anc range of this node = {:?}",
        input_ancestry.ranges[node.as_index()]
    );
    println!("ar = {ranges:?}");
    while start < ranges.len() {
        if let Some(i) = ranges[start..].iter().position(|r| r.start == r.stop) {
            println!("node {} has no ancestry", start + i);
            let current_len = output_ancestry.ancestry.len();
            let current_ranges_len = output_ancestry.ranges.len();
            let j = ranges[start].start;
            let k = ranges[start + i].start;
            println!("copying ancestry: {:?}", &input_ancestry.ancestry[j..k]);
            output_ancestry
                .ancestry
                .extend_from_slice(&input_ancestry.ancestry[j..k]);
            for i in output_ancestry.ancestry.iter_mut().skip(current_len) {
                if let Some(mapped_node) = output_node_map[i.mapped_node.as_index()] {
                    i.mapped_node = mapped_node;
                } else {
                    output_node_map[i.mapped_node.as_index()] = Some(Node(next_output_node));
                    next_output_node += 1;
                }
                i.parent = None;
                println!(
                    "copied: {i:?}, mapped to {:?}",
                    output_node_map[i.mapped_node.as_index()]
                );
            }
            println!(
                "copying ranges {:?}",
                &input_ancestry.ranges[start..start + i]
            );
            output_ancestry
                .ranges
                .extend_from_slice(&input_ancestry.ranges[start..start + i]);
            let mut offset = current_len;
            for i in &mut output_ancestry.ranges[current_ranges_len..] {
                let delta = i.stop - i.start;
                i.start = offset;
                i.stop = i.start + delta;
                offset += i.stop;
            }
            start += i + 1;
        } else {
            //todo!(
            //    "have to copy over the rest?, {:?}, for node {} ({last:?})",
            //    &ranges[start..],
            //    last.as_index() + start
            //);
            start += ranges.len();
        }
        //ranges = &ranges[start..];
    }
    next_output_node
}

fn liftover_ancestry_from_start(
    node: Node,
    mut next_output_node: usize,
    input_ancestry: &Ancestry,
    output_ancestry: &mut Ancestry,
    output_node_map: &mut [Option<Node>],
) -> usize {
    let mut start = 0_usize;
    let mut ranges = &input_ancestry.ranges[start..node.as_index()];
    println!(
        "the anc range of this node = {:?}",
        input_ancestry.ranges[node.as_index()]
    );
    println!("ar = {ranges:?}");
    while start < ranges.len() {
        if let Some(i) = ranges[start..].iter().position(|r| r.start == r.stop) {
            println!("node {} has no ancestry", start + i);
            let current_len = output_ancestry.ancestry.len();
            let current_ranges_len = output_ancestry.ranges.len();
            let j = ranges[start].start;
            let k = ranges[start + i].start;
            println!("copying ancestry: {:?}", &input_ancestry.ancestry[j..k]);
            output_ancestry
                .ancestry
                .extend_from_slice(&input_ancestry.ancestry[j..k]);
            for i in output_ancestry.ancestry.iter_mut().skip(current_len) {
                if let Some(mapped_node) = output_node_map[i.mapped_node.as_index()] {
                    i.mapped_node = mapped_node;
                } else {
                    output_node_map[i.mapped_node.as_index()] = Some(Node(next_output_node));
                    next_output_node += 1;
                }
                i.parent = None;
                println!(
                    "copied: {i:?}, mapped to {:?}",
                    output_node_map[i.mapped_node.as_index()]
                );
            }
            println!(
                "copying ranges {:?}",
                &input_ancestry.ranges[start..start + i]
            );
            output_ancestry
                .ranges
                .extend_from_slice(&input_ancestry.ranges[start..start + i]);
            let mut offset = current_len;
            for i in &mut output_ancestry.ranges[current_ranges_len..] {
                let delta = i.stop - i.start;
                i.start = offset;
                i.stop = i.start + delta;
                offset += i.stop;
            }
            start += i + 1;
        } else {
            todo!("have to copy over the rest?, {:?}", &ranges[start..]);
            start += ranges.len();
        }
        //ranges = &ranges[start..];
    }
    next_output_node
}

fn liftover_edges_from_start(
    node: Node,
    input_edges: &Edges,
    output_node_map: &[Option<Node>],
    output_edges: &mut Edges,
) {
    let mut start = 0_usize;
    let mut ranges = &input_edges.ranges[start..node.as_index()];
    println!("input edge ranges: {:?}", input_edges.ranges);
    while start < ranges.len() {
        println!("current ranges = {ranges:?}");
        if let Some(i) = ranges[start..].iter().position(|r| r.start == r.stop) {
            println!(
                "i = {i:?}, start = {start}, output id = {:?}",
                output_node_map[start + i]
            );
            if output_node_map[start + i].is_some() {
                let current_len = output_edges.edges.len();
                let current_ranges_len = output_edges.ranges.len();
                let j = ranges[start].start;
                let k = ranges[start + i].start;
                println!("copying edges: {:?}", &input_edges.edges[j..k]);
                output_edges
                    .edges
                    .extend_from_slice(&input_edges.edges[j..k]);
                for i in output_edges.edges.iter_mut().skip(current_len) {
                    if let Some(child) = output_node_map[i.child.as_index()] {
                        i.child = child;
                    } else {
                        panic!()
                    }
                }
                println!(
                    "copying ranges {:?}",
                    &input_edges.ranges[start..start + i + 1]
                );
                output_edges
                    .ranges
                    .extend_from_slice(&input_edges.ranges[start..start + i + 1]);
                let mut offset = current_len;
                for i in &mut output_edges.ranges[current_ranges_len..] {
                    let delta = i.stop - i.start;
                    i.start = offset;
                    i.stop = i.start + delta;
                    offset += i.stop;
                }
            }
            start += i + 1;
        } else {
            todo!("have to copy over the rest?, {ranges:?}");
            start += ranges.len();
        }
        //ranges = &ranges[start..];
    }
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
// 7. For copied ancestry segments, remap their parent field.
//    Is this the FATAL FLAW? Or do we have to remap
//    the parent node fields while processing later somehow?
//    We may have to punt on this step and see if we can
//    update this field later, but that has to wait until
//    we have an output node for any parent nodes?
// 8. Remap output nodes (at least in some cases?)
fn liftover_unchanged_data(
    node: Node,
    last_processed_node: Option<Node>,
    mut next_output_node: usize,
    graph: &mut Graph,
) -> usize {
    let range = graph.ancestry.ranges[node.as_index()];
    println!("range = {range:?}");
    if let Some(last) = last_processed_node {
        // liftover
        let last_range = graph.edges.ranges[last.as_index()];
        // TODO: remove this assert or put it all in a debug block
        if last_range.stop == range.start {
            assert_eq!(node.as_index() - last.as_index(), 1);
        }
        println!("{last_range:?} <=> {range:?}");
        //todo!("need to lift since the last node")
        next_output_node = liftover_ancestry_since_last_node(
            node,
            last,
            next_output_node,
            &graph.ancestry,
            &mut graph.simplified_ancestry,
            &mut graph.output_node_map,
        );
    } else {
        // FIXME: this is wrong.
        // And node whose value is > 0 implies
        // that something needs to be done here.
        next_output_node = liftover_ancestry_from_start(
            node,
            next_output_node,
            &graph.ancestry,
            &mut graph.simplified_ancestry,
            &mut graph.output_node_map,
        );
        liftover_edges_from_start(
            node,
            &graph.edges,
            &graph.output_node_map,
            &mut graph.simplified_edges,
        );
    }
    println!("output anc = {:?}", graph.simplified_ancestry);
    // Here, we need to copy all previous edges
    // and ancestry where the anscestry slice is > 0.
    // Have to be carefuly and allow for multiple
    // empty ancestry ranges separated by non-empty.
    // todo!("this block is wrong so far");
    // if range.start > 0 {
    //     todo!("lift from the beginning");
    // }
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
    let mut queue = vec![];
    let mut last_processed_node: Option<Node> = None;
    for parent in graph.new_parent_edges.keys() {
        graph
            .node_heap
            .insert(*parent, graph.birth_time[parent.as_index()]);
    }
    let mut temp_edges = vec![];
    let mut temp_ancestry = vec![];
    println!("{:?}", graph.output_node_map);
    while let Some(node) = graph.node_heap.pop() {
        println!(
            "{node:?}, birth time = {:?}",
            graph.birth_time[node.as_index()]
        );
        next_output_node =
            liftover_unchanged_data(node, last_processed_node, next_output_node, graph);
        let range = graph.edges.ranges[node.as_index()];
        let parent_edges = &graph.edges.edges[range.start..range.stop];
        let range = graph.ancestry.ranges[node.as_index()];
        println!("parent edges = {parent_edges:?}");
        println!(
            "(input) parent ancestry = {:?}",
            &graph.ancestry.ancestry[range.start..range.stop]
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
                    let child_ancestry =
                        &graph.simplified_ancestry.ancestry[range.start..range.stop];
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
        let node_input_ancestry = &mut graph.ancestry.ancestry[range.start..range.stop];
        next_output_node += process_node(
            node,
            node_input_ancestry,
            &queue,
            &mut graph.output_node_map,
            next_output_node,
            &graph.birth_time,
            &mut graph.node_heap,
            &mut temp_edges,
            &mut temp_ancestry,
        );
        println!(
            "temp anc for {node:?} => {:?} = {temp_ancestry:?}",
            graph.output_node_map[node.as_index()]
        );
        println!("temp_edges = {temp_edges:?}");

        //if !temp_edges.is_empty() {
        //    assert!(output_node_map[node.as_index()].is_none());
        //    output_node_map[node.as_index()] = Some(Node(next_output_node));
        //    next_output_node += 1;
        //}
        //todo!("logic in the next section is totally wrong");
        // Formal possibilities
        // 1. No ancestry intersection -- no output data, no remapping of the node.
        // 2. Ancestry, no edges -- node is extinct: no output edges, output ancestry, node is
        // remapped. Example: all unary transmission
        // 3. Edges and ancestry -- output the data, the node should already be remapped.

        //if !temp_edges.is_empty() {
        //    debug_assert!(!temp_ancestry.is_empty());
        //    graph.simplified_edges.edges.extend_from_slice(&temp_edges);
        //    graph.simplified_edges.ranges.push(Range {
        //        start: graph.simplified_edges.edges.len() - temp_edges.len(),
        //        stop: graph.simplified_edges.edges.len(),
        //    });
        //    assert_eq!(
        //        graph.simplified_edges.ranges.len(),
        //        graph.output_node_map[node.as_index()].unwrap().as_index() + 1
        //    );
        //}
        if !temp_ancestry.is_empty() {
            debug_assert!(!temp_ancestry.is_empty());
            graph.simplified_edges.edges.extend_from_slice(&temp_edges);
            graph.simplified_edges.ranges.push(Range {
                start: graph.simplified_edges.edges.len() - temp_edges.len(),
                stop: graph.simplified_edges.edges.len(),
            });
            assert_eq!(
                graph.simplified_edges.ranges.len(),
                graph.output_node_map[node.as_index()].unwrap().as_index() + 1,
                "{:?}",
                graph.simplified_edges.ranges,
            );
            graph
                .simplified_ancestry
                .ancestry
                .extend_from_slice(&temp_ancestry);
            graph.simplified_ancestry.ranges.push(Range {
                start: graph.simplified_ancestry.ancestry.len() - temp_ancestry.len(),
                stop: graph.simplified_ancestry.ancestry.len(),
            });
            assert_eq!(
                graph.simplified_ancestry.ranges.len(),
                graph.output_node_map[node.as_index()].unwrap().as_index() + 1
            );
        }

        //if !temp_ancestry.is_empty() {
        //    let current = graph.simplified_ancestry.ancestry.len();
        //    graph
        //        .simplified_ancestry
        //        .ancestry
        //        .extend_from_slice(&temp_ancestry);
        //    graph.simplified_ancestry.ranges.push(Range {
        //        start: current,
        //        stop: graph.simplified_ancestry.ancestry.len(),
        //    });
        //    println!(
        //        "added anc range: {:?}",
        //        graph.simplified_ancestry.ranges.last()
        //    );
        //}

        //if !temp_edges.is_empty() {
        //    let current = graph.simplified_edges.edges.len();
        //    graph.simplified_edges.edges.extend_from_slice(&temp_edges);
        //    graph.simplified_edges.ranges.push(Range {
        //        start: current,
        //        stop: graph.simplified_edges.edges.len(),
        //    });
        //    // Don't output ancestry for extinct nodes...
        //    // This step causes the problem referred to above:
        //    // we need some other concept of "ancestry for extinct nodes"
        //    // on order to deal with this.
        //    let current_output_ancestry_len = graph.simplified_ancestry.ancestry.len();
        //    graph
        //        .simplified_ancestry
        //        .ancestry
        //        .extend_from_slice(&temp_ancestry);
        //    graph.simplified_ancestry.ranges.push(Range {
        //        start: current_output_ancestry_len,
        //        stop: graph.simplified_ancestry.ancestry.len(),
        //    });
        //    println!(
        //        "added anc range: {:?}",
        //        graph.simplified_ancestry.ranges.last()
        //    );
        //} else {
        //    println!("extinct node {node:?} ancestry = {temp_ancestry:?}");
        //    if !temp_ancestry.is_empty() {
        //        let current = graph.simplified_edges.edges.len();
        //        graph.simplified_edges.ranges.push(Range {
        //            start: current,
        //            stop: current,
        //        });
        //    }
        //}

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
        let mut queue = vec![AncestryIntersection {
            left: 0,
            right: 2,
            mapped_node: Node(1),
        }];
        finalize_ancestry_intersection(&mut queue);
        let next_output_node = 0;

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
        assert_eq!(temp_ancestry.len(), 1);
        assert_eq!(
            temp_ancestry[0],
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
        let next_output_node = 0;

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
        assert_eq!(temp_ancestry.len(), 1);
        assert_eq!(
            temp_ancestry[0],
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

    fn setup_graph(
        edges: Vec<Vec<(i64, i64, usize)>>,
        ancestry: Vec<Vec<(i64, i64, usize, Option<usize>)>>,
        birth_time: Vec<i64>,
    ) -> Graph {
        let edges = setup_input_edges(edges);
        let ancestry = setup_input_ancestry(ancestry);
        Graph {
            edges,
            ancestry,
            birth_time,
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
        let edges = &simplified_edges.edges[range.start..range.stop];
        for (left, right, child) in expected {
            let child = output_node_map[child].unwrap();
            let edge = Edge { left, right, child };
            assert!(edges.contains(&edge), "{edge:?} not in {edges:?}");
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
        let ancestry = &simplified_ancestry.ancestry[range.start..range.stop];
        for (left, right, parent, mapped_node) in expected {
            let parent = parent.map(Node);
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
        graph.birth_time = birth_time;
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
            graph.node_heap.insert(Node(node), graph.birth_time[node]);
        }
        setup_output_node_map(&mut graph);

        propagate_ancestry_changes(&mut graph, None);
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
    }
}
