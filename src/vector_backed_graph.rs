use crate::Node;
use crate::NodeHash;
use crate::QueuedNode;

#[derive(Default, Debug)]
struct NodeHeap {
    queued_nodes: NodeHash,
    node_queue: std::collections::BinaryHeap<QueuedNode>,
}

impl NodeHeap {
    fn pop(&mut self) -> Option<Node> {
        match self.node_queue.pop() {
            Some(qn) => {
                self.queued_nodes.remove(&qn.node);
                Some(qn.node)
            }
            None => None,
        }
    }
}

#[derive(Default, Debug, PartialEq, Eq)]
struct Ancestry {
    left: Vec<i64>,
    right: Vec<i64>,
    unary_mapping: Vec<Option<Node>>,
}

impl Ancestry {
    fn new_sample(genome_length: i64) -> Self {
        Self {
            left: vec![0],
            right: vec![genome_length],
            unary_mapping: vec![None],
        }
    }

    fn add_ancestry(&mut self, left: i64, right: i64, unary_mapping: Option<Node>) {
        self.left.push(left);
        self.right.push(right);
        self.unary_mapping.push(unary_mapping);
    }

    fn clear(&mut self) {
        self.left.clear();
        self.right.clear();
        self.unary_mapping.clear();
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        debug_assert_eq!(self.left.len(), self.right.len());
        debug_assert_eq!(self.left.len(), self.unary_mapping.len());
        self.left.len()
    }

    fn push(&mut self, left: i64, right: i64, unary_mapping: Option<Node>) {
        let to_squash = if let Some(lright) = self.right.last_mut() {
            if lright == &left && self.unary_mapping[self.unary_mapping.len() - 1] == unary_mapping
            {
                Some(lright)
            } else {
                None
            }
        } else {
            None
        };
        if let Some(lright) = to_squash {
            *lright = right
        } else {
            self.left.push(left);
            self.right.push(right);
            self.unary_mapping.push(unary_mapping);
        }
    }

    fn ancestry(&self, i: usize) -> (i64, i64, Option<Node>) {
        (self.left[i], self.right[i], self.unary_mapping[i])
    }
}

struct Edge {
    left: i64,
    right: i64,
    child: Node,
}

#[derive(Default, Debug)]
struct Edges {
    left: Vec<i64>,
    right: Vec<i64>,
    child: Vec<Node>,
}

impl Edges {
    fn clear(&mut self) {
        self.left.clear();
        self.right.clear();
        self.child.clear();
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        debug_assert_eq!(self.left.len(), self.right.len());
        debug_assert_eq!(self.left.len(), self.child.len());
        self.left.len()
    }

    fn push(&mut self, left: i64, right: i64, child: Node) {
        self.left.push(left);
        self.right.push(right);
        self.child.push(child);
    }
}

#[derive(Default)]
struct Nodes {
    birth_time: Vec<i64>,
}

#[derive(Default)]
struct Tables {
    ancestry: Vec<Ancestry>,
    edges: Vec<Edges>,
    parents: Vec<Vec<Node>>,
    children: Vec<Vec<Node>>,
    nodes: Nodes,
}

impl Tables {
    fn recycle_index_as_birth_node(&mut self, index: usize, current_time: i64, genome_length: i64) {
        self.nodes.birth_time[index] = current_time;
        self.edges[index].clear();
        self.parents[index].clear();
        self.children[index].clear();
        self.ancestry[index].clear();
        self.ancestry[index].add_ancestry(0, genome_length, None);
    }

    fn add_new_birth_node(&mut self, current_time: i64, genome_length: i64) -> usize {
        self.nodes.birth_time.push(current_time);
        self.ancestry.push(Ancestry::new_sample(genome_length));
        self.edges.push(Edges::default());
        self.parents.push(vec![]);
        self.children.push(vec![]);
        self.nodes.birth_time.len() - 1
    }
}

struct Graph {
    tables: Tables,
    node_heap: NodeHeap,
    current_time: i64,
    genome_length: i64,
    free_nodes: Vec<usize>,
}

impl Graph {
    pub fn new(genome_length: i64) -> Self {
        Self {
            genome_length,
            current_time: 0,
            tables: Tables::default(),
            node_heap: NodeHeap::default(),
            free_nodes: vec![],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct AncestryIntersection {
    left: i64,
    right: i64,
    node: Node,
    unary_mapping: Option<Node>,
}

struct Overlapper<'q> {
    queue: &'q [AncestryIntersection],
    overlaps: Vec<AncestryIntersection>,
    current_overlap: usize,
    num_overlaps: usize,
    left: i64,
    right: i64,
}

impl<'q> Overlapper<'q> {
    fn new(queue: &'q [AncestryIntersection]) -> Self {
        let num_overlaps = if queue.is_empty() { 0 } else { queue.len() - 1 };
        let right = if num_overlaps > 0 {
            queue[0].right
        } else {
            i64::MAX
        };
        let left = i64::MAX;
        Self {
            queue,
            left,
            right,
            num_overlaps,
            current_overlap: 0,
            overlaps: vec![],
        }
    }

    fn calculate_next_overlap_set(&mut self) -> Option<(i64, i64, &[AncestryIntersection])> {
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
            Some((self.left, self.right, &self.overlaps))
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
                Some((self.left, self.right, &self.overlaps))
            } else {
                None
            }
        }
    }
}

#[derive(Default)]
struct TempBuffers {
    edges: Vec<Edge>,
    ancestry: Ancestry,
    children: Vec<Node>,
}

impl TempBuffers {
    fn clear(&mut self) {
        self.edges.clear();
        self.ancestry.clear();
        self.children.clear();
    }

    fn push_edge(&mut self, left: i64, right: i64, child: Node) {
        self.edges.push(Edge { left, right, child })
    }

    fn sort_edges(&mut self) {
        self.edges.sort_unstable_by_key(|e| (e.child, e.left));
    }
}

fn ancestry_intersection(
    parent: Node,
    edges: &Edges,
    ancestry: &[Ancestry],
    parents: &mut [Vec<Node>],
    queue: &mut Vec<AncestryIntersection>,
) {
    queue.clear();
    for ((&eleft, &eright), &node) in edges
        .left
        .iter()
        .zip(edges.right.iter())
        .zip(edges.child.iter())
    {
        // NOTE: we are now doing a linear search PER INPUT EDGE,
        // which is redundant. It will be more efficient to cache the child
        // value and then iterate over them at the end.
        if let Some(index) = parents[node.as_index()].iter().position(|&x| x == parent) {
            let _ = parents[node.as_index()].swap_remove(index);
        }
        let anode = &ancestry[node.as_index()];
        for ((&aleft, &aright), &unary_mapping) in anode
            .left
            .iter()
            .zip(anode.right.iter())
            .zip(anode.unary_mapping.iter())
        {
            if eright > aleft && aright > eleft {
                let left = std::cmp::max(eleft, aleft);
                let right = std::cmp::min(eright, aright);
                queue.push(AncestryIntersection {
                    left,
                    right,
                    node,
                    unary_mapping,
                })
            } else if aleft >= eright {
                // Ancestry must be sorted by left coordinat,
                // meaning no more segments can possibly overlap
                // NOTE: could also do this via take_while in the zip steps.
                break;
            }
        }
    }
    queue.sort_unstable_by_key(|x| x.left);
    if !queue.is_empty() {
        queue.push(AncestryIntersection {
            left: i64::MAX,
            right: i64::MAX,
            node: Node(usize::MAX),
            unary_mapping: None,
        });
    }
}

// TODO: needs temp edge and temp ancestry as inputs?
//
// # Some design notes
//
// ## Definition of an ancestry change
//
// 1. Change in left OR right coordinate.
// 2. Change from coalescent to unary status of a segment.
// 3. Change from unary to coao status of a segment.
//    Is this even possible?
// 4. Total loss of a segment.
//
// Squashing will be important:
//
// It is conceivable that overlaps squash down into an unchanged
// input segment. We do NOT want to label this as a change.
//
// ## Avoiding queuing ALL parents when a change occurs.
//
// It is tempting to simply pass all of node's parents into the
// queue.
// An alternative is:
//
// 1. cache intervals that have changed.
// 2. When done, look for parent edge / changed interval overlap
//    and add overlapping parents into the queue.
fn process_queued_node(
    node: Node,
    queue: &[AncestryIntersection],
    buffers: &mut TempBuffers,
    graph: &mut Graph,
) -> bool {
    // println!("visiting node {node:?}");
    let mut overlapper = Overlapper::new(queue);
    let mut current_overlaps = overlapper.calculate_next_overlap_set();
    let mut changed = false;
    let mut input_ancestry = 0_usize;
    let input_ancestry_len = graph.tables.ancestry[node.as_index()].len();
    while let Some((left, right, overlaps)) = current_overlaps {
        while input_ancestry < input_ancestry_len
            && graph.tables.ancestry[node.as_index()].left[input_ancestry] > right
        {
            changed = true;
            input_ancestry += 1;
        }
        let (input_left, input_right, input_unary) =
            graph.tables.ancestry[node.as_index()].ancestry(input_ancestry);
        if left != input_left || right != input_right {
            changed = true;
        }
        // println!("{left},{right},{overlaps:?}");
        if overlaps.len() == 1 {
            let unary_mapping = match overlaps[0].unary_mapping {
                // Propagate the unary mapping up the graph
                Some(u) => Some(u),
                // The unary mapping becomes the overlapped child node
                None => Some(overlaps[0].node),
            };
            // println!("unary {unary_mapping:?} <-> {input_unary:?}");
            if unary_mapping != input_unary {
                changed = true;
            }
            buffers.ancestry.push(left, right, unary_mapping);
        } else {
            for o in overlaps.iter() {
                let child = match o.unary_mapping {
                    Some(u) => u,
                    None => o.node,
                };
                buffers.push_edge(left, right, child);

                // Should be faster than a hash for scores of children.
                if !buffers.children.contains(&child) {
                    buffers.children.push(child);
                }
            }
            buffers.ancestry.push(left, right, None);
        }
        current_overlaps = overlapper.calculate_next_overlap_set();
    }

    // There are input segments w/no outputs
    if input_ancestry < input_ancestry_len {
        changed = true;
    }
    changed
}

fn propagate_changes(graph: &mut Graph) {
    let mut queue = vec![];
    let mut buffers = TempBuffers::default();
    let mut visited = 0;
    while let Some(node) = graph.node_heap.pop() {
        visited += 1;
        // println!("processing {node:?}");
        graph.node_heap.queued_nodes.remove(&node);
        ancestry_intersection(
            node,
            &graph.tables.edges[node.as_index()],
            &graph.tables.ancestry,
            &mut graph.tables.parents,
            &mut queue,
        );
        if queue.is_empty() {
            // FIXME: this behavior must change if node
            // is a "sample" or "alive".
            // println!("node {node:?} has no overlaps");
            // TODO: this logic can be a separate fn

            #[cfg(debug_assertions)]
            for &c in graph.tables.edges[node.as_index()].child.iter() {
                assert!(!graph.tables.parents[c.as_index()].contains(&node))
            }
            // FIXME: remove b/c this is only for testing
            for &c in graph.tables.children[node.as_index()].iter() {
                assert!(!graph.tables.parents[c.as_index()].contains(&node))
            }

            // clearing edges is the "mark" of a node
            // that is not part of the graph
            graph.tables.edges[node.as_index()].clear();
            graph.tables.ancestry[node.as_index()].clear();

            graph.tables.children[node.as_index()].clear();
            // queue all parents for processing
            for &parent in graph.tables.parents[node.as_index()].iter() {
                enqueue_parent(parent, &graph.tables.nodes.birth_time, &mut graph.node_heap)
            }
            graph.tables.parents[node.as_index()].clear();
            // this node can be recycled
            graph.free_nodes.push(node.as_index());
        } else {
            let changed = process_queued_node(node, &queue, &mut buffers, graph);
            assert!(!graph.tables.ancestry[node.as_index()].is_empty());
            // println!("{node:?} -> {changed}");

            // TODO: the next steps should be a new fn
            if buffers.edges.is_empty() {
                // Node has gone extinct
                graph.free_nodes.push(node.as_index());
            }
            buffers.sort_edges();
            {
                let node_edges = &mut graph.tables.edges[node.as_index()];
                node_edges.clear();
                let mut last_right: Option<i64> = None;
                let mut last_child: Option<Node> = None;
                for edge in buffers.edges.iter() {
                    let to_squash = if let Some(lright) = last_right {
                        if lright == edge.left && last_child.unwrap() == edge.child {
                            Some(lright)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    if let Some(right) = to_squash {
                        let len = node_edges.right.len() - 1;
                        node_edges.right[len] = edge.right;
                    } else {
                        node_edges.left.push(edge.left);
                        node_edges.right.push(edge.right);
                        node_edges.child.push(edge.child);
                    }
                    last_right = Some(edge.right);
                    last_child = Some(edge.child);
                }
            }
            std::mem::swap(
                &mut graph.tables.ancestry[node.as_index()],
                &mut buffers.ancestry,
            );
            for &c in buffers.children.iter() {
                debug_assert!(!graph.tables.parents[c.as_index()].contains(&node));
                graph.tables.parents[c.as_index()].push(node);
            }
            std::mem::swap(
                &mut graph.tables.children[node.as_index()],
                &mut buffers.children,
            );

            // TODO: parent queuing should be a separate fn
            if changed {
                debug_assert_ne!(graph.tables.ancestry[node.as_index()], buffers.ancestry);
                for &parent in graph.tables.parents[node.as_index()].iter() {
                    enqueue_parent(parent, &graph.tables.nodes.birth_time, &mut graph.node_heap)
                }
            }

            #[cfg(debug_assertions)]
            if !changed {
                debug_assert_eq!(graph.tables.ancestry[node.as_index()], buffers.ancestry);
            }
            buffers.clear();
        }
    }
    //println!("visited {visited}");
}

fn enqueue_parent(parent: Node, birth_time: &[i64], node_heap: &mut NodeHeap) {
    if !node_heap.queued_nodes.contains(&parent) {
        node_heap.queued_nodes.insert(parent);
        node_heap.node_queue.push(QueuedNode {
            node: parent,
            birth_time: birth_time[parent.as_index()],
        })
    }
}

impl Graph {
    pub fn add_birth(&mut self) -> Node {
        if let Some(index) = self.free_nodes.pop() {
            self.tables
                .recycle_index_as_birth_node(index, self.current_time, self.genome_length);
            Node(index)
        } else {
            Node(
                self.tables
                    .add_new_birth_node(self.current_time, self.genome_length),
            )
        }
    }

    // NOTE: separating this out allows it to be called once
    // each time a Node is chosen as a parent rather than
    // once per segment per time chosen as a parent.
    pub fn enqueue_parent(&mut self, parent: Node) {
        let h = &mut self.node_heap;
        enqueue_parent(parent, &self.tables.nodes.birth_time, &mut self.node_heap)
    }

    // TODO: validate parent/child birth times?
    pub fn record_transmission(&mut self, left: i64, right: i64, parent: Node, child: Node) {
        let e = &mut self.tables.edges[parent.as_index()];
        e.left.push(left);
        e.right.push(right);
        e.child.push(child);
    }
}

#[cfg(test)]
fn edges_contains(edges: &Edges, left: i64, right: i64, child: Node) -> bool {
    let lslice = edges.left.as_slice();
    let mut offset = 0_usize;
    let mut position = lslice.iter().position(|&x| x == left);

    while let Some(p) = position {
        if edges.right[offset + p] == right && edges.child[offset + p] == child {
            return true;
        }
        offset += p + 1;
        position = lslice[offset..].iter().position(|&x| x == left)
    }

    false
}

#[cfg(test)]
fn ancestry_contains(
    ancestry: &Ancestry,
    left: i64,
    right: i64,
    unary_mapping: Option<Node>,
) -> bool {
    let lslice = ancestry.left.as_slice();
    let mut offset = 0_usize;
    let mut position = lslice.iter().position(|&x| x == left);

    while let Some(p) = position {
        if ancestry.right[offset + p] == right
            && ancestry.unary_mapping[offset + p] == unary_mapping
        {
            return true;
        }
        offset += p + 1;
        position = lslice[offset..].iter().position(|&x| x == left)
    }

    false
}

#[cfg(test)]
fn validate_reachable_nodes(graph: &Graph, alive: &[Node]) {
    let mut node_heap = NodeHeap::default();
    for &node in alive {
        for &parent in graph.tables.parents[node.as_index()].iter() {
            enqueue_parent(parent, &graph.tables.nodes.birth_time, &mut node_heap);
        }
    }
    let mut reachable = vec![];
    let mut edges = 0;
    let mut ancestry = 0;
    while let Some(node) = node_heap.pop() {
        assert!(!reachable.contains(&node));
        reachable.push(node);
        edges += graph.tables.edges[node.as_index()].left.len();
        ancestry += graph.tables.ancestry[node.as_index()].left.len();
        for &child in graph.tables.children[node.as_index()].iter() {
            assert_eq!(
                graph.tables.children[node.as_index()]
                    .iter()
                    .filter(|c| **c == child)
                    .count(),
                1
            );
            assert!(
                graph.tables.nodes.birth_time[child.as_index()]
                    > graph.tables.nodes.birth_time[node.as_index()]
            )
        }
        for &parent in graph.tables.parents[node.as_index()].iter() {
            assert_eq!(
                graph.tables.parents[node.as_index()]
                    .iter()
                    .filter(|p| **p == parent)
                    .count(),
                1
            );
            enqueue_parent(parent, &graph.tables.nodes.birth_time, &mut node_heap);
        }
    }
    for (i, edges) in graph.tables.edges.iter().enumerate() {
        if !edges.is_empty() {
            assert!(reachable.contains(&Node(i)));
            assert!(!graph.tables.children[i].is_empty());
            for &c in edges.child.iter() {
                assert!(graph.tables.parents[c.as_index()].contains(&Node(i)));
            }
            let mut parents_copy = graph.tables.parents.clone();
            let mut queue = vec![];
            ancestry_intersection(
                Node(i),
                &graph.tables.edges[i],
                &graph.tables.ancestry,
                &mut parents_copy,
                &mut queue,
            );
            assert!(!queue.is_empty());
            for o in queue.iter().take(queue.len() - 1) {
                assert!(edges.child.contains(&o.node));
            }
        }
    }
    println!(
        "{} {} {}",
        (edges as f64) / reachable.len() as f64,
        (ancestry as f64) / reachable.len() as f64,
        reachable.len()
    )
}

#[cfg(test)]
fn haploid_wf(popsize: usize, ngenerations: i64, genome_length: i64, seed: u64) -> Graph {
    use rand::Rng;
    use rand::SeedableRng;

    let mut graph = Graph::new(genome_length);
    let mut parents = vec![];
    for _ in 0..popsize {
        parents.push(graph.add_birth())
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let sample_parent = rand::distributions::Uniform::new(0, popsize);
    let sample_breakpoint = rand::distributions::Uniform::new(1, genome_length);
    let mut children = vec![];

    for gen in 0..ngenerations {
        //println!("{gen}");
        graph.current_time += 1;
        for _ in 0..popsize {
            let child = graph.add_birth();
            children.push(child);
            let left_parent = parents[rng.sample(sample_parent)];
            let right_parent = parents[rng.sample(sample_parent)];
            let breakpoint = rng.sample(sample_breakpoint);
            graph.enqueue_parent(left_parent);
            graph.enqueue_parent(right_parent);
            graph.record_transmission(0, breakpoint, left_parent, child);
            graph.record_transmission(breakpoint, genome_length, right_parent, child);
        }
        propagate_changes(&mut graph);

        validate_reachable_nodes(&graph, &children);

        for &i in &children {
            // mark them as "dead".
            graph.enqueue_parent(i);
        }

        std::mem::swap(&mut parents, &mut children);
        children.clear();
    }

    graph
}

#[cfg(test)]
mod single_tree_tests {
    use super::*;

    // This test starts with some slightly hokey data
    // in order to test our unary propagation logic in a
    // straightforward way.
    //
    //   0
    //  ---
    //  1 2
    //  | |
    //  | 3
    //  | |
    //  4 5
    #[test]
    fn test0() {
        let mut graph = Graph::new(100);
        graph.tables.nodes.birth_time = vec![0, 1, 1, 2, 3, 3];
        graph.tables.edges.push(Edges {
            left: vec![0, 0],
            right: vec![100, 100],
            child: vec![Node(1), Node(2)],
        });
        graph.tables.edges.push(Edges {
            left: vec![0],
            right: vec![100],
            child: vec![Node(4)],
        });
        graph.tables.edges.push(Edges {
            left: vec![0],
            right: vec![100],
            child: vec![Node(3)],
        });
        graph.tables.edges.push(Edges {
            left: vec![0],
            right: vec![100],
            child: vec![Node(5)],
        });
        graph.tables.edges.push(Edges {
            left: vec![],
            right: vec![],
            child: vec![],
        });
        graph.tables.edges.push(Edges {
            left: vec![],
            right: vec![],
            child: vec![],
        });

        graph.tables.ancestry.push(Ancestry {
            left: vec![0],
            right: vec![100],
            unary_mapping: vec![None],
        });
        graph.tables.ancestry.push(Ancestry {
            left: vec![0],
            right: vec![100],
            unary_mapping: vec![None],
        });
        graph.tables.ancestry.push(Ancestry {
            left: vec![0],
            right: vec![100],
            unary_mapping: vec![Some(Node(3))],
        });
        graph.tables.ancestry.push(Ancestry {
            left: vec![0],
            right: vec![100],
            unary_mapping: vec![None],
        });
        graph.tables.ancestry.push(Ancestry {
            left: vec![0],
            right: vec![100],
            unary_mapping: vec![None],
        });
        graph.tables.ancestry.push(Ancestry {
            left: vec![0],
            right: vec![100],
            unary_mapping: vec![None],
        });

        for _ in 0..graph.tables.nodes.birth_time.len() {
            graph.tables.parents.push(vec![]);
            graph.tables.children.push(vec![]);
        }
        for node in [1, 2] {
            graph.tables.children[0].push(Node(node));
            graph.tables.parents[node].push(Node(0));
        }
        for (node, unary_child) in [(1_usize, 4), (2, 3), (3, 5)] {
            graph.tables.children[node].push(Node(unary_child));
            graph.tables.parents[unary_child].push(Node(node))
        }

        for node in [1, 3] {
            graph.enqueue_parent(Node(node))
        }

        propagate_changes(&mut graph);

        assert_eq!(graph.tables.children[0].len(), 2);
        for node in [4, 5] {
            assert!(graph.tables.children[0].contains(&Node(node)));
            assert_eq!(graph.tables.parents[node].len(), 1);
            assert_eq!(graph.tables.parents[node][0], Node(0));
        }

        for (node, unary) in [(1, 4), (2, 5), (2, 5)] {
            assert!(graph.tables.children[node].is_empty());
            assert!(graph.tables.parents[node].is_empty());
            assert!(graph.tables.edges[node].is_empty());
            assert!(!graph.tables.ancestry[node].is_empty());
            assert_eq!(graph.tables.ancestry[node].len(), 1);
            assert!(graph.tables.ancestry[node].left.contains(&0));
            assert!(graph.tables.ancestry[node].right.contains(&100));
            assert!(graph.tables.ancestry[node]
                .unary_mapping
                .contains(&Some(Node(unary))));
        }

        for node in [1, 2, 3] {
            assert!(graph.free_nodes.contains(&node))
        }
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
        let mut graph = Graph::new(100);
        graph.tables.nodes.birth_time = vec![0, 1, 2, 2, 2];

        graph.tables.edges.push(Edges {
            left: vec![0, 0],
            right: vec![100, 100],
            child: vec![Node(4), Node(1)],
        });
        graph.tables.edges.push(Edges {
            left: vec![0, 0],
            right: vec![100, 100],
            child: vec![Node(3), Node(2)],
        });
        for _ in 0..3 {
            graph.tables.edges.push(Edges::default());
        }

        for _ in 0..graph.tables.nodes.birth_time.len() {
            graph.tables.ancestry.push(Ancestry {
                left: vec![0],
                right: vec![100],
                unary_mapping: vec![None],
            });
        }

        // This node has lost all ancestry,
        // which is the state that we have to propagate.
        graph.tables.ancestry[3].clear();

        graph.tables.children.push(vec![Node(1), Node(4)]);
        graph.tables.children.push(vec![Node(2), Node(3)]);
        for _ in 0..3 {
            graph.tables.children.push(vec![]);
        }

        graph.tables.parents.push(vec![]);
        graph.tables.parents.push(vec![Node(0)]);
        graph.tables.parents.push(vec![Node(1)]);
        graph.tables.parents.push(vec![Node(2)]);
        graph.tables.parents.push(vec![Node(0)]);

        graph.enqueue_parent(Node(1));
        propagate_changes(&mut graph);

        assert_eq!(graph.tables.children[0].len(), 2);
        for i in [2, 4] {
            assert!(graph.tables.children[0].contains(&Node(i)));
            assert!(graph.tables.parents[i].contains(&Node(0)))
        }

        assert_eq!(graph.tables.edges[0].len(), 2);
        assert!(edges_contains(&graph.tables.edges[0], 0, 100, Node(4)));
        assert!(edges_contains(&graph.tables.edges[0], 0, 100, Node(2)));

        assert!(graph.tables.edges[1].is_empty());

        assert!(ancestry_contains(
            &graph.tables.ancestry[1],
            0,
            100,
            Some(Node(2))
        ));
        assert!(graph.tables.children[1].is_empty());

        assert!(graph.free_nodes.contains(&1));
    }
}

#[cfg(test)]
mod multi_tree_tests {
    use super::*;

    // Tree 0, span [0,50)
    //    0
    //  -----
    //  1   |
    //  |   2
    // ---  |
    // 3 4  5
    //
    // Tree 1, span [50,100)
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
    fn test0() {
        let mut graph = Graph::new(100);

        graph.tables.nodes.birth_time = vec![0, 1, 2, 3, 3, 3];

        graph.tables.edges.push(Edges {
            left: vec![0, 0, 50, 50],
            right: vec![50, 50, 100, 100],
            child: vec![Node(1), Node(2), Node(1), Node(2)],
        });

        graph.tables.edges.push(Edges {
            left: vec![0, 0, 50],
            right: vec![50, 50, 100],
            child: vec![Node(3), Node(4), Node(5)],
        });

        graph.tables.edges.push(Edges {
            left: vec![0, 50, 50],
            right: vec![50, 100, 100],
            child: vec![Node(5), Node(3), Node(4)],
        });

        for _ in 0..3 {
            graph.tables.edges.push(Edges::default());
        }

        graph.tables.ancestry.push(Ancestry {
            left: vec![0, 50],
            right: vec![50, 100],
            unary_mapping: vec![None, None],
        });
        graph.tables.ancestry.push(Ancestry {
            left: vec![0, 50],
            right: vec![50, 100],
            unary_mapping: vec![None, Some(Node(5))],
        });
        graph.tables.ancestry.push(Ancestry {
            left: vec![0, 50],
            right: vec![50, 100],
            unary_mapping: vec![Some(Node(5)), None],
        });
        for _ in 0..3 {
            graph.tables.ancestry.push(Ancestry::new_sample(100));
        }

        // This will force chages up the gree once we enqueue
        // the parents
        graph.tables.ancestry[5].clear();

        graph.tables.parents.push(vec![]);
        graph.tables.parents.push(vec![Node(0)]);
        graph.tables.parents.push(vec![Node(0)]);
        graph.tables.parents.push(vec![Node(1), Node(2)]);
        graph.tables.parents.push(vec![Node(1), Node(2)]);
        graph.tables.parents.push(vec![Node(1), Node(2)]);

        graph.tables.children.push(vec![Node(1), Node(2)]);
        graph.tables.children.push(vec![Node(3), Node(4), Node(5)]);
        graph.tables.children.push(vec![Node(3), Node(4), Node(5)]);
        for _ in 0..3 {
            graph.tables.children.push(vec![]);
        }

        for &i in graph.tables.parents[5].iter() {
            enqueue_parent(i, &graph.tables.nodes.birth_time, &mut graph.node_heap);
        }
        assert!(!graph.tables.parents[5].is_empty());
        propagate_changes(&mut graph);

        for node in [1, 2] {
            assert_eq!(
                graph.tables.ancestry[node].len(),
                1,
                "{node:?} -> {:?}",
                graph.tables.ancestry[node]
            )
        }

        assert!(
            graph.tables.edges[0].is_empty(),
            "{:?}",
            graph.tables.edges[0]
        );
        assert_eq!(
            graph.tables.ancestry[0].len(),
            2,
            "{:?}",
            graph.tables.ancestry[0]
        );
        assert!(ancestry_contains(
            &graph.tables.ancestry[0],
            0,
            50,
            Some(Node(1))
        ));
        assert!(ancestry_contains(
            &graph.tables.ancestry[0],
            50,
            100,
            Some(Node(2))
        ));
        assert!(graph.tables.children[0].is_empty());
        for node in [0, 2, 1] {
            assert!(graph.tables.parents[node].is_empty());
        }
        for node in [1, 2] {
            assert_eq!(graph.tables.children[node].len(), 2);
            for c in [3, 4] {
                assert!(graph.tables.children[node].contains(&Node(c)))
            }
        }

        assert!(graph.tables.parents[5].is_empty());

        // NOTE: 5 is not in the list because we have
        // manually forced it to be "extinct".
        for node in [0] {
            assert!(graph.free_nodes.contains(&node), "{node}")
        }
        assert_eq!(graph.free_nodes.len(), 1)
    }

    // Tree 0:
    //    0
    //  -----
    //  1   2
    // ---  |
    // | |  |
    // 3 4  5
    //
    //
    //
    // Tree 1:
    //
    //    0
    //  -----
    //  1   2
    // ---  |
    // | |  |
    // 3 4  5
    //      |
    //      6
    //
    // 3, 4, 5 "die", with 3/4 failing to reproduce
    // 6 is a birth, but only on part of the tree.
    // (Hokey, but gets the job done for testing.)
    //
    // The death of nodes 3 and 4 must trigger an "ancestry loss",
    // eliminating that subtree.
    // In the end, nodes 0 thru 5 should be extinct and set up for recycling.
    // All nodes ancestral to 6 should have a unary mapping to 6 on [0,50)
    // and no edges
    #[test]
    fn test3() {
        let mut graph = Graph::new(100);

        graph.tables.nodes.birth_time = vec![0, 1, 1, 2, 2, 2, 3];

        graph.tables.edges.push(Edges {
            left: vec![0, 0, 50, 50],
            right: vec![50, 50, 100, 100],
            child: vec![Node(1), Node(2), Node(1), Node(2)],
        });
        graph.tables.edges.push(Edges {
            left: vec![0, 0],
            right: vec![100, 100],
            child: vec![Node(3), Node(4)],
        });
        graph.tables.edges.push(Edges {
            left: vec![0],
            right: vec![100],
            child: vec![Node(5)],
        });
        graph.tables.edges.push(Edges::default());
        graph.tables.edges.push(Edges::default());
        graph.tables.edges.push(Edges {
            left: vec![50],
            right: vec![100],
            child: vec![Node(6)],
        });
        graph.tables.edges.push(Edges::default());

        graph.tables.ancestry.push(Ancestry::new_sample(100));
        graph.tables.ancestry.push(Ancestry::new_sample(100));
        graph.tables.ancestry.push(Ancestry::new_sample(100));
        graph.tables.ancestry.push(Ancestry::new_sample(100));
        graph.tables.ancestry.push(Ancestry::new_sample(100));
        graph.tables.ancestry.push(Ancestry::new_sample(100));
        graph.tables.ancestry.push(Ancestry::new_sample(100));

        graph.tables.parents.push(vec![]);
        graph.tables.parents.push(vec![Node(0)]);
        graph.tables.parents.push(vec![Node(0)]);
        graph.tables.parents.push(vec![Node(1)]);
        graph.tables.parents.push(vec![Node(1)]);
        graph.tables.parents.push(vec![Node(2)]);
        graph.tables.parents.push(vec![Node(5)]);

        graph.tables.children.push(vec![Node(1), Node(2)]);
        graph.tables.children.push(vec![Node(3), Node(4)]);
        graph.tables.children.push(vec![Node(5)]);
        graph.tables.children.push(vec![]);
        graph.tables.children.push(vec![]);
        graph.tables.children.push(vec![Node(6)]);
        graph.tables.children.push(vec![]);

        for node in [3, 4, 5] {
            graph.enqueue_parent(Node(node));
        }

        propagate_changes(&mut graph);

        assert!(graph.tables.parents[6].is_empty());
        assert_eq!(graph.tables.ancestry[6].len(), 1);
        assert!(ancestry_contains(&graph.tables.ancestry[6], 0, 100, None));

        for node in [0, 1, 3, 4, 2, 5] {
            assert!(graph.free_nodes.contains(&node));
        }
        assert_eq!(graph.free_nodes.len(), 6);

        for node in [1, 3, 4] {
            assert!(graph.tables.children[node].is_empty());
            assert!(graph.tables.parents[node].is_empty());
            assert!(graph.tables.edges[node].is_empty());
            assert!(graph.tables.ancestry[node].is_empty());
        }
        for node in [2, 5, 0] {
            assert!(graph.tables.children[node].is_empty());
            assert!(graph.tables.parents[node].is_empty());
            assert!(graph.tables.edges[node].is_empty());
            assert_eq!(graph.tables.ancestry[node].len(), 1, "{node:?}");
            assert!(ancestry_contains(
                &graph.tables.ancestry[node],
                50,
                100,
                Some(Node(6))
            ))
        }
    }
}

#[cfg(test)]
mod haploid_wf_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_10_individuals(seed in 0..u64::MAX) {
            let g = haploid_wf(10, 50, 10000000, seed);
        }
    }

    #[ignore]
    #[test]
    fn test_1000_individuals() {
        let g = haploid_wf(1000, 1000, 10000000, 161363643);
    }
}

#[cfg(test)]
mod design_list_difference_calculations {
    #[derive(PartialEq, Eq, Copy, Clone, Debug)]
    struct Interval {
        left: i64,
        right: i64,
    }

    impl Interval {
        fn new(left: i64, right: i64) -> Self {
            Self { left, right }
        }
    }

    fn interval_delta(a: &[Interval], b: &[Interval]) -> Vec<Interval> {
        let mut rv = vec![];

        let mut ai = 0_usize;
        let mut bi = 0_usize;
        while ai < a.len() && bi < b.len() {
            while ai < a.len() && a[ai].right < b[bi].left {
                // segments entirely lost from a b4 anything in b exists
                rv.push(a[ai]);
                ai += 1;
            }
            if ai >= a.len() {
                break;
            }

            let aleft = a[ai].left;
            let aright = a[ai].right;
            let bleft = b[bi].left;
            let bright = b[bi].right;

            if aright > bleft && bright > aleft {
                if aleft != bleft {
                    rv.push(Interval::new(
                        std::cmp::min(aleft, bleft),
                        std::cmp::max(aleft, bleft),
                    ))
                }
                if aright != bright {
                    rv.push(Interval::new(
                        std::cmp::min(aright, bright),
                        std::cmp::max(aright, bright),
                    ))
                }
            }
            ai += 1;
            bi += 1;
        }

        // segments at the end of a not found in b
        for i in &a[ai..] {
            rv.push(*i)
        }

        rv
    }

    fn validate_delta_contents(delta: &[Interval], expected: &[(i64, i64)]) {
        assert_eq!(delta.len(), expected.len(), "{delta:?}");
        for e in expected {
            assert!(delta.contains(&Interval::new(e.0, e.1)));
        }
    }

    #[test]
    fn test0() {
        let a = vec![Interval::new(0, 10), Interval::new(10, 100)];
        let b = vec![Interval::new(6, 10), Interval::new(50, 73)];

        let c = interval_delta(&a, &b);
        validate_delta_contents(&c, &[(0, 6), (10, 50), (73, 100)]);
    }

    #[test]
    fn test1() {
        let a = vec![Interval::new(0, 10), Interval::new(10, 100)];
        let b = vec![Interval::new(50, 73)];

        let c = interval_delta(&a, &b);
        validate_delta_contents(&c, &[(0, 10), (10, 50), (73, 100)]);
    }

    #[test]
    fn test2() {
        let a = vec![
            Interval::new(0, 10),
            Interval::new(40, 80),
            Interval::new(90, 100),
        ];
        let b = vec![Interval::new(50, 73)];

        let c = interval_delta(&a, &b);
        validate_delta_contents(&c, &[(0, 10), (40, 50), (73, 80), (90, 100)]);
    }
}
