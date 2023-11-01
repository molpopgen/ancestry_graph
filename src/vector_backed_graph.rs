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

#[derive(Default)]
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
}

#[derive(Default)]
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

#[derive(Clone, Copy)]
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

fn ancestry_intersection(
    parent: Node,
    edges: &Edges,
    ancestry: &[Ancestry],
    parents: &mut [Vec<Node>],
    queue: &mut Vec<AncestryIntersection>,
) {
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

fn process_queued_node(node: Node, queue: &[AncestryIntersection], graph: &mut Graph) {
    // needs temp edge and temp ancestry as inputs?
    let mut overlapper = Overlapper::new(queue);
    todo!()
}

fn propagate_changes(graph: &mut Graph) {
    let mut queue = vec![];
    while let Some(node) = graph.node_heap.pop() {
        graph.node_heap.queued_nodes.remove(&node);
        ancestry_intersection(
            node,
            &graph.tables.edges[node.as_index()],
            &graph.tables.ancestry,
            &mut graph.tables.parents,
            &mut queue,
        );
        if queue.is_empty() {
            // Delete node from parents of all children.
            // Clear out children
            // Recycle the node id
            todo!()
        } else {
            process_queued_node(node, &queue, graph);
        }
        todo!()
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
        if !h.queued_nodes.contains(&parent) {
            h.node_queue.push(QueuedNode {
                node: parent,
                birth_time: self.tables.nodes.birth_time[parent.as_index()],
            })
        }
    }

    // TODO: validate parent/child birth times?
    pub fn record_transmission(&mut self, left: i64, right: i64, parent: Node, child: Node) {
        let e = &mut self.tables.edges[parent.as_index()];
        e.left.push(left);
        e.right.push(right);
        e.child.push(child);
    }
}
