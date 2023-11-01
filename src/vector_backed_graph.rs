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

#[derive(Default, Debug)]
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
    let mut current_overlaps = overlapper.calculate_next_overlap_set();
    let mut temp_edges = Edges::default();
    let mut temp_ancestry = Ancestry::default();
    let mut temp_children: Vec<Node> = vec![];
    while let Some((left, right, ref overlaps)) = current_overlaps {
        println!("{left},{right},{overlaps:?}");
        if overlaps.len() == 1 {
            let unary_mapping = overlaps[0]
                .unary_mapping
                .map_or_else(|| overlaps[0].node, |u| u);
            // TODO: Ancestry should handle
            temp_ancestry.left.push(left);
            temp_ancestry.right.push(right);
            temp_ancestry.unary_mapping.push(Some(unary_mapping));
        } else {
            for o in overlaps.iter() {
                let child = match o.unary_mapping {
                    Some(u) => u,
                    None => o.node,
                };
                // TODO: Edges should handle
                temp_edges.left.push(left);
                temp_edges.right.push(right);
                temp_edges.child.push(child);
                // TODO: Ancestry should handle
                temp_ancestry.left.push(left);
                temp_ancestry.right.push(right);
                temp_ancestry.unary_mapping.push(None);

                // Should be faster than a hash for scores of children.
                if !temp_children.contains(&child) {
                    temp_children.push(child);
                }
            }
        }
        current_overlaps = overlapper.calculate_next_overlap_set();
    }
    std::mem::swap(&mut graph.tables.edges[node.as_index()], &mut temp_edges);
    std::mem::swap(
        &mut graph.tables.ancestry[node.as_index()],
        &mut temp_ancestry,
    );
    for &c in temp_children.iter() {
        debug_assert!(!graph.tables.parents[c.as_index()].contains(&node));
        graph.tables.parents[c.as_index()].push(node);
    }
    std::mem::swap(
        &mut graph.tables.children[node.as_index()],
        &mut temp_children,
    );
    // FIXME: next step is wrong.
    // We should only do this IF ANCESTRY CHANGES
    todo!("need to handle detecting ancestry changes");
    for &parent in graph.tables.parents[node.as_index()].iter() {
        enqueue_parent(parent, &graph.tables.nodes.birth_time, &mut graph.node_heap)
    }
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
            todo!("{node:?}");
        } else {
            process_queued_node(node, &queue, graph);
        }
    }
}

fn enqueue_parent(parent: Node, birth_time: &[i64], node_heap: &mut NodeHeap) {
    if !node_heap.queued_nodes.contains(&parent) {
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
mod single_tree_tests {
    use super::*;

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
            unary_mapping: vec![Some(Node(4))],
        });
        graph.tables.ancestry.push(Ancestry {
            left: vec![0],
            right: vec![100],
            unary_mapping: vec![Some(Node(3))],
        });
        graph.tables.ancestry.push(Ancestry {
            left: vec![0],
            right: vec![100],
            unary_mapping: vec![Some(Node(5))],
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
    }
}
