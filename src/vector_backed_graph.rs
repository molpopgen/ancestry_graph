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
    nodes: Nodes,
}

struct Graph {
    tables: Tables,
    node_heap: NodeHeap,
    current_time: i64,
    genome_length: i64,
    free_nodes: Vec<usize>,
}

struct AncestryIntersection {
    left: i64,
    right: i64,
    node: Node,
}

fn ancestry_intersection(
    edges: &Edges,
    ancestry: &[Ancestry],
    queue: &mut Vec<AncestryIntersection>,
) {
    for ((&eleft, &eright), &node) in edges
        .left
        .iter()
        .zip(edges.right.iter())
        .zip(edges.child.iter())
    {
        for (&aleft, &aright) in ancestry[node.as_index()]
            .left
            .iter()
            .zip(ancestry[node.as_index()].right.iter())
        {
            if eright > aleft && aright > eleft {
                let left = std::cmp::max(eleft, aleft);
                let right = std::cmp::min(eright, aright);
                queue.push(AncestryIntersection { left, right, node })
            }
        }
    }
    queue.sort_unstable_by_key(|x| x.left);
    if !queue.is_empty() {
        queue.push(AncestryIntersection {
            left: i64::MAX,
            right: i64::MAX,
            node: Node(usize::MAX),
        });
    }
}

fn propagate_changes(graph: &mut Graph) {
    let mut queue = vec![];
    while let Some(node) = graph.node_heap.pop() {
        graph.node_heap.queued_nodes.remove(&node);
        ancestry_intersection(
            &graph.tables.edges[node.as_index()],
            &graph.tables.ancestry,
            &mut queue,
        );
        todo!()
    }
}

impl Graph {
    pub fn add_birth(&mut self) -> Node {
        if let Some(index) = self.free_nodes.pop() {
            self.tables.edges[index].clear();
            self.tables.ancestry[index].clear();
            self.tables.ancestry[index].add_ancestry(0, self.genome_length, None);
            Node(index)
        } else {
            self.tables.nodes.birth_time.push(self.current_time);
            self.tables
                .ancestry
                .push(Ancestry::new_sample(self.genome_length));
            self.tables.edges.push(Edges::default());
            Node(self.tables.nodes.birth_time.len() - 1)
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
