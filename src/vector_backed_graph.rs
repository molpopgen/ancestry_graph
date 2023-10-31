use crate::Node;
use crate::NodeHash;
use crate::QueuedNode;

#[derive(Default, Debug)]
struct NodeHeap {
    queued_nodes: NodeHash,
    node_queue: std::collections::BinaryHeap<QueuedNode>,
}

#[derive(Default)]
struct Ancestry {
    left: Vec<i64>,
    right: Vec<i64>,
    unary_mapping: Vec<Option<Node>>,
}

impl Ancestry {
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

#[derive(Default)]
struct Graph {
    tables: Tables,
    node_heap: NodeHeap,
    current_time: i64,
    free_nodes: Vec<usize>,
}

impl Graph {
    pub fn add_birth(&mut self) -> Node {
        if let Some(index) = self.free_nodes.pop() {
            self.tables.edges[index].clear();
            self.tables.ancestry[index].clear();
            Node(index)
        } else {
            self.tables.nodes.birth_time.push(self.current_time);
            Node(self.tables.nodes.birth_time.len() - 1)
        }
    }
}
