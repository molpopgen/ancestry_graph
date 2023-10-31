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

#[derive(Default)]
struct Edges {
    left: Vec<i64>,
    right: Vec<i64>,
    child: Vec<Node>,
}

#[derive(Default)]
struct Nodes {
    birth_time: Vec<i64>,
    edge_start: Vec<Option<usize>>,
    ancestry_start: Vec<Option<usize>>,
}

#[derive(Default)]
struct Tables {
    ancestry: Ancestry,
    edges: Edges,
    nodes: Nodes,
}

#[derive(Default)]
struct Graph {
    tables: Tables,
    simplified_tables: Tables,
    node_heap: NodeHeap,
    simplified_node_map: Vec<Option<Node>>,
    current_time: i64,
}

impl Graph {
    pub fn add_birth(&mut self) -> Node {
        self.tables.nodes.birth_time.push(self.current_time);
        self.tables.nodes.edge_start.push(None);
        self.tables.nodes.ancestry_start.push(None);
        Node(self.tables.nodes.birth_time.len() - 1)
    }
}
