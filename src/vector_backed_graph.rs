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
    node: Vec<Node>,
    // NOTE: it is possible that we will need to HASH
    // a Node -> birth time mapping.
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
}
