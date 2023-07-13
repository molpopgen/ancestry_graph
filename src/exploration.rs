use crate::Node;
use crate::Segment;

struct Ancestry {
    segment: Segment,
    node: Node,
}

struct Edge {
    segment: Segment,
    child: Node,
}

pub struct Graph {
    birth_time: Vec<Option<i64>>,
    ancestry: Vec<Vec<Ancestry>>,
    edges: Vec<Vec<Edge>>,
}

#[cfg(test)]
mod test_fixtures {
    use super::*;

    //     0
    //  -------
    //  |     |
    //  1     2
    //  |     |
    //  3     4
    pub struct Topology0 {
        pub node0: Node,
        pub node1: Node,
        pub node2: Node,
        pub node3: Node,
        pub node4: Node,
        pub graph: Graph,
    }

    impl Topology0 {
        pub fn new() -> Self {
            let mut birth_time = vec![];
            for i in [0_i64, 1, 1, 2, 2] {
                birth_time.push(Some(i))
            }
            let (node0, node1, node2, node3, node4) = (Node(0), Node(1), Node(2), Node(3), Node(4));
            let mut ancestry = vec![];
            let segment = Segment { left: 0, right: 50 };
            ancestry.push(vec![Ancestry {
                segment,
                node: node0,
            }]);
            ancestry.push(vec![Ancestry {
                segment,
                node: node1,
            }]);
            ancestry.push(vec![Ancestry {
                segment,
                node: node2,
            }]);
            ancestry.push(vec![Ancestry {
                segment,
                node: node3,
            }]);
            ancestry.push(vec![Ancestry {
                segment,
                node: node4,
            }]);
            let edges = vec![
                vec![
                    Edge {
                        segment,
                        child: node1,
                    },
                    Edge {
                        segment,
                        child: node2,
                    },
                ],
                vec![Edge {
                    segment,
                    child: node3,
                }],
                vec![Edge {
                    segment,
                    child: node4,
                }],
                vec![],
                vec![],
            ];
            let graph = Graph {
                birth_time,
                ancestry,
                edges,
            };

            Self {
                node0,
                node1,
                node2,
                node3,
                node4,
                graph,
            }
        }
    }
}

#[cfg(test)]
mod test_standard_case {
    #[test]
    fn test_topology0() {
        let test_fixtures::Topology0 {
            node0,
            node1,
            node2,
            node3,
            node4,
            graph,
        } = test_fixtures::Topology0::new();
    }
}
