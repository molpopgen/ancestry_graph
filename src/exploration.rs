use crate::Node;
use crate::Segment;

#[derive(Debug)]
struct Ancestry {
    segment: Segment,
    node: Node,
}

#[derive(Debug, PartialEq, Eq)]
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
mod graph_fixtures {
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
    use super::*;

    fn build_queue(
        graph: &Graph,
        node: Node,
        children: &mut Vec<usize>,
    ) -> (Vec<Ancestry>, Vec<usize>) {
        println!("{node:?} <-- {children:?}");
        let mut q = vec![];

        let mut edges_not_found = vec![];
        for (idx, e) in graph.edges[node.as_index()].iter().enumerate() {
            let mut found = false;
            while let Some(child) = children.pop() {
                for a in graph.ancestry[child].iter() {
                    if a.segment.right > e.segment.left && e.segment.right > a.segment.left {
                        let left = std::cmp::max(e.segment.left, a.segment.left);
                        let right = std::cmp::max(e.segment.right, a.segment.right);
                        if a.node == e.child {
                            found = true;
                        }
                        q.push(Ancestry {
                            segment: Segment { left, right },
                            node: a.node,
                        });
                    }
                }
            }
            if !found {
                edges_not_found.push(idx);
            }
            // Below is "classic tskit" style
            //for a in graph.ancestry[e.child.as_index()].iter() {
            //    if a.segment.right > e.segment.left && e.segment.right > a.segment.left {
            //        let left = std::cmp::max(e.segment.left, a.segment.left);
            //        let right = std::cmp::max(e.segment.right, a.segment.right);
            //        q.push(Ancestry {
            //            segment: Segment { left, right },
            //            node: a.node,
            //        });
            //    }
            //}
        }
        println!("{node:?} <-- {children:?}");
        println!("lost edges = {edges_not_found:?}");

        (q, edges_not_found)
    }

    #[test]
    fn test_topology0() {
        let graph_fixtures::Topology0 {
            node0,
            node1,
            node2,
            node3,
            node4,
            mut graph,
        } = graph_fixtures::Topology0::new();

        // NOTE: we have to treat node3/4 as "special"
        // because they are births.
        // We skip that for now.
        let nodes = vec![node0, node1, node2];
        let mut children_to_check = vec![
            vec![1_usize, 2_usize],
            vec![3_usize],
            vec![4_usize],
            vec![],
            vec![],
        ];

        // Seems we need this in graph!
        let parents = vec![
            None,
            Some(vec![0_usize]),
            Some(vec![0_usize]),
            Some(vec![1_usize]),
            Some(vec![2_usize]),
        ];

        // backwards in time thru nodes.
        for node in nodes.iter().rev() {
            let (q, lost_edges) =
                build_queue(&graph, *node, &mut children_to_check[node.as_index()]);
            // The latter case is "no overlaps with children" == extinct node
            if q.len() == 1 || q.is_empty() {
                graph.ancestry[node.as_index()].clear();
                if let Some(parents) = &parents[node.as_index()] {
                    for edge in graph.edges[node.as_index()].iter() {
                        for &parent in parents {
                            if !children_to_check[parent].contains(&edge.child.as_index()) {
                                children_to_check[parent].push(edge.child.as_index());
                            }
                        }
                    }
                }
                // Set node status to "extinct"
                graph.edges[node.as_index()].clear();
                graph.birth_time[node.as_index()] = None;
            } else if q.len() > 1 {
                assert!(lost_edges.windows(2).all(|w| w[0] < w[1]));

                // Remove lost edges.
                for lost in lost_edges.iter().rev() {
                    graph.edges[node.as_index()].remove(*lost);
                }

                // Insert overlaps.
                // NOTE: this will do bad things if a node
                // is retained as an edge -- we won't move it,
                // and we'll re-push it.
                for a in q {
                    graph.edges[node.as_index()].push(Edge {
                        segment: a.segment,
                        child: a.node,
                    });
                }
            }
        }

        for node in 0..graph.ancestry.len() {
            println!(
                "{node} => {:?}, {:?}",
                graph.ancestry[node], graph.edges[node]
            );
        }
        assert!(graph.ancestry[1].is_empty());
        assert!(graph.ancestry[2].is_empty());
        assert!(!graph.ancestry[3].is_empty());
        assert!(!graph.ancestry[4].is_empty());

        for node in [1, 2, 3, 4] {
            assert!(graph.edges[node].is_empty());
        }

        // This is the tough case
        assert!(!graph.ancestry[0].is_empty());
        assert_eq!(graph.ancestry[0].len(), 1);
        assert_eq!(graph.edges[0].len(), 2);
        let segment = Segment { left: 0, right: 50 };
        for child in [node3, node4] {
            let edge = Edge { segment, child };
            assert!(
                graph.edges[0].contains(&edge),
                "{edge:?}, {:?}",
                graph.edges[0]
            );
        }
    }
}
