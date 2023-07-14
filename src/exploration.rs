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

    //     0
    //  -------
    //  |     |
    //  1     2
    //  |     ---
    //  |     | |
    //  3     4 5
    pub struct Topology1 {
        pub node0: Node,
        pub node1: Node,
        pub node2: Node,
        pub node3: Node,
        pub node4: Node,
        pub node5: Node,
        pub graph: Graph,
    }

    impl Topology1 {
        pub fn new() -> Self {
            let mut birth_time = vec![];
            for i in [0_i64, 1, 1, 2, 2, 2] {
                birth_time.push(Some(i))
            }
            let (node0, node1, node2, node3, node4, node5) =
                (Node(0), Node(1), Node(2), Node(3), Node(4), Node(5));
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
            ancestry.push(vec![Ancestry {
                segment,
                node: node5,
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
                vec![
                    Edge {
                        segment,
                        child: node4,
                    },
                    Edge {
                        segment,
                        child: node5,
                    },
                ],
                vec![],
                vec![],
                vec![],
            ];
            assert_eq!(birth_time.len(), 6);
            assert_eq!(edges.len(), 6);
            assert_eq!(ancestry.len(), 6);

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
                node5,
                graph,
            }
        }
    }

    // Tree 1:
    //     0
    //  -------
    //  |     |
    //  1     2
    //  |     -----
    //  |     | | |
    //  3     4 5 6
    //
    // Tree 2:
    //     0
    //  -------
    //  |     |
    //  1     2
    //  ---   ---
    //  | |   | |
    //  4 6   3 5
    pub struct Topology2 {
        pub node0: Node,
        pub node1: Node,
        pub node2: Node,
        pub node3: Node,
        pub node4: Node,
        pub node5: Node,
        pub node6: Node,
        pub graph: Graph,
    }

    impl Topology2 {
        pub fn new() -> Self {
            let mut birth_time = vec![];
            for i in [0_i64, 1, 1, 2, 2, 2, 2] {
                birth_time.push(Some(i))
            }
            let (node0, node1, node2, node3, node4, node5, node6) = (
                Node(0),
                Node(1),
                Node(2),
                Node(3),
                Node(4),
                Node(5),
                Node(6),
            );
            let mut ancestry = vec![];
            let segment = Segment { left: 0, right: 50 };
            for i in 0..birth_time.len() {
                ancestry.push(vec![Ancestry {
                    segment,
                    node: Node(i),
                }]);
            }
            let lsegment = Segment { left: 0, right: 25 };
            let rsegment = Segment {
                left: 25,
                right: 50,
            };
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
                vec![
                    Edge {
                        segment: lsegment,
                        child: node3,
                    },
                    Edge {
                        segment: rsegment,
                        child: node6,
                    },
                    Edge {
                        segment: rsegment,
                        child: node4,
                    },
                ],
                vec![
                    Edge {
                        segment: lsegment,
                        child: node5,
                    },
                    Edge {
                        segment: lsegment,
                        child: node6,
                    },
                    Edge {
                        segment: lsegment,
                        child: node4,
                    },
                    Edge {
                        segment: rsegment,
                        child: node3,
                    },
                    Edge {
                        segment: rsegment,
                        child: node5,
                    },
                ],
                vec![],
                vec![],
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
                node5,
                node6,
                graph,
            }
        }
    }
}

#[cfg(test)]
mod test_standard_case {
    use super::*;

    // NOTE: ancestry to ancestry overlap is flawed:
    // * unary ancestry is used to mutation simplification
    // * therefore, we will FALSELY detect overlaps later
    //
    // POSSIBLE SOLUTION: label ancestry as unary or overlap,
    // and use that in combination w/options re: retaining unary status
    //
    // Oh, we actually know this...the output node is either == current
    // node (overlap) or not (unary)
    fn build_queue(graph: &Graph, node: Node, children: &[usize]) -> (Vec<Ancestry>, Vec<usize>) {
        println!("{node:?} <-- {children:?}");
        let mut q = vec![];

        let mut edges_not_found = vec![];
        for (idx, e) in graph.ancestry[node.as_index()].iter().enumerate() {
            let mut found = false;
            //while let Some(child) = children.pop() {
            for &child in children.iter() {
                println!("child {child} has ancestry {:?}", graph.ancestry[child]);
                for a in graph.ancestry[child].iter() {
                    println!("child segment is {a:?}");
                    if a.segment.right > e.segment.left && e.segment.right > a.segment.left {
                        let left = std::cmp::max(e.segment.left, a.segment.left);
                        let right = std::cmp::max(e.segment.right, a.segment.right);
                        if a.node == e.node {
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
                println!("parental node {e:?} has no overlaps");
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

    fn propagate_changes(
        nodes: &[Node],
        graph: Graph,
        parents: &mut Vec<Option<Vec<usize>>>,
        children_to_check: &mut Vec<Vec<usize>>,
    ) -> Graph {
        let mut graph = graph;

        //  todo!("it seems like we should be able to check if an existing anc segment changes and, if so, queue the parent for updating?");
        // backwards in time thru nodes.
        for node in nodes.iter().rev() {
            println!("{children_to_check:?}");
            println!("{parents:?}");
            let (q, lost_edges) = build_queue(&graph, *node, &children_to_check[node.as_index()]);
            println!("q =  {q:?}");
            // The latter case is "no overlaps with children" == extinct node
            if q.len() == 1 || q.is_empty() {
                // FIXME: don't do this -- it is bad for mutation
                // simplification
                graph.ancestry[node.as_index()].clear();
                for edge in graph.edges[node.as_index()].iter() {
                    if let Some(node_parents) = &parents[node.as_index()] {
                        for &parent in node_parents.iter() {
                            // NOTE: unclear on the utility of this...
                            // The ONE benefit is that it will let us
                            // NOT CLEAR the unary ancestry from an extinct node,
                            // making mutation simplification more feasible.
                            if let Some(needle) = children_to_check[parent]
                                .iter()
                                .position(|x| x == &node.as_index())
                            {
                                children_to_check[parent].remove(needle);
                            }

                            if !children_to_check[parent].contains(&edge.child.as_index()) {
                                children_to_check[parent].push(edge.child.as_index());
                            }
                        }
                    }
                    if let Some(cparents) = &mut parents[edge.child.as_index()] {
                        if let Some(needle) = cparents.iter().position(|x| x == &node.as_index()) {
                            cparents.remove(needle);
                        }
                    }
                }
                // Set node status to "extinct"
                graph.edges[node.as_index()].clear();
                graph.birth_time[node.as_index()] = None;
                parents[node.as_index()] = None;
            } else if q.len() > 1 {
                println!("overlaps {node:?}: {q:?}");
                assert!(lost_edges.windows(2).all(|w| w[0] < w[1]));

                // Remove lost edges.
                for lost in lost_edges.iter().rev() {
                    graph.edges[node.as_index()].remove(*lost);
                }

                // Insert overlaps.
                // NOTE: this will do bad things if a node
                // is retained as an edge -- we won't move it,
                // and we'll re-push it.
                println!("current edges = {:?}", graph.edges[node.as_index()]);
                graph.edges[node.as_index()].clear();
                for a in q {
                    println!("Adding edge to node {:?}", a.node);
                    graph.edges[node.as_index()].push(Edge {
                        segment: a.segment,
                        child: a.node,
                    });
                    if let Some(cparents) = &mut parents[a.node.as_index()] {
                        if !cparents.contains(&node.as_index()) {
                            cparents.push(node.as_index());
                        }
                    }
                }
            }
        }

        graph
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
        let mut parents = vec![
            None,
            Some(vec![0_usize]),
            Some(vec![0_usize]),
            Some(vec![1_usize]),
            Some(vec![2_usize]),
        ];

        graph = propagate_changes(&nodes, graph, &mut parents, &mut children_to_check);

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
        //validate parents
        for node in [3, 4] {
            assert!(parents[node].as_ref().unwrap().contains(&0));
            assert_eq!(parents[node].as_ref().unwrap().len(), 1);
        }
        for node in [0, 1, 2] {
            assert!(parents[node].is_none());
        }
    }

    #[test]
    fn test_topology1() {
        let graph_fixtures::Topology1 {
            node0,
            node1,
            node2,
            node3,
            node4,
            node5,
            mut graph,
        } = graph_fixtures::Topology1::new();

        // NOTE: we have to treat node3/4/5 as "special"
        // because they are births.
        // We skip that for now.
        let nodes = vec![node0, node1, node2];
        let mut children_to_check = vec![
            vec![1_usize, 2_usize],
            vec![3_usize],
            vec![4_usize, 5_usize],
            vec![],
            vec![],
            vec![],
        ];
        assert_eq!(graph.edges[2].len(), 2);

        // Seems we need this in graph!
        let mut parents = vec![
            None,
            Some(vec![0_usize]),
            Some(vec![0_usize]),
            Some(vec![1_usize]),
            Some(vec![2_usize]),
            Some(vec![2_usize]),
        ];

        graph = propagate_changes(&nodes, graph, &mut parents, &mut children_to_check);

        for node in 0..graph.ancestry.len() {
            println!(
                "{node} => {:?}, {:?}",
                graph.ancestry[node], graph.edges[node]
            );
        }
        assert!(graph.ancestry[1].is_empty());
        assert!(!graph.ancestry[3].is_empty());
        assert!(!graph.ancestry[4].is_empty());
        assert!(!graph.ancestry[5].is_empty());

        for node in [1, 3, 4, 5] {
            assert!(graph.edges[node].is_empty());
        }

        // This is the tough case
        assert_eq!(graph.ancestry[2].len(), 1);
        assert_eq!(graph.edges[2].len(), 2);

        assert!(!graph.ancestry[0].is_empty());
        assert_eq!(graph.ancestry[0].len(), 1);
        assert_eq!(graph.edges[0].len(), 2);
        let segment = Segment { left: 0, right: 50 };
        for child in [node3, node2] {
            let edge = Edge { segment, child };
            assert!(
                graph.edges[0].contains(&edge),
                "{edge:?}, {:?}",
                graph.edges[0]
            );
        }
    }

    #[test]
    fn test_topology2() {
        let graph_fixtures::Topology2 {
            node0,
            node1,
            node2,
            node3,
            node4,
            node5,
            node6,
            mut graph,
        } = graph_fixtures::Topology2::new();

        let mut children_to_check = vec![
            vec![1_usize, 2_usize],
            vec![3_usize, 4, 6],
            vec![4_usize, 5_usize, 3, 6],
            vec![],
            vec![],
            vec![],
            vec![],
        ];

        // Seems we need this in graph!
        let mut parents = vec![
            None,
            Some(vec![0_usize]),
            Some(vec![0_usize]),
            Some(vec![1_usize, 2]),
            Some(vec![2_usize, 1]),
            Some(vec![2_usize]),
            Some(vec![2_usize]),
        ];
        let nodes = (0..3).map(Node).collect::<Vec<Node>>();
        graph = propagate_changes(&nodes, graph, &mut parents, &mut children_to_check);

        for node in 0..graph.ancestry.len() {
            println!(
                "{node} => {:?}, {:?}",
                graph.ancestry[node], graph.edges[node]
            );
        }
    }
}
