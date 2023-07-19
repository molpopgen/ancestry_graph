use std::sync::Arc;

use crate::Node;
use crate::Segment;

#[derive(Debug, Clone, Copy)]
struct Ancestry {
    segment: Segment,
    node: Node,
    num_overlaps: usize,
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

#[derive(Clone, Copy, Debug)]
enum ChangeType {
    Loss,
    Overlap,
}

#[derive(Clone, Copy, Debug)]
struct AncestryChange {
    segment: Segment,
    mapped_node: Node,
    source_node: Node,
    change_type: ChangeType,
}

#[derive(Debug)]
struct Overlaps<'overlapper> {
    segment: Segment,
    overlaps: &'overlapper [AncestryChange],
}

#[derive(Debug)]
struct AncestryOverlapper {
    queue: Vec<AncestryChange>,
    num_overlaps: usize,
    current_overlap: usize,
    parent: Node,
    left: i64,
    right: i64,
    overlaps: Vec<AncestryChange>,
}

impl AncestryOverlapper {
    fn new(parent: Node, queue: Vec<AncestryChange>) -> Self {
        let mut queue = queue;
        let num_overlaps = queue.len();
        if num_overlaps > 0 {
            queue.push(AncestryChange {
                segment: Segment::sentinel(),
                mapped_node: parent,
                source_node: parent,
                change_type: ChangeType::Overlap,
            });
        }
        let right = if num_overlaps > 0 {
            queue[0].segment.right()
        } else {
            i64::MAX
        };
        let left = i64::MAX;
        Self {
            queue,
            num_overlaps,
            current_overlap: 0,
            parent,
            left,
            right,
            overlaps: vec![],
        }
    }
    fn filter_overlaps(&mut self) {
        self.overlaps.retain(|x| x.segment.right() > self.left);
    }

    fn update_right_from_overlaps(&mut self) {
        self.right = match self
            .overlaps
            .iter()
            .map(|&overlap| overlap.segment.right())
            .min()
        {
            Some(right) => right,
            None => self.right,
        };
    }

    fn calculate_next_overlap_set(&mut self) -> Option<Overlaps> {
        if self.current_overlap < self.num_overlaps {
            self.left = self.right;
            self.filter_overlaps();

            // TODO: this should be a function call
            if self.overlaps.is_empty() {
                self.left = self.queue[self.current_overlap].segment.left();
            }
            self.current_overlap += self
                .queue
                .iter()
                .skip(self.current_overlap)
                .take_while(|x| x.segment.left() == self.left)
                .inspect(|x| {
                    self.right = std::cmp::min(self.right, x.segment.right());
                    println!("pushing {x:?}");
                    self.overlaps.push(**x);
                })
                .count();
            self.update_right_from_overlaps();
            self.right = std::cmp::min(self.right, self.queue[self.current_overlap].segment.left());
            Some(Overlaps {
                segment: Segment::new(self.left, self.right).unwrap(),
                overlaps: self.overlaps.as_slice(),
            })
        } else {
            if !self.overlaps.is_empty() {
                self.left = self.right;
                self.filter_overlaps();
            }
            if !self.overlaps.is_empty() {
                self.update_right_from_overlaps();
                Some(Overlaps {
                    segment: Segment::new(self.left, self.right).unwrap(),
                    overlaps: self.overlaps.as_slice(),
                })
            } else {
                None
            }
        }
    }
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
                num_overlaps: 2,
            }]);
            ancestry.push(vec![Ancestry {
                segment,
                node: node1,
                num_overlaps: 1,
            }]);
            ancestry.push(vec![Ancestry {
                segment,
                node: node2,
                num_overlaps: 1,
            }]);
            ancestry.push(vec![Ancestry {
                segment,
                node: node3,
                num_overlaps: 1,
            }]);
            ancestry.push(vec![Ancestry {
                segment,
                node: node4,
                num_overlaps: 1,
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
                num_overlaps: 2, // a previous ancestor
            }]);
            ancestry.push(vec![Ancestry {
                segment,
                node: node1,
                num_overlaps: 1,
            }]);
            ancestry.push(vec![Ancestry {
                segment,
                node: node2,
                num_overlaps: 1, // A death that will become an overlap
            }]);
            ancestry.push(vec![Ancestry {
                segment,
                node: node3,
                num_overlaps: 1,
            }]);
            ancestry.push(vec![Ancestry {
                segment,
                node: node4,
                num_overlaps: 1,
            }]);
            ancestry.push(vec![Ancestry {
                segment,
                node: node5,
                num_overlaps: 1,
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
        pub lsegment: Segment,
        pub rsegment: Segment,
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
                    num_overlaps: 1,
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
                lsegment,
                rsegment,
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
        changes: &[Vec<AncestryChange>],
    ) -> Vec<AncestryChange> {
        let mut q = vec![];

        let mut current_ancestry_index = 0_usize;
        for e in graph.edges[node.as_index()].iter() {
            // Suss out how many overlaps there are for the each parental anc segment
            // Is that not something that we should just book-keep in the ancestry
            // type?
            while current_ancestry_index < graph.ancestry[node.as_index()].len() {
                let aseg = &graph.ancestry[node.as_index()][current_ancestry_index];
                if aseg.segment.right > e.segment.left && e.segment.right > aseg.segment.left {
                    break;
                }
                current_ancestry_index += 1;
            }
            let aseg = &graph.ancestry[node.as_index()][current_ancestry_index];
            if aseg.segment.right > e.segment.left && e.segment.right > aseg.segment.left {
                println!("edge to self anc overlap");
            }
            for a in changes[e.child.as_index()].iter() {
                if a.source_node == e.child
                    && a.segment.right > e.segment.left
                    && e.segment.right > a.segment.left
                {
                    let left = std::cmp::max(e.segment.left, a.segment.left);
                    let right = std::cmp::max(e.segment.right, a.segment.right);
                    println!("OVERLAP {e:?} => {a:?}");
                    // This could be a gain or a loss...
                    q.push(AncestryChange {
                        segment: Segment { left, right },
                        ..*a
                    });
                    // This is a remapping of
                    // this edge on this interval
                    // In a sense, this is a "gain", but we need
                    // to know how to represent it...
                    if a.mapped_node != e.child {
                        let overlap = AncestryChange {
                            segment: Segment { left, right },
                            mapped_node: a.mapped_node,
                            source_node: node,
                            // This may not be the right enum variant.
                            // Perhaps we need something to represent
                            // a "remapping"
                            change_type: ChangeType::Overlap,
                        };
                        println!("we also have an overlap to handle: {overlap:?}");
                        q.push(overlap);
                    }
                }
            }
        }

        q
    }

    fn process_unary_overlap(
        node: Node,
        overlaps: &Overlaps,
        ancestry_changes: &mut [Vec<AncestryChange>],
        parents: &mut Vec<Option<Vec<usize>>>,
        children_to_check: &mut Vec<Vec<usize>>,
        graph: &mut Graph,
    ) {
        todo!()
    }

    fn process_coalescent_overlap(
        node: Node,
        overlaps: &Overlaps,
        ancestry_changes: &mut [Vec<AncestryChange>],
        parents: &mut Vec<Option<Vec<usize>>>,
        children_to_check: &mut Vec<Vec<usize>>,
        graph: &mut Graph,
    ) {
        todo!()
    }

    fn process_overlaps(
        node: Node,
        overlaps: &Overlaps,
        ancestry_changes: &mut [Vec<AncestryChange>],
        parents: &mut Vec<Option<Vec<usize>>>,
        graph: &mut Graph,
    ) {
        todo!()
    }

    fn propagate_changes(
        nodes: &[Node],
        graph: Graph,
        ancestry_changes: &mut [Vec<AncestryChange>],
        parents: &mut Vec<Option<Vec<usize>>>,
    ) -> Graph {
        let mut graph = graph;

        //  todo!("it seems like we should be able to check if an existing anc segment changes and, if so, queue the parent for updating?");
        // backwards in time thru nodes.
        for node in nodes.iter().rev() {
            println!("{parents:?}");
            println!("changes = {:?}", ancestry_changes[node.as_index()]);
            let q = build_queue(&graph, *node, ancestry_changes);
            println!("q =  {q:?}");
            let mut overlapper = AncestryOverlapper::new(*node, q);
            println!("edges = {:?}", graph.edges[node.as_index()]);
            for loss in ancestry_changes[node.as_index()].iter() {
                graph.edges[node.as_index()].retain(|edge| edge.child != loss.mapped_node);
            }
            println!("edges after = {:?}", graph.edges[node.as_index()]);
            while let Some(overlaps) = overlapper.calculate_next_overlap_set() {
                process_overlaps(*node, &overlaps, ancestry_changes, parents, &mut graph);
            }
            // The latter case is "no overlaps with children" == extinct node
            //if q.len() == 1 || q.is_empty() {
            //    // FIXME: don't do this -- it is bad for mutation
            //    // simplification
            //    graph.ancestry[node.as_index()].clear();
            //    for edge in graph.edges[node.as_index()].iter() {
            //        if let Some(node_parents) = &parents[node.as_index()] {
            //            for &parent in node_parents.iter() {
            //                // NOTE: unclear on the utility of this...
            //                // The ONE benefit is that it will let us
            //                // NOT CLEAR the unary ancestry from an extinct node,
            //                // making mutation simplification more feasible.
            //                if let Some(needle) = children_to_check[parent]
            //                    .iter()
            //                    .position(|x| x == &node.as_index())
            //                {
            //                    children_to_check[parent].remove(needle);
            //                }

            //                if !children_to_check[parent].contains(&edge.child.as_index()) {
            //                    children_to_check[parent].push(edge.child.as_index());
            //                }
            //            }
            //        }
            //        if let Some(cparents) = &mut parents[edge.child.as_index()] {
            //            if let Some(needle) = cparents.iter().position(|x| x == &node.as_index()) {
            //                cparents.remove(needle);
            //            }
            //        }
            //    }
            //    // Set node status to "extinct"
            //    graph.edges[node.as_index()].clear();
            //    graph.birth_time[node.as_index()] = None;
            //    parents[node.as_index()] = None;
            //} else if q.len() > 1 {
            //    println!("overlaps {node:?}: {q:?}");
            //    assert!(lost_edges.windows(2).all(|w| w[0] < w[1]));

            //    // Remove lost edges.
            //    for lost in lost_edges.iter().rev() {
            //        graph.edges[node.as_index()].remove(*lost);
            //    }

            //    // Insert overlaps.
            //    // NOTE: this will do bad things if a node
            //    // is retained as an edge -- we won't move it,
            //    // and we'll re-push it.
            //    println!("current edges = {:?}", graph.edges[node.as_index()]);
            //    graph.edges[node.as_index()].clear();
            //    for a in q {
            //        println!("Adding edge to node {:?}", a.node);
            //        graph.edges[node.as_index()].push(Edge {
            //            segment: a.segment,
            //            child: a.node,
            //        });
            //        if let Some(cparents) = &mut parents[a.node.as_index()] {
            //            if !cparents.contains(&node.as_index()) {
            //                cparents.push(node.as_index());
            //            }
            //        }
            //    }
            //}
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

        let mut ancestry_changes = vec![vec![]; graph.birth_time.len()];
        for node in [node3, node4] {
            ancestry_changes[node.as_index()].push(AncestryChange {
                segment: Segment { left: 0, right: 50 },
                mapped_node: node,
                source_node: node,
                change_type: ChangeType::Overlap,
            })
        }

        graph = propagate_changes(&nodes, graph, &mut ancestry_changes, &mut parents);

        for node in 0..graph.ancestry.len() {
            println!(
                "{node} => {:?}, {:?}",
                graph.ancestry[node], graph.edges[node]
            );
        }
        assert!(!graph.ancestry[1].is_empty());
        assert!(!graph.ancestry[2].is_empty());
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

        let mut ancestry_changes = vec![vec![]; graph.birth_time.len()];
        graph = propagate_changes(&nodes, graph, &mut ancestry_changes, &mut parents);

        for node in 0..graph.ancestry.len() {
            println!(
                "{node} => {:?}, {:?}",
                graph.ancestry[node], graph.edges[node]
            );
        }
        assert!(!graph.ancestry[1].is_empty());
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
            lsegment,
            rsegment,
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
        let mut ancestry_changes = vec![vec![]; graph.birth_time.len()];
        graph = propagate_changes(&nodes, graph, &mut ancestry_changes, &mut parents);

        for node in 0..graph.ancestry.len() {
            println!(
                "{node} => {:?}, {:?}",
                graph.ancestry[node], graph.edges[node]
            );
        }

        for node in [node1, node2] {
            assert_eq!(
                graph.edges[node.as_index()]
                    .iter()
                    .filter(|e| e.segment == rsegment)
                    .count(),
                2
            );
        }
        assert_eq!(
            graph.edges[node2.as_index()]
                .iter()
                .filter(|e| e.segment == lsegment)
                .count(),
            3
        );
    }
}

#[test]
fn nothing_to_queue() {
    let edges = [0, 1];
    let ancestry = [[0], [1]];

    let mut q = vec![];

    for e in &edges {
        for a in &ancestry[*e] {
            if a != e {
                q.push(*a);
            }
        }
    }
    assert!(q.is_empty());
}

#[test]
fn one_to_queue() {
    let edges = [0, 1];
    let ancestry = [[0], [2]];

    let mut q = vec![];

    for e in &edges {
        for a in &ancestry[*e] {
            if a != e {
                q.push(*a);
            }
        }
    }
    assert!(!q.is_empty());
    assert_eq!(q, [2]);

    // Now, we "just" have to update the child of 1 to point to 2!!
}

#[test]
fn gain_overlap_from_birth() {
    // Some transmission
    let edges = vec![vec![1, 2], vec![], vec![]];
    // All nodes map to self
    let ancestry = [vec![0], vec![1], vec![2]];
    let mut q = vec![];

    for e in &edges[0] {
        for a in &ancestry[*e] {
            // either a different node OR
            // current parental anc. segment "maps to self"
            if a != e || ancestry[0][0] == 0 {
                q.push(*a);
            }
        }
    }
    assert!(!q.is_empty());
    assert_eq!(q, [1, 2]);
}

#[test]
fn detect_edge_death() {
    todo!("this need refinement, but may be in the right direction");
    struct Change {
        node: usize,
        change_type: ChangeType,
    }
    // Some transmission
    let edges = vec![vec![1, 2], vec![], vec![]];
    // Node 2 has "gone totally extinct"
    // NOTE: I **think** that this would normally happen
    // only if a node is born and then has no offspring?
    // OR, we need some way to represent an "overlap to"
    let ancestry = [vec![0], vec![1], vec![]];
    let mut q = vec![];
    let mut edge_losses = vec![];
    for e in &edges[0] {
        let mut overlaps = 0;
        for a in &ancestry[*e] {
            // child node ancestry is either unary
            // or
            // current parental anc. segment "maps to self"
            if a != e || ancestry[0][0] == 0 {
                overlaps += 1;
                q.push(*a);
            }
        }
        if overlaps == 0 {
            edge_losses.push(*e);
        }
    }
    assert!(!q.is_empty());
    assert_eq!(q, [1]);
    assert!(!edge_losses.is_empty());
    assert_eq!(edge_losses, [2]);
}
