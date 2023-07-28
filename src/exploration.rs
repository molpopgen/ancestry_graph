use crate::Node;

// TODO: this should be in another module
// and made pub for internal use
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, PartialOrd, Ord, Eq)]
pub struct Index(usize);

impl Index {
    #[inline(always)]
    fn sentinel() -> Self {
        Self(usize::MAX)
    }

    #[inline(always)]
    fn is_sentinel(&self) -> bool {
        self.0 == usize::MAX
    }

    #[inline(always)]
    fn into_option(self) -> Option<Self> {
        if self.is_sentinel() {
            None
        } else {
            Some(self)
        }
    }
}

// TODO: this should be in another module
// and made pub for internal use
#[derive(Debug)]
struct CursorList<T> {
    data: Vec<T>,
    next: Vec<usize>,
    free_list: Vec<usize>,
}

impl<T> CursorList<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            next: Vec::with_capacity(capacity),
            data: Vec::with_capacity(capacity),
            free_list: vec![],
        }
    }

    pub fn get(&self, at: Index) -> &T {
        &self.data[at.0]
    }

    pub fn get_mut(&mut self, at: Index) -> &mut T {
        &mut self.data[at.0]
    }

    pub fn new_index(&mut self, datum: T) -> Index {
        if let Some(index) = self.free_list.pop() {
            let _ = std::mem::replace(&mut self.data[index], datum);
            Index(index)
        } else {
            self.next.push(Index::sentinel().0);
            self.data.push(datum);
            Index(self.data.len() - 1)
        }
    }

    fn setup_insertion(&mut self, at: Index, datum: T) -> (Index, Index) {
        let new_index = self.new_index(datum);
        (new_index, at)
    }

    fn finalize_insertion(&mut self, next: Index, next_value: Index) {
        self.set_next(next, next_value);
    }

    fn set_next(&mut self, at: Index, value: Index) {
        self.next[at.0] = value.0
    }

    pub fn add_list(&mut self, datum: T) -> Index {
        self.new_index(datum)
    }

    pub fn next(&self, at: Index) -> Option<Index> {
        self.next_raw(at).into_option()
    }

    fn next_raw(&self, at: Index) -> Index {
        Index(self.next[at.0])
    }

    pub fn insert_after(&mut self, at: Index, datum: T) -> Index {
        let (new_index, index_at) = self.setup_insertion(at, datum);
        if let Some(next) = self.next(at) {
            self.set_next(new_index, next);
        }
        self.finalize_insertion(index_at, new_index);
        new_index
    }

    // Excise a node from a list.
    // The Index goes into the free list for
    // later recycling, making it a logic error
    // to use the value of `at` for further operations.
    //pub fn remove(&mut self, at: Index) {
    //    let prev = self.prev_raw(at);
    //    let next = self.next_raw(at);

    //    if !prev.is_sentinel() {
    //        self.set_next(prev, next);
    //    }
    //    if !next.is_sentinel() {
    //        self.set_prev(next, prev);
    //    }
    //    self.set_prev(at, Index::sentinel());
    //    self.set_next(at, Index::sentinel());
    //    self.free_list.push(at.0);
    //}
}

type NodeAncestry = CursorList<AncestrySegment>;

#[derive(Clone, Copy, Debug)]
struct Segment {
    left: i64,
    right: i64,
}

impl Segment {
    fn overlaps(&self, other: &Segment) -> bool {
        self.right > other.left && other.right > self.left
    }
}

#[derive(Clone, Copy, Debug)]
struct AncestrySegment {
    segment: Segment,
    mapped_node: Node,
}

struct Edge {
    segment: Segment,
    child: Node,
}

struct AncestryIntersection {
    segment: Segment,
    mapped_node: Node,
    edge_index: Index,
}

struct AncestryOverlapper {
    queue: Vec<AncestryIntersection>,
    num_overlaps: usize,
    current_overlap: usize,
    parent: Node,
    left: i64,
    right: i64,
    overlaps: Vec<AncestryIntersection>,
}

struct Graph {
    birth_time: Vec<i64>,
    edges: CursorList<Edge>,
    edge_head: Vec<Index>,
    ancestry: CursorList<AncestrySegment>,
    ancestry_head: Vec<Index>,
}

fn ancestry_intersection(node: Node, graph: &Graph, queue: &mut Vec<AncestryIntersection>) {
    let mut current_edge = Some(graph.edge_head[node.as_index()]);
    let mut current_ancestry = Some(graph.ancestry_head[node.as_index()]);

    while let Some(edge_index) = current_edge {
        while let Some(aseg) = current_ancestry {
            let edge_ref = graph.edges.get(edge_index);
            let anc_ref = graph.ancestry.get(aseg);
            if anc_ref.segment.overlaps(&edge_ref.segment) {
                let left = std::cmp::max(edge_ref.segment.left, anc_ref.segment.left);
                let right = std::cmp::min(edge_ref.segment.right, anc_ref.segment.right);
                queue.push(AncestryIntersection {
                    segment: Segment { left, right },
                    mapped_node: anc_ref.mapped_node,
                    edge_index,
                });
            }
            current_ancestry = graph.ancestry.next(aseg);
        }
        current_edge = graph.edges.next(edge_index);
    }
    queue.sort_unstable_by_key(|x| x.segment.left);
}

fn update_ancestry(
    left: i64,
    right: i64,
    mapped_node: Node,
    current_ancestry_index: Index,
    ancestry: &mut NodeAncestry,
    head: Option<Index>,
    prev: Option<Index>,
) -> (Option<Index>, Option<Index>) {
    let mut head = head;
    let mut prev = prev;
    let mut current_ancestry_index = current_ancestry_index;
    let (anc_current_left, anc_current_right) = {
        let current = ancestry.get(current_ancestry_index);
        (current.segment.left, current.segment.right)
    };
    let temp_left = std::cmp::max(left, anc_current_left);
    let temp_right = std::cmp::min(right, anc_current_right);

    let mut seg_left: Option<Index> = None;
    let mut seg_right: Option<Index> = None;

    if anc_current_left != temp_left {
        seg_left = Some(ancestry.new_index(AncestrySegment {
            segment: Segment {
                left: anc_current_left,
                right: temp_left,
            },
            mapped_node,
        }));
    }

    if anc_current_right != temp_right {
        let current = ancestry.get_mut(current_ancestry_index);
        current.segment.left = temp_right;
        seg_right = Some(current_ancestry_index);
    } else {
        seg_right = ancestry.next(current_ancestry_index);
        // seg_right = current.next
        // TODO: free current
    }

    let out_seg = AncestrySegment {
        segment: Segment { left, right },
        mapped_node,
    };

    if let Some(index) = prev {
        let temp = ancestry.new_index(out_seg);
        ancestry.next[index.0] = temp.0;
        prev = Some(temp);

        //if let Some(value) = seg_right {
        //    ancestry.next[index.0] = value.0;
        //} else {
        //    ancestry.next[index.0] = Index::sentinel().0;
        //}
    } else {
        assert!(seg_right.is_some());
        head = Some(ancestry.new_index(out_seg));
        //ancestry.next[prev.unwrap().0] = seg_right.unwrap().0;
        prev = head;
    }

    (head, prev)
}

fn update_ancestry_design(
    node: Node,
    overlaps: &[(i64, i64, Node)],
    ancestry: &mut NodeAncestry,
    ancestry_head: &mut [Index],
    ancestry_tail: &mut [Index],
) {
    assert_eq!(ancestry_head.len(), ancestry_tail.len());
    let mut head: Option<Index> = None;
    let mut prev: Option<Index> = None;
    let mut ahead = ancestry_head[node.as_index()];
    for o in overlaps {
        println!("FOO {o:?} {head:?}, {prev:?}");
        while !ahead.is_sentinel() {
            let (anc_current_left, anc_current_right) = {
                let current = ancestry.get(ahead);
                (current.segment.left, current.segment.right)
            };
            let (left, right, mapped_node) = *o;
            if right > anc_current_left && anc_current_right > left {
                (head, prev) =
                    update_ancestry(left, right, mapped_node, ahead, ancestry, head, prev);
            } else {
                ahead = ancestry.next_raw(ahead);
            }
            break;
        }
    }
    if let Some(index) = head {
        ancestry_head[node.as_index()] = index;
    } else {
        ancestry_head[node.as_index()] = Index::sentinel();
    }
    if let Some(index) = prev {
        ancestry_tail[node.as_index()] = index;
    } else {
        ancestry_tail[node.as_index()] = Index::sentinel();
    }
    println!(
        "final {:?}, {:?}",
        ancestry_head[node.as_index()],
        ancestry_tail[node.as_index()]
    );
}

// this is test3 from the python prototype
#[test]
fn test_list_updating() {
    let mut ancestry_head = vec![];
    let mut ancestry = NodeAncestry::with_capacity(1000);
    let node0head = ancestry.new_index(AncestrySegment {
        segment: Segment { left: 0, right: 2 },
        mapped_node: Node(0),
    });
    let node1head = ancestry.new_index(AncestrySegment {
        segment: Segment { left: 1, right: 2 },
        mapped_node: Node(1),
    });
    let node2head = ancestry.new_index(AncestrySegment {
        segment: Segment { left: 0, right: 1 },
        mapped_node: Node(2),
    });
    for i in [node0head, node1head, node2head] {
        ancestry_head.push(i);
    }
    let mut ancestry_tail = ancestry_head.clone();

    // (left, right, mapped_node)
    // cribbed from manual calculation/the python prototype
    let overlaps = [(0_i64, 1_i64, Node(2)), (1, 2, Node(1))];
    update_ancestry_design(
        Node(0),
        &overlaps,
        &mut ancestry,
        &mut ancestry_head,
        &mut ancestry_tail,
    );

    let mut extracted = vec![];
    let mut h = ancestry_head[0];
    while !h.is_sentinel() {
        let a = ancestry.get(h);
        extracted.push((a.segment.left, a.segment.right, a.mapped_node));
        h = ancestry.next_raw(h);
    }
    for i in &extracted {
        assert!(overlaps.contains(i), "{i:?}, {overlaps:?} != {extracted:?}");
    }
    for o in &overlaps {
        assert!(extracted.contains(o), "{o:?}, {extracted:?}");
    }
    println!("{ancestry:?}");
}
