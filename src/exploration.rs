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

    pub fn new_index(&mut self, datum: T) -> Index {
        if let Some(index) = self.free_list.pop() {
            let _ = std::mem::replace(&mut self.data[index], datum);
            Index(index)
        } else {
            self.next.push(Index::sentinel().0);
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

struct Segment {
    left: i64,
    right: i64,
}

impl Segment {
    fn overlaps(&self, other: &Segment) -> bool {
        self.right > other.left && other.right > self.left
    }
}

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
