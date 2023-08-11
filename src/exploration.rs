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
            println!("recycling {index}");
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

pub struct Graph {
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
    last_ancestry_index: Index,
    current_ancestry_index: Index,
    ancestry: &mut NodeAncestry,
    head: Option<Index>,
    prev: Option<Index>,
) -> Index {
    let mut head = head;
    let mut prev = prev;
    let mut seg_right = None;
    let mut current_ancestry_index = current_ancestry_index;
    let (mut current_left, current_right) = {
        let current = ancestry.get(current_ancestry_index);
        (current.segment.left, current.segment.right)
    };
    let temp_left = std::cmp::max(current_left, left);
    let temp_right = std::cmp::min(current_right, right);
    println!("{:?} {temp_left} {temp_right}", current_ancestry_index);
    let mut rv = ancestry.next_raw(current_ancestry_index);
    if current_left != temp_left {
        println!("we have a left dangle on {current_left}, {temp_left}");
    }
    if current_right != temp_right {
        {
            let current = ancestry.get_mut(current_ancestry_index);
            current.segment.left = temp_right;
            current_left = temp_right;
        }
        seg_right = Some(AncestrySegment {
            segment: Segment {
                left: temp_right,
                right: current_right,
            },
            mapped_node: ancestry.get(current_ancestry_index).mapped_node,
        });
    }
    let out_seg = AncestrySegment {
        segment: Segment { left, right },
        mapped_node,
    };
    println!("out = {out_seg:?}");
    if left == current_left && right == current_right {
        // perfect overlap, all we need to do is update
        // the mapped node...
        let current = ancestry.get_mut(current_ancestry_index);
        println!(
            "update mapping from {:?} to {:?}",
            current.mapped_node, out_seg.mapped_node
        );
        current.mapped_node = out_seg.mapped_node;
        // ... but if the mapping has changed, then this
        // segment is possibly a CHANGE TO UNARY that we
        // must record
    } else {
        println!("insert out");
        // We insert out_seg at current_ancestry_index
        if current_ancestry_index == last_ancestry_index {
            println!("case A");
            // replace current with out_seg and insert the
            // current value next
            let current = *ancestry.get(current_ancestry_index);
            let next = ancestry.next_raw(current_ancestry_index);
            let new_index = ancestry.new_index(current);
            println!("new_index = {new_index:?}");
            ancestry.next[new_index.0] = next.0;
            let _ = std::mem::replace(&mut ancestry.data[current_ancestry_index.0], out_seg);
            ancestry.next[current_ancestry_index.0] = new_index.0;
            // Needed for handling right_seg below
            current_ancestry_index = new_index;
            rv = current_ancestry_index;
            println!("rv = {rv:?}");
        } else {
            println!("case B");
            let next = ancestry.next_raw(last_ancestry_index);
            let new_index = ancestry.new_index(out_seg);
            ancestry.next[last_ancestry_index.0] = new_index.0;
            ancestry.next[new_index.0] = next.0;
            rv = new_index;
        }
    }

    if let Some(right_seg) = seg_right {
        println!("seg_right = {right_seg:?}");
        if right_seg.segment.left == current_left && right_seg.segment.right == current_right {
            let current = ancestry.get_mut(current_ancestry_index);
            println!(
                "updating node from {:?} to {:?}",
                current.mapped_node, right_seg.mapped_node
            );
            // Could be an ancestry change!
            current.mapped_node = right_seg.mapped_node;
            //rv = ancestry.next_raw(current_ancestry_index);
        } else {
            println!("inserting seg_right");
            let next = ancestry.next_raw(current_ancestry_index);
            let new_index = ancestry.new_index(right_seg);
            ancestry.next[current_ancestry_index.0] = new_index.0;
            ancestry.next[new_index.0] = next.0;
            rv = new_index;
        }
    }
    rv
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
    let mut last_ancestry_index = ahead;
    // let mut last_anc_segment: Option<AncestrySegment> = None;
    let mut current_overlap = 0_usize;
    while !ahead.is_sentinel() && current_overlap < overlaps.len() {
        println!(
            "ahead: {ahead:?} = {:?}, out head: {head:?}, out tail {prev:?}",
            ancestry.get(ahead)
        );
        // todo!("revisit this after we add more tests to our Py prototype to hit more code paths");
        //let (anc_current_left, anc_current_right) = if let Some(aseg) = last_anc_segment {
        //    (aseg.segment.left, aseg.segment.right)
        //} else {
        //    let current = ancestry.get(ahead);
        //    (current.segment.left, current.segment.right)
        //};
        //println!("processing: {anc_current_left}, {anc_current_right} and {ahead:?}");
        let (left, right, mapped_node) = overlaps[current_overlap];
        if right > ancestry.get(ahead).segment.left && ancestry.get(ahead).segment.right > left {
            println!(
                "yes {left}, {right}, {:?}, {:?}",
                ancestry.get(ahead).segment.left,
                ancestry.get(ahead).segment.right
            );
            //last_anc_segment = {
            //    let current = ancestry.get(ahead);
            //    Some(*current)
            //};
            last_ancestry_index = ahead;
            ahead = update_ancestry(
                left,
                right,
                mapped_node,
                last_ancestry_index,
                ahead,
                ancestry,
                head,
                prev,
            );
            println!("updated to {ahead:?}");
            // println!("seg_right = {:?}", ancestry.get(seg_right));
            //println!("returned {head:?}, {prev:?}, {seg_right:?}");
            //ahead = seg_right;
            current_overlap += 1;
        } else {
            println!(
                "no {left}, {right}, {:?} | vs {}, {}",
                ancestry.next(ahead),
                ancestry.get(ahead).segment.left,
                ancestry.get(ahead).segment.right,
            );
            // Here, it is likely that we want to free the ancestry
            // segment.
            // Will need test coverage of that idea later.
            last_ancestry_index = ahead;
            let temp = ancestry.next_raw(ahead);
            // Remove the node.
            // Fragile if ahead is a previous entry's next item.
            ancestry.next[ahead.0] = usize::MAX;
            ancestry.free_list.push(ahead.0);
            ahead = temp;
            //last_anc_segment = None;
        }
    }
    println!("done: {:?} {:?}", last_ancestry_index, ahead);
    // WARNING: EPIC HACK ALERT
    if !ahead.is_sentinel() {
        ancestry.next[last_ancestry_index.0] = usize::MAX
    }
    // WARNING: everything below is very fragile and
    // needs testing
    //if let Some(index) = head {
    //    ancestry_head[node.as_index()] = index;
    //} else {
    //    // This is WRONG.
    //    // We need to TRAVERSE THE ENTIRE LIST AND FREE IT
    //    ancestry_head[node.as_index()] = Index::sentinel();
    //}
    //// THIS IS WRONG: we only do this work if we ARE NOT
    //// trashing the entire list. (See above...)
    //// The inner logic is also WRONG.
    //// If we are changing the tail, then we need to
    //// make sure that we TRUNCATE THE LIST FROM THE
    //// CURRENT TAIL ONWARDS, BUT WE MAY HAVE TO WORRY
    //// ABOUT SOME SUBTLE ISSUES THERE THAT I AM NOT
    //// ABLE TO ARTICULATE YET.
    //if let Some(index) = prev {
    //    ancestry_tail[node.as_index()] = index;
    //} else {
    //    ancestry_tail[node.as_index()] = Index::sentinel();
    //}
    //assert_eq!(ancestry.next[ancestry_tail[node.as_index()].0], usize::MAX);
    //println!(
    //    "final {:?}, {:?} = {:?}, {:?}",
    //    ancestry_head[node.as_index()],
    //    ancestry_tail[node.as_index()],
    //    ancestry.get(ancestry_head[node.as_index()]),
    //    ancestry.get(ancestry_tail[node.as_index()])
    //);
}

#[cfg(test)]
mod test_utils {
    use super::*;

    #[must_use]
    pub(super) fn setup_ancestry(
        input_ancestry: &[Vec<(i64, i64, Node)>],
        ancestry: &mut CursorList<AncestrySegment>,
    ) -> (Vec<Index>, Vec<Index>) {
        let mut ancestry_head = vec![];
        let mut ancestry_tail = vec![];

        for inner in input_ancestry {
            if !inner.is_empty() {
                let head = ancestry.new_index(AncestrySegment {
                    segment: Segment {
                        left: inner[0].0,
                        right: inner[0].1,
                    },
                    mapped_node: inner[0].2,
                });
                ancestry_head.push(head);
                let mut last = head;
                for (left, right, mapped_node) in inner.iter().skip(1) {
                    last = ancestry.insert_after(
                        last,
                        AncestrySegment {
                            segment: Segment {
                                left: *left,
                                right: *right,
                            },
                            mapped_node: *mapped_node,
                        },
                    );
                }
                ancestry_tail.push(last);
            }
        }

        (ancestry_head, ancestry_tail)
    }

    #[must_use]
    pub(super) fn run_ancestry_tests(
        input_ancestry: &[Vec<(i64, i64, Node)>],
        overlaps: &[(i64, i64, Node)],
    ) -> (NodeAncestry, Vec<Index>, Vec<Index>) {
        let mut ancestry = NodeAncestry::with_capacity(1000);
        let (mut ancestry_head, mut ancestry_tail) =
            test_utils::setup_ancestry(input_ancestry, &mut ancestry);
        for i in 0..ancestry.data.len() {
            println!("{i}: {:?} => {:?}", ancestry.data[i], ancestry.next[i])
        }
        update_ancestry_design(
            Node(0),
            overlaps,
            &mut ancestry,
            &mut ancestry_head,
            &mut ancestry_tail,
        );
        let mut extracted = vec![];
        let mut h = ancestry_head[0];
        while !h.is_sentinel() {
            let a = ancestry.get(h);
            println!("extracted {a:?}");
            extracted.push((a.segment.left, a.segment.right, a.mapped_node));
            let next = ancestry.next_raw(h);
            // Check that our tail is properly updated
            //if next.is_sentinel() {
            //    assert_eq!(ancestry_tail[0], h);
            //}
            h = next;
        }
        for i in &extracted {
            assert!(overlaps.contains(i), "{i:?}, {overlaps:?} != {extracted:?}");
        }
        for o in overlaps {
            assert!(
                extracted.contains(o),
                "{o:?}, {extracted:?} != {overlaps:?}"
            );
        }

        (ancestry, ancestry_head, ancestry_tail)
    }
}

// this is test3 from the python prototype
#[test]
fn test_list_updating_1() {
    let anc0 = vec![(0_i64, 2_i64, Node(0))];
    let anc1 = vec![(1_i64, 2_i64, Node(1))];
    let anc2 = vec![(0_i64, 1_i64, Node(2))];
    let input_ancestry = vec![anc0, anc1, anc2];

    // (left, right, mapped_node)
    // cribbed from manual calculation/the python prototype
    let overlaps = [(0_i64, 1_i64, Node(2)), (1, 2, Node(1))];
    let (ancestry, _, _) = test_utils::run_ancestry_tests(&input_ancestry, &overlaps);
    // FIXME: below should be audited carefully.
    // Adding in "seg_right" is creating new entries.
    //assert_eq!(ancestry.data.len(), 5);
    //assert_eq!(ancestry.next.len(), 5);
}

// this is test0 from the python prototype
#[test]
fn test_list_updating_2() {
    let input_ancestry = vec![
        vec![(0_i64, 3_i64, Node(0))],
        vec![(0_i64, 1_i64, Node(1)), (2_i64, 3_i64, Node(1))],
        vec![(0_i64, 1_i64, Node(2))],
    ];

    // (left, right, mapped_node)
    // cribbed from manual calculation/the python prototype
    let overlaps = [(0_i64, 1_i64, Node(0)), (2, 3, Node(1))];
    let (ancestry, _, _) = test_utils::run_ancestry_tests(&input_ancestry, &overlaps);
    // FIXME: below should be audited carefully.
    // Adding in "seg_right" is creating new entries.
    assert_eq!(ancestry.data.len(), 6);
    assert_eq!(ancestry.next.len(), 6);
}

// This is test5b_b_less_contrived from python prototype
#[test]
fn test_list_updating_3() {
    let input_ancestry = vec![
        vec![
            (0_i64, 1_i64, Node(1)),
            (1_i64, 8_i64, Node(0)),
            (8, 16, Node(1)),
        ],
        vec![(2, 8, Node(1)), (12, 14, Node(1))],
        vec![],
        vec![],
    ];
    let overlaps = vec![(2_i64, 8_i64, Node(1)), (12, 14, Node(1))];
    let (ancestry, _, _) = test_utils::run_ancestry_tests(&input_ancestry, &overlaps);
    for i in 0..ancestry.data.len() {
        println!("{i}: {:?} => {:?}", ancestry.data[i], ancestry.next[i])
    }
}
