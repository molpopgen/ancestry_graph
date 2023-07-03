# Development notes

## TODO

[X] internal sample nodes and unary overlap retention.
   * Internal sample handling is rather subtle
     and may need revisiting later.
[X] extinct node recycling
[] add "deme" column to graph.  Or a NodeData column.
   (low priority for this prototype)
[X] Node state changes during update.
   Birth -> Alive
   Death -> "Loss" or Ancestor
   etc..

   NOTE: support is probably incomplete.
   I think that this enum may go away later.
   (see below).
[X] enforce that all births are visited during
   ancestry change propagation
[] Need API to mark a node as sample
   (low priority for this prototype)
[] Ancestry change for a sample node
   should always be None (I think...?)
[] Consider the NOTEs section of test `test_ancestry_completeness_of_internal_samples`.

## General issues.

### Flags vs status enumeration

* `NodeStatus::Birth` may be useless?
   We need a way to track new births and
   all alive nodes.  Separate containers
   seems useful, but separate enum flags doesn't
   seem to be solving anything...

   In fact, can we get rid of a status enum altogether?

   The ONLY place that status seems to affect logic
   is when dispatching work out to process_node_death in lib.rs. 
   It would seem that HASHING the dead nodes could be (part of)
   the solution for eliminating this enum entirely.

   An alternative to hashing (which is slower access than the Vec)
   would be to take the enum and refactor it to be Death/NonDeath,
   and pass it along to the simplification function.
   This method avoids the hash AND removes the column from the
   Graph.

#### Type redundancy

* Currently, the flags and the status are redundant.
* We only define ONE flag ("is sample").
* We define a few "status" variants that are mutually-exclusive.
* We are thus using too much storage for questionable value.

At this point, it seems sensible to:

* Only use the enum.
* Convert the enum to `u32` to save space.
* Drop the `NodeFlags` type entirely.

#### Enum variant redundancy

It probably makes the most sense to:

* Collapse `NodeStatus::Alive` into `NodeStatus::Sample`.
  The reason is that an alive node is just a sample
  that will eventually be marked as "Death" and then,
  perhaps, as `Ancestor`

### Birth times

* We probably want all births to be at the EXACT SAME TIME
so that we are simplifying at each "tick" of the simulation clock?

  - This is now enforced.  We may be able to relax this later.

### Management of alive nodes

* It seems that this should be handled "externally" by client code.
* We need a "mark as dead" function to update the node status and copy
  the node to a vector of dead nodes.

### Sample nodes

* See comments for Topology5. It is currently hard to make a node
  a sample after its initial birth time.

### Ancestry change enumeration

Really, we are more interested in "changed" or "not changed"
for a segment rather than changes to Unary, to Overlap, etc..

Is it even useful to distinguish a Gain from a Loss, or is
Change sufficient?  If Change will work, then Option<Change>, where
Change is a ZST, is all we need.

I think we do need to distinguish:

* Overlap((Segment, Node))
* Loss((Segment, Node))

Then, the queue/overlapper duo will figure out
if the parental genome ends up changed.

### AncestryType

This enumeration would be clearer as:

```rust
enum SegmentMapping {
    Self(Node),
    Child(Node)
}
```

## Use of a typestate idiom

Something like this, which was quickly hacked together
on playground, would seem to reduce our over-typing:

```rust
enum OverlapState {
    ToSelf,
    Child,
}

enum ChangeState {
    Loss,
    Overlap,
}

trait State {}

impl State for OverlapState {}
impl State for ChangeState {}

struct Segment<T: State> {
    left: i64,
    right: i64,
    node: usize,
    state: T,
}

impl<T: State> Segment<T> {
    fn new(left: i64, right: i64, node: usize, state: T) -> Self {
        Self {
            left,
            right,
            node,
            state,
        }
    }
    fn overlaps(&self, other: &Self) -> bool {
        self.right > other.left && other.right > self.left
    }
}

impl Segment<ChangeState> {
    fn new_loss(left: i64, right: i64, node: usize) -> Self {
        Self::new(left, right, node, ChangeState::Loss)
    }
}

fn main() {
    let a = Segment {
        left: 0,
        right: 10,
        node: 1,
        state: OverlapState::ToSelf,
    };
    let b = Segment::new_loss(0, 10, 1);
    // compile fail: the "T" are different
    //let o = a.overlaps(&b);
}
```

## Ancestry table

It may be useful to design the ancestry table as:

```rust
enum Ancestry {
    ToSelf(Node), // Birth/Sample node
    Unary(Node),  // Unary transmission
    // EntryPoint is an index into a place
    // where we can efficiently manage
    // the nodes being overlapped
    Overlap(EntryPoint) // Coalescence
}
```

With a design like this, we can use a vector-backed
data structure to manage the overlaps in a way
that allows us to recycle memory locations.
