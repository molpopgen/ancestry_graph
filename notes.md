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

### Graph public/internal API issues 

#### Caching births

NOTE: this is fixed in this branch

This is NOT necessary!
This is O(N) memory and time that we can lose.
We should be able to:

1. Count how many births happened.
2. During transmission propagation, change status from Birth -> Alive,
   incrementing a count of how many status changes we made.
3. Assert no. status changes == no. births.

## Making the queue generation work over an iterable.

Ideally we want to iterate over input parental ancestry.
Doing so is a bit tricky to use the iterator in a piecemeal manner.
Looks like we need to use itertools.
From playground:

```rust
extern crate itertools;

use itertools::Itertools;

fn doit<'i, I>(i: I) -> Vec<i32>
where
    I: Iterator<Item = &'i i32> + itertools::PeekingNext,
{
    let mut i = i;
    let mut n = i.next();
    let mut rv = vec![];
    while let Some(x) = n {
        rv.push(*x);
        for &z in i.peeking_take_while(|&y| {
            y == x
        }) {
            rv.push(z)
        }
        n = i.next();
    }
    rv
}

fn main() {
    let v = vec![0, 1, 1, 2, 3];
    let vv = doit(v.iter());
    assert_eq!(v, vv);
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
