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
