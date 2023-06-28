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

## General issues.

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

* We probably want all births to be at the EXACT SAME TIME
  so that we are simplifying at each "tick" of the simulation clock?

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
