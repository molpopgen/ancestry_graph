# Development notes

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
