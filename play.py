from dataclasses import dataclass


@dataclass
class Segment:
    left: int
    right: int
    node: int


def update_overlap(a, left, right, node, anc):
    ai = anc[a]
    tleft = max(ai.left, left)
    tright = min(ai.right, right)
    segright = None
    rv = a + 1
    print(a, tleft, tright)
    if ai.left != tleft:
        # This is a segment loss,
        # and we need to register it somehow??
        print(f"we have a left dangle loss on {ai.left}, {tleft} != {ai} ")
        # raise NotImplementedError(f"lefts: {ai.left} != {tleft}, {ai}")
    if ai.right != tright:
        segright = Segment(tright, ai.right, ai.node)
        ai.left = tright
    # if segright is not None:
    out = Segment(left, right, node)
    print(f"out={out}, ai = {ai}")
    if out.left == ai.left and out.right == ai.right:
        print(f"update mapping from {ai.node} to {out.node}")
        ai.node = out.node
    else:
        print("insert out")
        # NOTE: this means that we need to cache
        # the previous index in our rust version,
        # so that we can insert out before it and
        # update all the list-y stuff.
        anc.insert(a, out)
        rv = a + 1
        # rv -= 1
    if segright is not None:
        print(f"segright = {segright}")
        if segright.left == ai.left and segright.right == ai.right:
            print(f"updating node from {ai.node} to {segright.node}")
            ai.node = segright.node
            rv = a + 1
        else:
            print(f"not equal to {ai.left == segright.left}")
            anc.insert(a + 1, segright)
            rv = a + 1
        # rv += 1

    # if left == ai.left and right == ai.right:
    #     ai.node = node
    # else:
    #     out = Segment(left, right, node)
    #     print(f"out = {out}")
    #     anc.insert(a, out)
    # rv += 1
    print(f"returning {a} => {rv} | {anc}")
    return rv


def update(anc, overlaps):
    a = 0
    o = 0

    # while a < len(anc) and o < len(overlaps):
    while o < len(overlaps):
        oi = overlaps[o]
        ai = anc[a]
        if oi.right > ai.left and ai.right > oi.left:
            a = update_overlap(a, oi.left, oi.right, oi.node, anc)
            o += 1
        else:
            # Delete the input segment
            anc.pop(a)
            # raise NotImplementedError
            # a += 1

    print(f"done: {a} {len(anc)}")

    if len(anc) >= a:
        print(f"truncating from {anc} to: {anc[:a]}")
        anc[:] = anc[:a]


print("TEST 1")

anc = [Segment(0, 2, 0)]
overlaps = [Segment(0, 1, 2), Segment(1, 2, 1)]

update(anc, overlaps)
assert sorted(anc, key=lambda x: x.left) == [
    Segment(0, 1, 2),
    Segment(1, 2, 1),
], f"{anc}"

print("TEST 2")

anc = [Segment(0, 3, 0)]
overlaps = [Segment(0, 1, 0), Segment(2, 3, 1)]

update(anc, overlaps)
assert sorted(anc, key=lambda x: x.left) == overlaps, f"{anc}"

print("TEST 3")

anc = [Segment(0, 5, 0)]
overlaps = [Segment(1, 2, 0), Segment(3, 4, 1)]

update(anc, overlaps)
assert sorted(anc, key=lambda x: x.left) == overlaps, f"{anc}"

print("TEST 3")

anc = [Segment(0, 1, 17), Segment(1, 5, 0)]
overlaps = [Segment(1, 2, 0), Segment(3, 4, 1)]

update(anc, overlaps)
assert sorted(anc, key=lambda x: x.left) == overlaps, f"{anc}"

print("TEST 4")
anc = [Segment(1, 8, 0), Segment(8, 14, 1), Segment(14, 16, 1)]
overlaps = [Segment(2, 8, 1), Segment(12, 14, 1)]
update(anc, overlaps)
assert sorted(anc, key=lambda x: x.left) == overlaps, f"{anc}"
