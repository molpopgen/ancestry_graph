def interval_delta(a, b):
    rv = []
    ai = 0
    bi = 0

    while ai < len(a):
        right = a[ai][1]
        left = a[ai][0]
        match = False
        while bi < len(b) and b[bi][0] < right:
            bleft = b[bi][0]
            bright = b[bi][1]
            if right > bleft and bright > left:
                match = True
                print(f"{left}, {right} | {bleft}, {bright}")
                if bleft > left:
                    print(f"left {left} {bleft}")
                    rv.append((left, bleft))
                if bright != right:
                    mr = min(right, bright)
                    mxr = max(right, bright)
                    print(f"right {mr} {mxr}")
                    rv.append((mr, mxr))
            else:
                break
            bi += 1
            print(rv)
        if not match:
            rv.append((left, right))
        ai += 1
    print(right)
    return rv


def validate(delta, expected):
    assert len(delta) == len(expected)
    for e in expected:
        assert e in delta


def test0():
    a = [(0, 10), (10, 100)]
    b = [(6, 10), (50, 73)]
    c = interval_delta(a, b)
    validate(c, [(0, 6), (10, 50), (73, 100)])


def test1():
    a = [(0, 10), (10, 100)]
    b = [(50, 73)]
    c = interval_delta(a, b)
    validate(c, [(0, 10), (10, 50), (73, 100)])


def test2():
    a = [(0, 10), (40, 80), (90, 100)]
    b = [(50, 73)]
    c = interval_delta(a, b)
    validate(c, [(0, 10), (40, 50), (73, 80), (90, 100)])


def test3():
    a = [(0, 10)]
    b = [(0, 10)]
    c = interval_delta(a, b)
    validate(c, [])


def test4():
    a = [(0, 10)]
    b = [(3, 5), (6, 8)]
    c = interval_delta(a, b)
    validate(c, [(0, 3), (5, 6), (8, 10)])


def test5():
    a = [(3, 5), (6, 8)]
    b = [(0, 10)]
    c = interval_delta(a, b)
    validate(c, [(0, 3), (5, 6), (8, 10)])