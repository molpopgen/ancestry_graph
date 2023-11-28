def interval_delta(a, b):
    rv = []
    ai = 0
    bi = 0

    while ai < len(a):
        right = a[ai][1]
        left = a[ai][0]
        match = False
        while bi < len(b) and b[bi][0] < right:
            print(a[ai], b[bi])
            if b[bi][0] < right:
                match = True
                if b[bi][0] > left:
                    rv.append((left, b[bi][0]))
            if b[bi][1] != right:
                match = True
                rv.append((min(right, b[bi][1]), max((right, b[bi][1]))))
            if b[bi][0] < right:
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
