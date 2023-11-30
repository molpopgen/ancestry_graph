def interval_subraction(i, j):
    if i[1] > j[0] and j[1] > i[0]:
        # get the minimal overlap
        ml = max(i[0], j[0])
        mr = min(i[1], j[1])
        if ml != i[0]:
            left = (min(i[0], j[0]), ml)
        else:
            left = None
        if mr != i[1]:
            right = (mr, max(i[1], j[1]))
        else:
            right = None
        print(i, j, ml, mr, left, right)
        return (left, right)
    else:
        return (None, None)


def interval_delta(a, b):
    rv = []
    i = 0
    j = 0
    while i < len(a):
        while j < len(b) and b[j][1] < a[i][0]:
            rv.append(b[j])
            j += 1
        print(f"i = {i}")
        # if j >= len(b):
        #     rv.append(a[i])
        matched = False
        while j < len(b):
            print(f"{a[i]} {b[j]}")
            if b[j][1] > a[i][0] and a[i][1] > b[j][0]:
                print("overlap")
                matched = True
                if b[j][0] > a[i][0]:
                    rv.append((a[i][0], b[j][0]))
                    print("left:", rv[-1])
                if b[j][1] != a[i][1]:
                    tl = min(b[j][1], a[i][1])
                    tr = max(b[j][1], a[i][1])
                    rv.append((tl, tr))
                    print("right:", rv[-1])
                j += 1
            else:
                print(f"no overlap: {a[i]}")
                if not matched:
                    matched = True
                    rv.append(a[i])
                break
        print(f"{i} {j}")
        if not matched:
            rv.append(a[i])
        i += 1
    print("dun", i, len(a))
    return rv


def interval_delta_foo(a, b):
    rv = []
    k = 0
    last_right = a[0][0]
    for ai in a:
        matched = False
        for bi in b[k:]:
            if ai[0] >= bi[1]:
                break
            if bi[1] > ai[0] and ai[1] > bi[0]:
                matched = True
                if ai[0] < bi[0]:
                    rv.append((ai[0], bi[0]))
                # if bi[1] != ai[1]:
                #     mr = min(bi[1], ai[1])
                #     tl = min(mr, last_right)
                #     tr = max(mr, last_right)
                #     rv.append((tl, tr))
                last_right = bi[1]
                k += 1
            else:
                last_right = ai[1]
                if matched is True:
                    break
                rv.append(ai)
        if last_right < ai[1]:
            # rv.append((last_right, ai[1]))
            # last_right=ai[1]
            print(f"derp: {last_right} < {ai}")
    if last_right != a[-1][1]:
        assert last_right < a[-1][1], f"{last_right}, {a[-1]}"
        rv.append((last_right, a[-1][1]))
    return rv


def interval_delta_old(a, b):
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


def test_subtract0():
    a = (0, 10)
    b = (6, 10)
    c = interval_subraction(a, b)
    assert c[0] == (0, 6)
    assert c[1] is None


def test_subtract1():
    a = (0, 4)
    b = (6, 10)
    c = interval_subraction(a, b)
    assert c == (None, None)


def test_subtract2():
    a = (0, 10)
    b = (3, 7)
    c = interval_subraction(a, b)
    assert c[0] == (0, 3)
    assert c[1] == (7, 10)


def test_subtract3():
    a = (0, 10)
    b = (0, 7)
    c = interval_subraction(a, b)
    print(c)
    assert c[0] is None
    assert c[1] == (7, 10)


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
