import heapq


def buildQ(data, other_data):
    Q = []
    for i in data:
        # heapq.heappush(Q, i)
        for j in other_data:
            if i[1] > j[0] and j[1] > i[0]:
                left = max(i[0], j[0])
                right = min(i[1], j[1])
                heapq.heappush(Q, (left, right, j[2], i[2]))
    return Q


def overlaps_from_Q(Q, L):
    overlaps = []
    while len(Q) > 0:
        left_position = Q[0][0]
        right_position = L
        X = []
        while len(Q) > 0 and Q[0][0] == left_position:
            x = heapq.heappop(Q)
            X.append(x)
            right_position = min(right_position, x[1])
        if len(Q) > 0:
            right_position = min(right_position, Q[0][0])
        overlaps.append((left_position, right_position, X))
        if len(X) == 1:
            x = X[0]
            if len(Q) > 0 and Q[0][0] < x[1]:
                x = (Q[0][0], x[1], x[2], x[3])
                heapq.heappush(Q, x)
        else:
            for x in X:
                if x[1] > right_position:
                    xx = (right_position, x[1], x[2], x[3])
                    heapq.heappush(Q, xx)
    return overlaps


# NOTE: this method should be
# preffered
def overlaps_from_Q_no_push(Q, L):
    overlaps = []

    current_overlaps = []
    right_position = Q[0][0]
    n = len(Q)
    Q.append((L+1, L+1, 'sentinel', 'sentinel'))
    i = 0
    while i < n:
        left_position = right_position
        current_overlaps = [
            i for i in current_overlaps if i[1] > left_position]
        if len(current_overlaps) == 0:
            left_position = Q[i][0]
        while i < n and Q[i][0] == left_position:
            x = Q[i]
            current_overlaps.append(x)
            i += 1

        i -= 1
        right_position = min(j[1] for j in current_overlaps)
        right_position = min(right_position, Q[i+1][0])
        i += 1

        overlaps.append((left_position, right_position, current_overlaps))
    if len(current_overlaps) > 0:
        left_position = right_position
        current_overlaps = [
            i for i in current_overlaps if i[1] > left_position]
        if len(current_overlaps) > 0:
            right_position = min(i[1] for i in current_overlaps)
            overlaps.append((left_position, right_position, current_overlaps))

    return overlaps

# NOTE: if data elements
# have different labels in their 3rd
# fields, it is "easy" to propagate
# those labels into the overlaps.


print("case 1")
data = [(0, 10, 'parent')]
other_data = [(3, 4, 'a'), (3, 5, 'b'), (7, 9, 'a'), (8, 10, 'b')]
Q = buildQ(data, other_data)
overlaps = overlaps_from_Q(Q, 10)
for i in overlaps:
    print(i)

print("case 2")
data = [(0, 5, 'parent'), (7, 9, 'parent2')]
other_data = [(3, 4, 'a'), (3, 5, 'b'), (7, 9, 'a'), (8, 10, 'b')]
Q = buildQ(data, other_data)
overlaps = overlaps_from_Q(Q, 10)
for i in overlaps:
    print(i)
print("another method:")
Q = buildQ(data, other_data)
overlaps = overlaps_from_Q_no_push(Q, 10)
for i in overlaps:
    print(f"overlap = {i}")

print("case 2b")
data = [(0, 5, 'parent'), (7, 9, 'parent2')]
other_data = [(4, 5, 'a'), (5, 6, 'b'), (5, 6, 'a')]
Q = buildQ(data, other_data)
overlaps = overlaps_from_Q(Q, 10)
for i in overlaps:
    print(i)

print("case _1 from rust")
data = [(5, 9, 'parent')]
other_data = [(3, 7, 'a'), (4, 8, 'b')]
Q = buildQ(data, other_data)
overlaps = overlaps_from_Q(Q, 10)
for i in overlaps:
    print(i)

print("case _6 from rust")
data = [(0, 10, 'parent')]
other_data = [(3, 5, 'a'), (4, 9, 'b')]
Q = buildQ(data, other_data)
overlaps = overlaps_from_Q(Q, 10)
for i in overlaps:
    print(i)
print("another method:")
Q = buildQ(data, other_data)
overlaps = overlaps_from_Q_no_push(Q, 10)
for i in overlaps:
    print(f"overlap = {i}")
