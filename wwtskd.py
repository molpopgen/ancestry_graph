import tskit

tc = tskit.TableCollection(100)
n0 = tc.nodes.add_row(0, 1)
n1 = tc.nodes.add_row(0, 1)
n2 = tc.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)
n3 = tc.nodes.add_row(tskit.NODE_IS_SAMPLE, 0)

tc.edges.add_row(0, 45, n1, n2)
tc.edges.add_row(45, 100, n0, n2)
tc.edges.add_row(0, 73, n0, n3)
tc.edges.add_row(73, 100, n1, n3)

tc.sort()

idmap = tc.simplify(keep_unary=True)

ts = tc.tree_sequence()

print(idmap)

print(ts.draw_text())
