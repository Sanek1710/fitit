import networkx as nx
# from fullgraph import edges
from edges import edges


def fmt_node(shape, coord, rot):
    return f"{shape}_{coord[0]}_{coord[1]}_{rot}"


g = nx.DiGraph()
g.add_edges_from(
    map(lambda x: (fmt_node(x[0], x[2], x[4]), fmt_node(x[1], x[3], x[4])),
        edges)
)
print(g)

meta = list(map(lambda x: x[2:], edges))

# print(meta)

while len(g.edges):
  out_leaves = list(n for n in g.nodes if g.out_degree(n) == 1)
  if out_leaves:
    outs = []
    for l in out_leaves:
        outs.extend(g.out_edges(l))
    nexts = list(map(lambda x: x[1], outs))
    print(outs)
    ins = []
    for n in nexts:
        ins.extend(g.in_edges(n))

    g.remove_edges_from(outs)
    g.remove_edges_from(ins)

  in_leaves = list(n for n in g.nodes if g.in_degree(n) == 1)
  if in_leaves:
    outs = []
    for l in in_leaves:
        outs.extend(g.in_edges(l))
    nexts = list(map(lambda x: x[1], outs))
    print(outs)
    ins = []
    for n in nexts:
        ins.extend(g.out_edges(n))

    g.remove_edges_from(outs)
    g.remove_edges_from(ins)
  
  if not in_leaves and not out_leaves:
    break

print(g.edges)