import networkx as nx
# from fullgraph import edges
from graph_edges import edges


def fmt_node(shape, coord, rot = 0):
    return f"{shape}_{coord[0]}_{coord[1]}_{rot}"


g = nx.DiGraph()
g.add_edges_from(
    map(lambda x: (fmt_node(*x[:2]), fmt_node(*x[2:])),
        edges)
)
print(g)

meta = list(map(lambda x: x[2:], edges))

# print(meta)

clean_g = nx.DiGraph()

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

    clean_g.add_edges_from(outs)
    clean_g.add_edges_from(ins)
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

    clean_g.add_edges_from(outs)
    clean_g.add_edges_from(ins)
    g.remove_edges_from(outs)
    g.remove_edges_from(ins)
  
  if not in_leaves and not out_leaves:
    break

print(clean_g)

def dump_dot(g: nx.DiGraph):
  lines = ["digraph {"]
  for node in g.nodes:
    lines.append(f"  {node};")
  
  for n1, n2 in g.edges:
    lines.append(f"  {n1} -> {n2};")
  lines.append("}")
  return "\n".join(lines)

with open("minigraph.dot", "w") as f:
  print(dump_dot(clean_g), file=f)
