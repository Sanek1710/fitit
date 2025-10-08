from countour_edges import edges
from collections import defaultdict
import networkx as nx

G = nx.DiGraph()
# idx, rotation, length, parity
G.add_edges_from(edges)



grouped_nodes = defaultdict(list)
for n in G.nodes:
  grouped_nodes[n[0]].append(n)

redges = []
for v in grouped_nodes.values():
  v.sort(key=lambda x: x[1], reverse=True)
  for n1, n2 in zip(v, v[1:] + [v[0]]):
    if n1[1] == (n2[1] + 1) % 4:
      redges.append((n1, n2))

G.add_edges_from(redges)
print(G)
G.remove_nodes_from([
  g for g in G.nodes 
  if G.in_degree(g) == 0 or\
  G.out_degree(g) == 0
])
G.remove_nodes_from([
  g for g in G.nodes 
  if G.in_degree(g) == 1 or\
  G.out_degree(g) == 1
])
print(G)


redges = []
grouped_nodes = defaultdict(list)
for n in G.nodes:
  grouped_nodes[n[0]].append(n)
for v in grouped_nodes.values():
  v.sort(key=lambda x: x[1], reverse=True)
  for n1, n2 in zip(v, v[1:] + [v[0]]):
    if n1[1] == (n2[1] + 1) % 4:
      redges.append((n1, n2))

nodes = list(G.nodes)

# grouped_nodes = defaultdict(list)
# for n in G.nodes:
#   grouped_nodes[n[0]].append(n)

# corners = []

# for v in grouped_nodes.values():
#   v.sort(key=lambda x: x[1], reverse=True)
#   for n1, n2 in zip(v[:-1], v[1:]):
#     if n1[1] == n2[1] + 1:
#       # G.add_edge(n1, n2)
#       corners.append(n2[:2])

# print(corners)
possible_corners = [(47, 2), (40, 0), (53, 2), (1, 2), (1, 0), (3, 1), (11, 1), (12, 1), (12, 0), (13, 1), (18, 1), (25, 1), (26, 1), (27, 0), (28, 0), (33, 0), (38, 1), (41, 2), (42, 0), (45, 2), (46, 2), (51, 1), (51, 0), (52, 2), (54, 2), (57, 0), (58, 1), (58, 0)]
possible_corners = [(47, 2), (1, 0), (12, 0), (17, 1), (27, 0), (45, 2), (46, 2), (51, 0), (58, 0)]
starts = []
for n in G.nodes:
  if n[:2] in possible_corners:
    starts.append(n)

print(G)

G.remove_edges_from(redges)
rotation_edges = dict(redges)
for k, v in rotation_edges.items():
  if k[0] != v[0]:
    print("OHNP")
with open("cgraph/cinput.h", "w") as f:
  print(
"""struct Node {
  int id;
  int r;
  int w;
  int par;
};

""", file=f)
  print("std::vector<Node> nodes = {", file=f)
  nodes = list(G.nodes)
  for node in nodes:
    print("  {{ {}, {}, {}, {}, }}, ".format(*node), file=f)
  print("};\n", file=f)

  print("std::vector<std::vector<int>> edges = {", file=f)
  for node in nodes:
    tos = list(G[node])
    print(f"  /*{nodes.index(node)} = */ {{ ", end="", file=f)
    for to_node in tos:
      print(f"{nodes.index(to_node)}, ", end="", file=f)
    print(f"}},", file=f)
  print("};\n", file=f)

  print("std::vector<int> starts = {\n  ", end="", file=f)
  for start in starts:
    print(f"{nodes.index(start)}, ", end="", file=f)
  print("\n};\n", file=f)

  print("std::vector<int> rotation_edges = {", file=f)
  for node in nodes:
    e = rotation_edges.get(node, None)
    if e is None:
      print(f"-1, ", end="", file=f)
    else:
      print(f"{nodes.index(e)}, ", end="", file=f)
  print("};\n", file=f)
