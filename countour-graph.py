from countour_edges import edges
from collections import defaultdict
import networkx as nx

G = nx.DiGraph()
G.add_edges_from(edges)


grouped_nodes = defaultdict(list)
for n in G.nodes:
  grouped_nodes[n[0]].append(n)

for v in grouped_nodes.values():
  v.sort(key=lambda x: x[3])
  for n1, n2 in zip(v[:-1], v[1:]):
    if n1[1] + 1 == n2[2]:
      G.add_edge(n1, n2)

print(G)
G.remove_nodes_from([
  g for g in G.nodes 
  if G.in_degree(g) == 0 or\
  G.out_degree(g) == 0
])
print(G)

nodes = list(G.nodes)

nodes.sort(key=lambda x: x[2])
target = (4*35-4)
max_limit = 0
for max_limit in range(len(nodes)):
  target -= nodes[max_limit][2]
  if target <= 0:
    break

nodes.sort(key=lambda x: x[2], reverse=True)
target = (4*35-4)
min_limit = 0
for min_limit in range(len(nodes)):
  target -= nodes[min_limit][2]
  if target <= 0:
    break
target = (4*35-4)

print("min limit:", min_limit)
print("max limit:", max_limit)

print(G)
for cycle in nx.simple_cycles(G, 48):
  if min_limit <= len(cycle) and len(cycle) <= max_limit:
    weight = sum(map(lambda x: x[2], cycle))
    if weight != target:
      continue
    print(cycle)
    print(weight, weight > target)
    break
print(len(G.nodes))

# print(sum(map(lambda x: x[2], G.nodes)) / len(G.nodes))
# print((4*35+4) // 3)




print([
  g for g in G.nodes 
  if G.in_degree(g) == 1 or\
  G.out_degree(g) == 1
])



grouped_nodes = defaultdict(list)
for n in G.nodes:
  grouped_nodes[n[0]].append(n)

corners = []
for v in grouped_nodes.values():
  v.sort(key=lambda x: x[3])
  for n1, n2 in zip(v[:-1], v[1:]):
    if n1[1] + 1 == n2[2]:
      corners.append(n1[:2])
      print(n1, n2)

print(corners)