# %%

from collections import defaultdict
import networkx as nx
import sys
import numpy as np
from PIL import Image
from collections import Counter


def scale_image(self, scale):
  w, h = self.size
  return self.resize((int(scale*w), int(scale*h)), Image.Resampling.NEAREST)


setattr(Image.Image, "scale", scale_image)


HEAD_VAL = 2
NECK_VAL = 1
POLE_VAL = 6
HOLE_VAL = 4
EMPTY_VAL = 0


def compile(shape: np.ndarray):
  from scipy.ndimage import convolve
  KERNEL = np.array([[0, 1, 0],
                     [1, 0, 1],
                     [0, 1, 0]])
  nneighbours = convolve(
      shape.astype(bool).astype(int),
      KERNEL, mode='constant', cval=0)

  poles = ((nneighbours == 1) & (shape == 2))
  holes = ((nneighbours == 3) & (shape == 0))

  return shape | (poles << 2) | (holes << 2)


def count_holes(shape: np.ndarray):
  from scipy.ndimage import binary_fill_holes
  return (shape.astype(bool) != binary_fill_holes(shape)).sum()


def print_shape(shape: np.ndarray, file=sys.stdout):
  for row in shape:
    for cell in row:
      print(cell if cell != 0 else ' ', end=' ', file=file)
    print(file=file)


def shrink_shape(shape: np.ndarray):
  h = shape.any(axis=1).sum()
  w = shape.T.any(axis=1).sum()
  side = max(h, w) + 1
  return shape[:side, :side]


def read_shapes(filename: str, compile=False):
  COLOR_HEAD = (0, 0, 0, 255)
  COLOR_NECK = (63, 63, 116, 255)
  COLOR_POLE = (172, 50, 50, 255)
  COLOR_HOLE = (75, 105, 47, 255)
  COLOR_EMPTY = (0, 0, 0, 0)

  im = Image.open(filename)
  arr = np.asarray(im)
  heads_mask = (arr == COLOR_HEAD).all(axis=2)
  necks_mask = (arr == COLOR_NECK).all(axis=2)
  poles_mask = (arr == COLOR_POLE).all(axis=2)
  holes_mask = (arr == COLOR_HOLE).all(axis=2)

  res = np.zeros(arr.shape[:2], dtype=int)

  res[heads_mask] = HEAD_VAL
  res[necks_mask] = NECK_VAL

  res[poles_mask] = POLE_VAL if compile else HEAD_VAL
  res[holes_mask] = HOLE_VAL if compile else EMPTY_VAL

  BLOCK_SIZE = 16
  blocks = res.reshape(res.shape[0] // BLOCK_SIZE, BLOCK_SIZE,
                       res.shape[1] // BLOCK_SIZE, BLOCK_SIZE) \
      .transpose(0, 2, 1, 3)
  shapes = [
      shrink_shape(chunk)
      for row in blocks
      for chunk in row if chunk.any()
  ]
  shapes.sort(key=lambda x: (x.size, x.astype(bool).sum()), reverse=True)
  return shapes


def idshape_img(shape):
  def get_color(idx):
    if idx == 0:
      return (0, 0, 0, 0)
    idx = (idx*6 % 63)
    r, g, b = ((idx >> 4) & 0x3), ((idx >> 2) & 0x3), (idx & 0x3)
    r = 48*r + 32
    g = 48*g + 48
    b = 48*b + 64
    return r, g, b, 255
  arr = np.array([[get_color(cell) for cell in row]
                  for row in shape], dtype=np.uint8)
  return Image.fromarray(arr)  # pyright: ignore[reportCallIssue]


def print_image(im):
  def ansi_rgb(rgba):
    return "\x1b[38;2;{};{};{}m".format(*rgba[:3])
  for row in np.asarray(im):
    for cell in row:
      if not cell[:3].any():
        print("  ", end="")
      else:
        print(f"{ansi_rgb(cell)}[]", end="")
    print()
  print("\x1b[0m")


def shape_mask(shape: np.ndarray):
  return (shape & 0x3).astype(bool).astype(int)


def get_all_edges(shapes):
  from tqdm import tqdm
  shapes = [compile(shape) for shape in shapes]

  max_size = max(map(len, shapes))

  index_iter = tqdm(range(len(shapes)), total=len(shapes), ncols=64)

  edges = []
  for s1 in index_iter:
    shape_hole = np.pad(shapes[s1], pad_width=max_size, mode='constant',
                        constant_values=0)
    # print_shape(shape_hole)
    shape_hole_mask = shape_mask(shape_hole)
    x, y = np.where(shape_hole == HOLE_VAL)
    for xi, yi in zip(x, y):

      for s2 in range(len(shapes)):
        if s2 == s1:
          continue
        shape_con = shapes[s2]

        for i in range(4):
          shape_conn_mask = shape_mask(shape_con)
          xc, yc = np.where(shape_con == POLE_VAL)

          for xci, yci in zip(xc, yc):
            xx = xi - xci
            yy = yi - yci

            ss = np.array(shape_hole_mask)
            ss[xx:xx+len(shape_conn_mask), yy:yy+len(shape_conn_mask)] += \
                2*shape_conn_mask

            if (ss != 3).all():
              nholes = count_holes(ss)
              if nholes and nholes < 9000:
                continue

              edges.append(
                  ((s1, (xi - max_size, yi - max_size)), (s2, (xci, yci), i)))

          shape_con = np.rot90(shape_con)

  return edges


def trim_square(matrix: np.ndarray):
  matrix = matrix[~np.all(matrix == 0, axis=1)].T
  matrix = matrix[~np.all(matrix == 0, axis=1)].T
  rows, cols = matrix.shape

  max_dim = max(rows, cols)
  pad_rows = max_dim - rows
  pad_cols = max_dim - cols

  return np.pad(matrix, ((0, pad_rows), (0, pad_cols)),
                mode='constant', constant_values=0)


def join(shapes, edge):
  ((s1, (xi, yi)), (s2, (xci, yci), r2)) = edge
  shape_con = shapes[s2]
  max_size = len(shape_con)

  shape_hole = np.pad(shapes[s1], pad_width=max_size, mode='constant',
                      constant_values=0)
  xi += max_size
  yi += max_size

  for _ in range(r2):
    shape_con = np.rot90(shape_con)
  xx = xi - xci
  yy = yi - yci

  ss = shape_hole
  ss[xx:xx+len(shape_con), yy:yy+len(shape_con)] += shape_con

  return trim_square(ss)


def join_all(shapes, edges):
  added = set()
  new_shapes = []
  for n1, n2 in edges:
    if n1[0] in added or n2[0] in added:
      continue
    added.add(n1[0])
    added.add(n2[0])
    new_shapes.append(join(shapes, (n1, n2)))

  rest = [shape
          for i, shape in enumerate(shapes)
          if i not in added]
  return new_shapes, rest


def imdisplay(im):
  try:
    if not sys.stdout.isatty():
      display(im.scale(8))  # type: ignore
      return
  except:
    pass

  def ansi_rgb(rgba):
    return "\x1b[38;2;{};{};{}m".format(*rgba[:3])
  for row in np.asarray(im):
    for cell in row:
      if not cell[:3].any():
        print("  ", end="")
      else:
        print(f"{ansi_rgb(cell)}[]", end="\x1b[0m")
    print()
  print()


# %%

shapes = read_shapes("pixel-fitit.png")
idshapes = [
    shape.astype(bool) * i
    for i, shape in enumerate(shapes, 1)
]
edges = get_all_edges(shapes)

colors = np.array([range(5*13)], dtype=np.uint8)
arr = colors.reshape((5, 13))
imdisplay(idshape_img(arr))

jnd = join(idshapes, ((0, (3, 7)), (1, (0, 0), 0)))
imdisplay(idshape_img(idshapes[1]))
imdisplay(idshape_img(jnd))

used_edges = []

# %%


def preview(appl_edges):
  for edge in appl_edges:
    print(edge)
    imdisplay(idshape_img(join(idshapes, edge)))


def apply(appl_edges):
  global shapes
  global idshapes
  global edges
  joined_shapes, rest_shapes = join_all(shapes, appl_edges)
  joined_idshapes, rest_idshapes = join_all(idshapes, appl_edges)
  shapes = joined_shapes + rest_shapes
  idshapes = joined_idshapes + rest_idshapes
  edges = get_all_edges(shapes)


def filter_edges_limit(edges, limit=1):
  # edges = [e for e in edges if e[0][0] == 6 and e[0][1][0] == (5)]
  counts = Counter(map(lambda x: x[0], edges))
  act_edges = list(filter(lambda x: counts[x[0]] <= limit, edges))
  return act_edges

def show(topk=10):
  for i, shape in enumerate(idshapes[:topk]):
    print(i)
    imdisplay(idshape_img(shape))


apply(filter_edges_limit(edges))

# %%


apply_list = [
    [],
    [
        ((21, (5, 7)), (31, (0, 8), 3)),
        ((19, (3, 5)), (15, (10, 6), 2)),
        ((19, (3, 5)), (15, (10, 6), 2)),
        ((7, (3, 5)), (45, (0, 0), 0))
    ], [
        ((9, (5, 1)), (15, (4, 0), 0)),
        ((30, (1, 3)), (31, (1, 3), 2)),
    ], [
        ((20, (1, 5)), (49, (0, 0), 0)),
        ((8, (4, 3)), (39, (7, 4), 1))
    ], [
        ((0, (9, 6)), (17, (2, 4), 3))
    ], [
        ((0, (12, 7)), (3, (0, 7), 3))
    ], [
        ((1, (3, 5)), (43, (0, 4), 0))
    ], [
        ((1, (10, 9)), (47, (1, 3), 2))
    ], [
        ((1, (10, 9)), (47, (1, 3), 2))
    ], [
        ((12, (1, 3)), (10, (7, 11), 2))
    ], [
        ((0, (5, 11)), (28, (3, 3), 2))
    ], [
        ((16, (3, 3)), (18, (1, 5), 2))
    ], [
        ((0, (3, 1)), (8, (10, 6), 1))
    ], [
        ((15, (3, 3)), (22, (0, 0), 0))
    ],
]

for filtered_edges in apply_list:
  apply(filtered_edges)
  apply(filter_edges_limit(edges))

# %%


# %%


def filter_possible(edges, possible_edges):
  return [e for e in edges
          if (e[0][0], e[1][0]) in possible_edges]


def filter_left(edges, left, coord=None):
  if coord:
    search = (left, coord)
    return [e for e in edges if e[0][:2] == search]
  if not isinstance(left, list):
    left = [left]
  return [e for e in edges if e[0][0] in left]


preview(filter_edges_limit(edges, 3))
# show(20)

# %%
# preview(filter_left(edges, 0))
show(10)

# %%

nidshapes, nrest_idshapes = join_all(idshapes, filtered_edges)
print(len(nidshapes))
for shape in nidshapes[:20]:
  imdisplay(idshape_img(shape))
# %%

for shape in shapes[:8]:
  imdisplay(idshape_img(shape))


# %%

# %%


def dump_dot(g: nx.DiGraph):
  lines = ["digraph {"]
  for node in g.nodes:
    lines.append(f'  "{node}";')
  for n1, n2 in g.edges:
    lines.append(f'  "{n1}" -> "{n2}";')
  lines.append("}")
  return "\n".join(lines)


def dump_dot_record(g: nx.DiGraph):
  lines = ["digraph {"]
  lines.append("  node [shape=record];")
  holes = defaultdict(list)
  for node in g.nodes:
    holes[node[0]].append(node[1])
  for node in g.nodes:
    ports = "|".join(f"<{port}>{port}" for port in holes[node])
    lines.append(f'  "{node}" [label="{{{node}|{{{ports}}}}}"];')
  for n1, n2 in g.edges:
    n1, port1 = n1
    n2, port2 = n2
    lines.append(f'  "{n1}":"{port1}" -> "{n2}":"{port2}";')
  lines.append("}")
  return "\n".join(lines)


# %%
# apply(filter_edges_limit(edges))
edges2 = filter_edges_limit(edges, 4)
len(edges2)

for (n1, (x1, y1)), (n2, (x2, y2), r2) in edges2:
  print(n1, n2)
# nx.DiGraph(edges2)

# %%
redges = []

for sh1, sh2 in edges:
  crd = np.zeros(shapes[sh2[0]].shape)
  crd[sh2[1][0], sh2[1][1]] = 1
  crd = np.rot90(crd, (4 - sh2[2]) % 4)
  x, y = np.where(crd)
  x2, y2 = x[0], y[0]
  x1, y1 = sh1[1]
  redges.append((sh1, (sh2[0], (x2, y2))))

# %%


def dump_dot_record(g: nx.DiGraph):
  lines = ["digraph {"]
  lines.append("  node [shape=record];")
  holes = defaultdict(lambda: (set(), set()))
  for (n1, port1), (n2, port2) in g.edges:
    holes[n2][0].add(port2)
    holes[n1][1].add(port1)
  for n1, (inports, outports) in holes.items():
    inports = "|".join(f"<{port}>{port}" for port in inports)
    outports = "|".join(f"<{port}>{port}" for port in outports)
    lines.append(f'  "{n1}" [label="{{{{{inports}}}|{n1}|{{{outports}}}}}"];')
  for n1, n2 in g.edges:
    n1, port1 = n1
    n2, port2 = n2
    lines.append(f'  "{n1}":"{port1}" -> "{n2}":"{port2}";')
  lines.append("}")
  return "\n".join(lines)


redges2 = filter_edges_limit(redges, 5)
print(len(redges2))
G = nx.DiGraph()
G.add_edges_from(redges2)
len(G.edges)


with open("minidot.dot", "w") as f:
  f.write(dump_dot(G))


# %%
print(len(redges))
print(len(list(set(redges))))

# %%

for i, shape in enumerate(shapes):
  nholes = (compile(shape) == HOLE_VAL).sum()
  if nholes >= 0:
    edges5 = [e for e in edges if e[0][0] == i]
    print(i)
    print(nholes)
    print(len(edges5))
    imdisplay(idshape_img(idshapes[i]))

# %%


i = 10
display_limit = 12
ndisplayed = 0

# break


def work_on(i, ids):
  global display_limit
  edgesi = [e for e in edges if e[0][0] == i]
  portsi = {
      edgei[0][1]: []
      for edgei in edgesi
  }
  for edgei in edgesi:
    portsi[edgei[0][1]].append(edgei)

  portsikeys = list(portsi.keys())
  # portsikeys.sort(key = lambda x: len(portsi[x]))
  portsikeys = [portsikeys[i] for i in ids]

  # portsikeys = [portsikeys[0], portsikeys[2]]
  print(portsikeys)

  shapei = shapes[i]
  visited = set()

  max_size = max(map(len, shapes))
  shape_hole = np.pad(idshapes[i], pad_width=max_size, mode='constant',
                      constant_values=0)

  bed_edges = []

  work_on.ndisplayed = 0
  work_on.display_limit = 0

  def recursion(idx=0):
    global display_limit
    if work_on.ndisplayed > display_limit:
      return True
    if idx >= len(portsikeys):
      imdisplay(idshape_img(shape_hole))
      work_on.ndisplayed += 1
      return True
    x, y = portsikeys[idx]
    if shape_hole[x + max_size, y + max_size]:
      return recursion(idx+1)

    is_valid = False
    for polee in portsi[portsikeys[idx]]:
      ((s1, (xi, yi)), (s2, (xci, yci), r2)) = polee
      if s2 in visited:
        continue

      visited.add(s2)

      shape_con = idshapes[s2]
      xi += max_size
      yi += max_size

      for _ in range(r2):
        shape_con = np.rot90(shape_con)
      xx = xi - xci
      yy = yi - yci

      ss = shape_hole
      # overlap = np.array(shape_hole)
      # overlap[xx:xx+len(shape_con), yy:yy+len(shape_con)] += shape_con
      # imdisplay(idshape_img(overlap))
      if not (ss[xx:xx+len(shape_con), yy:yy+len(shape_con)] * shape_con).any():
        ss[xx:xx+len(shape_con), yy:yy+len(shape_con)] += shape_con
        if not count_holes(shape_hole):
          is_valid |= recursion(idx+1)
        ss[xx:xx+len(shape_con), yy:yy+len(shape_con)] -= shape_con
      else:
        bed_edges.append(polee)
      visited.remove(s2)

    if not is_valid:
      bed_edges.append(polee)

    return False
  recursion()


work_on(i, [2, 3])

for k in portsikeys:
  print(len(portsi[k]))
# ports10

print(bed_edges)


# %%
show(11)
# %%
len(edges9)
# %%


((s1, (xi, yi)), (s2, (xci, yci), r2)) = edge
shape_con = shapes[s2]
max_size = len(shape_con)

shape_hole = np.pad(shapes[s1], pad_width=max_size, mode='constant',
                    constant_values=0)
xi += max_size
yi += max_size

for _ in range(r2):
  shape_con = np.rot90(shape_con)
xx = xi - xci
yy = yi - yci

ss = shape_hole
ss[xx:xx+len(shape_con), yy:yy+len(shape_con)] += shape_con


# %%
counter = 0
print(len(shapes))
for shape in shapes:
  lines = []
  counter += 1
  first = True
  for row in shape:
    if not row.any() and not first:
      break
    first = False
    lines.append(''.join('o' if cell == 2 else 'x' if cell == 1 else '.'
                         for cell in row))
  print(len(lines), '\n'.join(lines), sep='\n')


# %%


npeaks = []
nholes = []

for shape in shapes:
  npeaks.append((shape == 2).sum())
  nholes.append((shape == 1).sum())


# %%
def count_combinations_to_sum(numbers, target_sum):
  """
  Calculates the number of combinations to reach a target sum using elements
  from an array of numbers, where each number can be used multiple times.

  Args:
      numbers (list): A list of integers representing the available numbers.
      target_sum (int): The target sum to achieve.

  Returns:
      int: The number of distinct combinations that sum up to the target_sum.
  """

  # Initialize a DP array where dp[i] will store the number of ways to make sum i.
  # dp[0] is 1 because there's one way to make a sum of 0 (by choosing no numbers).
  dp = [0] * (target_sum + 1)
  dp[0] = 1

  # Iterate through each number in the input array
  for num in numbers:
      # For each number, iterate from the number itself up to the target_sum
      # This ensures that we consider using the current 'num' to build up sums
    for i in range(num, target_sum + 1):
        # Add the number of ways to make sum (i - num) to dp[i].
        # This represents adding 'num' to all combinations that sum to (i - num).
      dp[i] += dp[i - num]

  return dp[target_sum]


nholes.sort()

# Example Usage:
numbers_array = nholes
target = 17*18
result = count_combinations_to_sum(numbers_array, target)
print(
    f"Number of combinations to reach sum {target} with {numbers_array}: {result}")

numbers_array_2 = npeaks
target_2 = 17*17
result_2 = count_combinations_to_sum(numbers_array_2, target_2)
print(
    f"Number of combinations to reach sum {target_2} with {numbers_array_2}: {result_2}")
# %%

countours = []
rshapes = [
  [np.rot90(sh, r) for r in range(4)]
  for sh in shapes
]
ridshapes = [
  [np.rot90(sh, r) for r in range(4)]
  for sh in idshapes
]
for i in range(2):
  rshapes = [
      [
        shape[~np.all(shape == 0, axis=1)].T
        for shape in rshape
      ]
      for rshape in rshapes
  ]
  ridshapes = [
      [
        shape[~np.all(shape == 0, axis=1)].T
        for shape in rshape
      ]
      for rshape in ridshapes
  ]


# ridshapes[1]
# %%

for rshape in rshapes:
  countour = [np.trim_zeros(shape[0]) for shape in rshape]
  countours.append([c if c.all() else np.array([], dtype=int) 
                    for c in countour])

countours[0]
# show(68)
# %%
board = np.pad(np.zeros((35, 35), dtype=int), 1, "constant", constant_values=1)


for i1 in range(len(countours)):
  for r1 in range(4):
    w = 1
    x, y = rshapes[i1][r1].shape
    board[1:1+x, w:w+y] += ridshapes[i1][r1]



    board[1:1+x, w:w+y] -= ridshapes[i1][r1]


# %%

board = np.pad(np.zeros((35, 35), dtype=int), 1, "constant", constant_values=1)

visited = 0
def try_it(w = 1):
  global visited
  if w == 36: 
    if count_holes(board[1:-1, 1:-1]):
      return False
    imdisplay(idshape_img(board))
    return True
  for i in range(len(countours)):
    if visited & (1 << i):
      continue
    for r in range(4):
      if len(countours[i][r]) and countours[i][r][0] == 2:
        try:
          if (board[1:1+rshapes[i][r].shape[0], w:w+rshapes[i][r].shape[1]] *\
            ridshapes[i][r]).any():
            continue
          board[1:1+rshapes[i][r].shape[0], w:w+rshapes[i][r].shape[1]] \
            += ridshapes[i][r]
          visited |= (1 << i)
          # w += len(countours[i][r])
          # i += 1
          if try_it(w + len(countours[i][r])):
            return True

          visited &= ~(1 << i)
          board[1:1+rshapes[i][r].shape[0], w:w+rshapes[i][r].shape[1]] \
            -= ridshapes[i][r]
        except:
          pass
  return False

try_it()

imdisplay(idshape_img(board))

# %%
rshapes[1]
# %%
