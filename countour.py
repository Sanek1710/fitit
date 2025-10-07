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

#%%

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

rnzids = []
for rshape in rshapes:
  countour = [shape[0] for shape in rshape]
  nzids = []
  for r in range(4):
    nzidx = np.nonzero(countour[r])[0]
    nzids.append((nzidx[0], nzidx[-1]))
  rnzids.append(nzids)
  countour = [np.trim_zeros(c) for c in countour]
  countours.append([c if c.all() else np.array([], dtype=int) 
                    for c in countour])

# ridshapes[1]
rnzids
# %%
board = np.pad(np.zeros((15, 35), dtype=int), 1, "constant", constant_values=1)

count = 0
show_limit = 10
edges = []
w = 1
for i1 in range(len(countours)):
  for r1 in range(4):
    if not len(countours[i1][r1]):
      continue

    ending = countours[i1][r1][-1]
    x1, y1 = rshapes[i1][r1].shape
    board[1:1+x1, 1:1+y1] += ridshapes[i1][r1]
    b1, e1 = rnzids[i1][r1]

    for i2 in range(len(countours)):
      if i2 == i1:
        continue
      for r2 in range(4):
        if not len(countours[i2][r2]):
          continue
        beginning = countours[i2][r2][0]
        if beginning == ending:
          continue
        x2, y2 = rshapes[i2][r2].shape
        xx = 1+e1+1
        if (board[1:1+x2, xx:xx+y2] * ridshapes[i2][r2]).any():
          continue
        board[1:1+x2, xx:xx+y2] += ridshapes[i2][r2]
        if not count_holes(board[:4]):
          count += 1
          edges.append(((i1, r1, len(countours[i1][r1]), countours[i1][r1][0]), 
                       (i2, r2, len(countours[i2][r2]), countours[i2][r2][0])))
          if show_limit:
            show_limit -= 1
            imdisplay(idshape_img(board))
        board[1:1+x2, xx:xx+y2] -= ridshapes[i2][r2]

    w -= len(countours[i1][r1])
    board[1:1+x1, 1:1+y1] -= ridshapes[i1][r1]

len(edges)
# %%
with open("countour-edges.txt", "w") as f:
  print("edges =", file=f)
  print(edges, file=f)

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

corners = [(40, 0), (53, 0), (3, 1), (5, 0), (11, 0), 
           (12, 0), (13, 0), (21, 0), (23, 0), (25, 0), 
           (28, 0), (33, 0), (38, 2), (42, 0), (42, 2),
          (43, 0), (44, 0), (51, 0), (52, 0), (56, 0),
          (57, 0), (37, 0)]
for n, r in corners:
  print((n, r), ",", sep="")
  imdisplay(idshape_img(np.rot90(idshapes[n], r)))
# %%

not_corners = {
  (53, 0),  (5, 0), (11, 0),(13, 0),(21, 0),
  (23, 0),(25, 0),(38, 2),(42, 2),(43, 0),
  (44, 0),(52, 0),
}
