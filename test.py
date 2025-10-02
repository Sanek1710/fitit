import sys
import numpy as np
from PIL import Image
from collections import Counter

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


def print_shape(shape: np.ndarray, file =None):
  if file is None:
    file = sys.stdout
  for row in shape:
    for cell in row:
      if cell >= 12:
        print("\033[31m", end="")
        cell //= 12
      print(cell if cell != 0 else ' ', end=' ', file=file)
      # print("\033[0m", end="")
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


def shape_mask(shape: np.ndarray):
  return (shape & 0x3).astype(bool).astype(int)


def get_all_edges(shapes):
  from tqdm import tqdm

  max_size = max(map(len, shapes))

  index_iter = tqdm(range(len(shapes)), total = len(shapes), ncols=64)

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

              edges.append((s1, (xi - max_size, yi - max_size), s2, (xci, yci), i))

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
  (s1, (xi, yi), s2, (xci, yci), r2) = edge
  shape_con = shapes[s2]
  max_size = len(shape_con)

  shape_hole = np.pad(shapes[s1], pad_width=max_size, mode='constant',
                      constant_values=0)
  shape_hole_mask = shape_mask(shape_hole)
  xi += max_size
  yi += max_size

  for _ in range(r2):
    shape_con = np.rot90(shape_con)
  shape_conn_mask = shape_mask(shape_con)
  xx = xi - xci
  yy = yi - yci

  ss = shape_hole & 0x3
  ss[xx:xx+len(shape_con), yy:yy+len(shape_con)] += shape_con & 0x3
  # print_shape(ss)
  return trim_square(ss)


used_edges = []
shapes = read_shapes("pixel-fitit-poles-n-holes.png")
import os

saves = []

choosers = [642]
nchoose = 0

while len(shapes) != 1:
  shapes = [compile(shape) for shape in shapes]
  edges = get_all_edges(shapes)
  # join(shapes, (3, (3, 5), 39, (4, 6), 0))

  new_shapes = []

  cedges = [(edge[:2], edge[2:]) for edge in edges]
  # counts = Counter(map(lambda x: x[1], cedges))
  # cedges = list(filter(lambda x: counts[x[1]] == 1, cedges))
  counts = Counter(map(lambda x: x[0], cedges))
  act_cedges = list(filter(lambda x: counts[x[0]] == 1, cedges))
  if not act_cedges:
    # for i, shape in enumerate(shapes):
    #   print(i, file=f)
    #   print_shape(shape, file=f)
    nr = sorted(counts.values())[:5]
    print(nr)
    for i, cedge in enumerate(cedges):
      if counts[cedge[0]] in nr:
        n1, n2 = cedge
        edge = (*n1, *n2)
        print(i, ":", cedge)
        print_shape(join(shapes, edge))
    i = int(input())
    os.system("clear")
    print(i)

    # counts = Counter(map(lambda x: x[1], cedges))
    # act_cedges = list(filter(lambda x: counts[x[1]] == 1, cedges))
    # saves.append(shapes.copy())
    # act_cedges = list(filter(lambda x: counts[x[0]] == 2, cedges))
    act_cedges = cedges[i:i+1]
    print(act_cedges)
  cedges = act_cedges

  # clean_g = nx.DiGraph()
  # clean_g.add_edges_from(cedges)
  # with open("minigraph.dot", "w") as f:
  #   print(dump_dot(clean_g), file=f)

  added = set()
  used_now_edges = []
  for n1, n2 in cedges:
    if n1[0] in added or n2[0] in added:
      continue
    used_now_edges.append((n1, n2))
    added.add(n1[0])
    added.add(n2[0])
    edge = (*n1, *n2)
    new_shapes.append(join(shapes, edge))

  used_edges.append(used_now_edges)

  new_shapes.extend(
    shapes[i] & 0x3
    for i in range(len(shapes)) if i not in added
  )
  shapes = new_shapes

print(used_edges)
