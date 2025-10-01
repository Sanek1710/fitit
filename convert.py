from PIL import Image
import numpy as np
import time
im = Image.open("pixel-fitit-poles-n-holes.png")

heads = (0, 0, 0, 255)
necks = (63, 63, 116, 255)

poles = (172, 50, 50, 255)
holes = (75, 105, 47, 255)

empties = (0, 0, 0, 0)

arr = np.asarray(im)


# for row in arr:
#   for cell in row:
#     print(cell == )
# arr.shape
necks_mask = (arr == necks).all(axis=2)
heads_mask = (arr == heads).all(axis=2)
poles_mask = (arr == poles).all(axis=2)
holes_mask = (arr == holes).all(axis=2)

res = np.zeros(arr.shape[:2], dtype=int)
res[necks_mask] = 1
res[heads_mask] = 2
res[poles_mask] = 3
res[holes_mask] = 4

p = 0

def shape_img(shape):
  size = len(shape)
  def mapper(x):
    if x == 0: return empties
    if x == 1: return necks
    if x == 2: return heads
    if x == 3: return poles
    if x == 4: return (*holes[:3], 50)
    if x == 50: return (75, 105, 47, 255)
    if x == 20: return (75, 47, 105, 255)
    if x == 40: return (105, 105, 47, 255)

    if x == 100: return (105, 47, 47, 255)
    if x == 120: return (47, 105, 47, 255)
    return (0, 0, 0, 128)
  arr = np.array([
    list(map(mapper, row))
    for row in shape
  ], dtype=np.uint8)
  return Image.fromarray(arr)

def count_holes(arr):
  from scipy.ndimage import binary_fill_holes
  # print(*arr.astype(bool).astype(int))
  # print(*np.array(binary_fill_holes(arr)).astype(int))
  # print((arr.astype(bool) != binary_fill_holes(arr)).astype(int))
  # print((arr.astype(bool) != binary_fill_holes(arr)).sum())
  return (arr.astype(bool) != binary_fill_holes(arr)).sum()



shapes = []
H, W = res.shape
for x in range(H // 16):
  for y in range(W // 16):
    chunk = res[16*x : 16*(x + 1), 16*y : 16*(y + 1)]
    chunk = chunk[::(1<<p),::(1<<p)]
    chunkT = chunk.T
    h = chunk.any(axis=1).sum()
    w = chunkT.any(axis=1).sum()
    if not h or not w: continue
    side = max(h//2*2+1, w//2*2+1)
    if not chunk[:side, :side].any(): continue
    shapes.append(chunk[:side, :side])

def mapval(x):
  if x == 1: return '><'
  if x == 2: return '()'
  if x == 3: return '[]'
  if x == 4: return '::'
  return '  '
  

def print_it(label, a):
  strings = [
    ''.join(map(mapval, row))
    for row in a
  ]
  print(f"Detail(\"{label}\", {{")
  for s in strings:
    print(f'  "{s}", //')
  print("}),")

def prints(shape):
  for row in shape:
    for cell in row:
      if cell >= 12:
        print("\033[31m", end="")
        cell //= 12
      print(cell if cell != 0 else ' ', end=' ')
      print("\033[0m", end="")
    print()


shapes.sort(key=lambda x: len(x), reverse=True)

all_shapes = np.zeros(res.shape)
idx = 0
for x in range(H // 16):
  if idx >= len(shapes): break
  for y in range(W // 16):
    shape = shapes[idx]
    idx += 1
    if idx >= len(shapes): break
    chunk = all_shapes[16*x : 16*(x + 1), 16*y : 16*(y + 1)]
    chunk[:len(shape), :len(shape)] = shape

shape_img(all_shapes).save("all-shapes.png")

from pathlib import Path
img_dir = Path("shapes")
for i, shape in enumerate(shapes):
  size = len(shape)
  for r in range(4):
    im = shape_img(shape)
    im\
      .resize((8*size, 8*size), Image.Resampling.NEAREST)\
      .save(img_dir / f"{i}.{r}.png")
    shape = np.rot90(shape)
    
# np.set_printoptions(threshold=10000000)


s1 = 0
s2 = 28

import sys

import os


cur = 0
iter = 0

S1 = None # 33
S2 = None # 38
DEBUG = False # True

edges = []

total_lines = ["digraph {"]
total_lines.append(f"  rankdir=TB;")
for i, _ in enumerate(shapes):
  total_lines.append(f"  subgraph cluster_{i} {{")
  
  total_lines.append(f"    label = {i};")
  total_lines.append(f"    style=filled;")
  total_lines.append(f"    color=lightgrey;")
  for r in range(4):
    total_lines.append(f'    _{i}_{r} [image="shapes/{i}.{r}.png", label=""];')
  total_lines.append(f"  }}")

for s1 in range(len(shapes)):
  if S1 is not None and s1 != S1: continue
  shape = np.array(shapes[s1])
  # print(shape)
  shape = np.pad(shape, pad_width=18, mode='constant',
                constant_values=0)
  x, y = np.where(shape == 4)
  
  print(cur, '/', len(shapes), file=sys.stderr, end="\r")
  cur += 1
  for xi, yi in zip(x, y):
    lines = []
    for s2 in range(len(shapes)):
      if S2 is not None and s2 != S2: continue
      if s2 == s1: continue
      shape_con = shapes[s2]
      xc, yc = np.where(shape_con == 3)
    # print(xc, yc)
      rotated = False
      for i in range(4):
        for xci, yci in zip(xc, yc):
          xx = xi - xci
          yy = yi - yci
          # print(xi, yi)
          ss = ((shape != 0) & (shape != 4))*1
          ss[xx:xx+len(shape_con), yy:yy+len(shape_con)] += \
              ((shape_con != 0) & (shape_con != 4))*2
          # sub_shape = \
          #   shape[xx:xx+len(shape_con), 
          #         yy:yy+len(shape_con)] * shape_con % 12
          # shape_img(20*ss).save("preview/preview.png")
          # time.sleep(10)
          
          if (ss != 3).all():
            # ss = ((shape != 0) & (shape != 4))*60
            # ss[xx:xx+len(shape_con), yy:yy+len(shape_con)] += \
            #   ((shape_con != 0) & (shape_con != 4))*70
            nholes = count_holes(ss)
            if nholes and nholes < 90: 
              ss = np.pad(ss, pad_width=1, mode='constant', constant_values=5)
              iter += 1
              if DEBUG:
                shape_img(20*ss).save("preview/preview.png")
                time.sleep(1)
              continue
            else:
              ss = np.pad(ss, pad_width=1, mode='constant', constant_values=6)
              if DEBUG:
                shape_img(20*ss).save("preview/preview.png")
                time.sleep(1)
            # lines.append(f"  {s1} -> {s2} [label=\"\"];")
            # if rotated: print("ROT!!!")
            lines.append(f"  _{s1}_{0} -> _{s2}_{i} [label=\"({xi-18}, {yi-18})\n({xci}, {yci})\"];")
            edges.append((s1, s2, (xi, yi), (xci, yci), i))
            # print(i, end="\"];\n")
            # print(i, ": ", f"({xci}, {yci}) -> ({xi}, {yi})", end="\"];\n")
            # print(xx, yy)
            # prints(ss)
        shape_con = np.rot90(shape_con)
        xc, yc = np.where(shape_con == 3)
        rotated += 1

      # print(len(lines))
    lines = list(set(lines))
    if lines and len(lines) < 20:
      total_lines.extend(lines)
      print("\n".join(lines))
  # break
# print(*shape, sep="\n")
total_lines.append("}")

with open("conns.dot", "w") as f:
  print("\n".join(total_lines), file=f)
with open("edges.py", "w") as f:
  print("edges =", edges, file=f)
os.system("dot -Tpng conns.dot -O")

exit()
for i, shape in enumerate(shapes):
  print_it(f".{i:02}", shape)
