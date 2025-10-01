from PIL import Image
import numpy as np

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
# np.set_printoptions(threshold=10000000)


s1 = 0
s2 = 28

import sys


total = len(shapes) * len(shapes)
cur = 0

print("digraph {")

for s1 in range(len(shapes)):
  # if s1 != 33: continue
  shape = shapes[s1]
  # print(shape)
  shape = np.pad(shape, pad_width=18, mode='constant',
                constant_values=0)
  x, y = np.where(shape == 4)
  
  print(cur, '/', total, file=sys.stderr, end="\r")
  cur += 1
  for xi, yi in zip(x, y):
    lines = []
    for s2 in range(s1 + 1, len(shapes)):
    # if s2 != 63: continue
      shape_con = shapes[s2]
      xc, yc = np.where(shape_con == 3)
    # print(xc, yc)
      for i in range(4):
        for xci, yci in zip(xc, yc):
          xx = xi - xci
          yy = yi - yci
          # print(xi, yi)
          sub_shape = \
            shape[xx:xx+len(shape_con), 
                  yy:yy+len(shape_con)] * shape_con % 12
          if (sub_shape == 0).all():
            ss = shape.copy()
            ss[xx:xx+len(shape_con), 
                  yy:yy+len(shape_con)] += 12*shape_con
            lines.append(f" {s1} -> {s2} [label=\"\"];")
            # print(s1, "->", s2, end=" [label=\"")
            # # print(i, end="\"];\n")
            # print(i, ": ", xci, yci, "->", xi, yi, end="\"];\n")
            # print(xx, yy)
            # prints(ss)
      shape_con = np.rot90(shape_con)
      xc, yc = np.where(shape_con == 3)

      # print(len(lines))
    lines = list(set(lines))
    if lines and len(lines) < 4:
      print("\n".join(lines))
  # break
# print(*shape, sep="\n")
print("}")

exit()
for i, shape in enumerate(shapes):
  print_it(f".{i:02}", shape)
