from PIL import Image
import numpy as np

im = Image.open("pixel-fitit.png")

heads = (0, 0, 0, 255)
necks = (63, 63, 116, 255)

arr = np.asarray(im)
# for row in arr:
#   for cell in row:
#     print(cell == )
# arr.shape
necks_mask = (arr == necks).all(axis=2)
heads_mask = (arr == heads).all(axis=2)

res = np.zeros(arr.shape[:2], dtype=int)
res[necks_mask] = 1
res[heads_mask] = 2



shapes = []
H, W = res.shape
for x in range(H // 16):
  for y in range(W // 16):
    chunk = res[16*x : 16*(x + 1), 16*y : 16*(y + 1)]
    chunkT = chunk.T
    h = chunk.any(axis=1).sum()
    w = chunkT.any(axis=1).sum()
    if not h and not w: continue
    side = max(h//2*2+1, w//2*2+1)
    shapes.append(chunk[:side, :side])

def print_it(label, a):
  strings = [
    ''.join(map(lambda x: '()' if x else '  ', row))
    for row in a
  ]
  print(f"Detail(\"{label}\", {{")
  for s in strings:
    print(f'  "{s}", //')
  print("}),")

shapes.sort(key=lambda x: len(x), reverse=True)
for i, shape in enumerate(shapes):
  print_it(f".{i:02}", shape)
