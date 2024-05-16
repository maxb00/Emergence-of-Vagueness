from main import main
import numpy as np
import math

def run():

  n_lst = [3]
  k_lst = [4, 5]
  c = 1
  d_lst = [0.5]
  i = 4000
  r = 100
  repeat = 5

  for n in n_lst:
    for k in k_lst:
      for ind, d in enumerate(d_lst):
        for null in [False]:
          for repeat_i in range(repeat):
            print(f"python main.py {2} {n} {k} {n} {c} {d}{' -n' if null else ''} {i} -r {r} #{repeat_i}")
            try:
              main((2, n, k, n, c, d, null, i, r))
            except ValueError:
              print("==> Failed")
              continue

run()