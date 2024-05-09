from main import main
import numpy as np
import math

def run():

  n_lst = [6]
  k_lst = [4]
  c = 1
  d_lst = [0.3]
  i = 5000
  r = 125
  repeat = 3

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