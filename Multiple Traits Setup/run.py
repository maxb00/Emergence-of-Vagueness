from main import main

def run():
  n_lst = [3, 4, 5, 6, 7, 8, 9, 10]
  k_lst = [2]
  c = 1
  d_lst = [0.5]
  i = 10000
  r = 250

  for n in n_lst:
    for k in k_lst:
      for ind, d in enumerate(d_lst):
        for null in [False]:
          print(f"python main.py {2} {n} {k} {n} {c} {d}{' -n' if null else ''} {i} -r {r}")
          try:
            main((2, n, k, n, c, d, null, i, r))
          except ValueError:
            print("==> Failed")
            continue

run()