from main import main

def run():
  n_lst = [20, 30]
  k_lst = [2, 3, 4]
  c = 1
  d_lst = [6]
  w_lst = [2, 4]
  i = 1000000
  r = 12500

  for n in n_lst:
    for k in k_lst:
      for ind, d in enumerate(d_lst):
        for w in w_lst:
          for null in [False]:
            print(f"python main.py {n} {k} {n} {c} {d} {w}{' -n' if null else ''} {i} -r {r}")
            try:
              main((n, k, n, c, d, w, null, i, r))
            except ValueError:
              print("==> Failed")
              continue

run()