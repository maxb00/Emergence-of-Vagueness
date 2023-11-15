from main import main

def run():
  n_lst = [30]
  k_lst = [4]
  c = 1
  d_lst = [0.5, 0.3, 0.25]
  i = 2000
  r = 25

  for n in n_lst:
    for k in k_lst:
      for ind, d in enumerate(d_lst):
        for null in [False, True]:
          print(f"python main.py {n} {k} {n} {c} {d}{' -n' if null else ''} {i} -r {r}")
          try:
            main((n, k, n, c, d, null, i, r))
          except ValueError:
            print("==> Failed")
            continue

run()