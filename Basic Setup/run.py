from main import main

def run():
  n_lst = [20, 30]
  k_lst = [2]
  # reward c - (d * abs(state - action))
  c = 1 # reward param 1
  d_lst = [0.5, 0.3, 0.25] # reward param 2
  i = 10000
  r = 100
  repeats = 10

  for n in n_lst:
    for k in k_lst:
      for ind, d in enumerate(d_lst):
        for repeat in range(repeats):
          print(f"python main.py {n} {k} {n} {c} {d} -n {i} -r {r}")
          # try:
          #   
          # except ValueError:
          #   print("==> Failed")
          #   continue
          main((n, k, n, c, d, True, i, r, repeat))

run()