from main import main

def run():
  n_lst = [20] # num states
  k_lst = [2] # num signals
  # reward c - (d * abs(state - action))
  c = 1 # reward param 1
  d_lst = [0.3] # reward param 2
  i = 100_000 # number of rollouts per game
  r = 25 # number of generations between image creation
  repeats = 1
  
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