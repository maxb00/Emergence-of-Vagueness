from main import main
import numpy as np
import math

def stats(inf, inf_sigstates):
  inf_states = np.sum(inf_sigstates, axis=0)

  print(f"Info measure = {inf}")
  print(f"Info measure by states:")

  for t1 in inf_states:
    for t2 in t1:
      for t3 in t2:
        print(f"{t3:.3f}", end=" ")
      print()
    print()

def run():
  t = 3 # num traits
  n_lst = [3] # num states
  k_lst = [6] # num signals
  c = 1 # reward param 1
  d_lst = [0.3] # reward param 2
  i = 100_000 # number of rollouts
  r = 250 # number of gens between image creation
  repeat = 10
  dist = "uniform"
  null = True

  new_size = [7]
  new_size.extend([3] * 3)

  total_payoff = 0
  total_inf = 0
  total_inf_state = np.zeros(tuple(new_size))
  total_w_inf = 0
  total_w_inf_state = np.zeros(tuple(new_size))


  for n in n_lst:
    for k in k_lst:
      for ind, d in enumerate(d_lst):
        for repeat_i in range(repeat):
          print(f"python main.py {t} {n} {k} {n} {c} {d}{' -n' if null else ''} {i} -r {r} #{repeat_i}")
          payoff, inf, inf_state, w_inf, w_inf_state = main((3, n, k, n, c, d, null, i, r, dist))
          total_payoff += payoff

          total_inf += inf
          total_inf_state += inf_state

          total_w_inf += w_inf
          total_w_inf_state += w_inf_state
            
  
  avg_payoff = total_payoff /repeat
  avg_info = total_inf / repeat
  avg_info_sigstates = total_inf_state / repeat
  avg_w_info = total_w_inf / repeat
  avg_w_info_sigstates = total_w_inf_state / repeat

  print(f"Number of runs: {repeat}")
  print(f"Expected payoff = {avg_payoff}\n")
  print("UNWEIGHTED")
  stats(avg_info, avg_info_sigstates)
  print()
  print("WEIGHTED")
  stats(avg_w_info, avg_w_info_sigstates)
 
run()