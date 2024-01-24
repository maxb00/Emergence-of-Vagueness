from main import main
from display import gen_graph

def run():

  n_lst = [1000]
  k_lst = [4]
  c = 1
  d_lst = [100]
  w_lst = [0, 2, 10, 20]
  i_lst = [10000]
  avg_i = 10
  r = 1000000

  avg_vague_lvls = []
  success_rates = []

  for n in n_lst:
    for k in k_lst:
      for ind, d in enumerate(d_lst):
        for i in i_lst:
          success_rates.append([])
          for null in [False]:
            for w in w_lst:   
              sum_success_rates = 0
              for ii in range(avg_i):
                if ii == 0:
                  print(f"python main.py {n} {k} {n} {c} {d} {w}{' -n' if null else ''} {i} -r {i}")
                try:
                  vl = main((n, k, n, c, d, w, null, i, i))
                  print(f"#{ii+1} --> {vl}")
                  sum_success_rates += vl
                except ValueError:
                  print("==> Failed")
                  continue
            
              success_rates[-1].append(sum_success_rates / avg_i)
              print(f" => success_rates = {sum_success_rates / avg_i}")
  
  gen_graph(success_rates)

run()