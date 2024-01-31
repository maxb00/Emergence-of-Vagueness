from main import main
from display import gen_graph, gen_extended_graph

def run():

  n_lst = [1000]
  k_lst = [4]
  c = 1
  d_lst = [100]
  w_lst = [0, 2, 5, 20]
  i_lst = [10000, 50000, 100000]
  avg_i = 10
  r = 1000000

  avg_vague_lvls = []
  success_rates = []

  for n in n_lst:
    for k in k_lst:
      for ind, d in enumerate(d_lst):
        for i in i_lst:
          avg_vague_lvls.append([])
          success_rates.append([])
          for null in [False]:
            for w in w_lst:
              sum_vague_lvls = 0
              sum_success_rates = 0
              for ii in range(avg_i):
                if ii == 0:
                  print(f"python main.py {n} {k} {n} {c} {d} {w}{' -n' if null else ''} {i} -r {i}")
                try:
                  vl, sr = main((n, k, n, c, d, w, null, i, i))
                  # print(f"#{ii+1} --> {vl}--{sr}")
                  sum_vague_lvls += vl
                  sum_success_rates += sr
                except ValueError:
                  print("==> Failed")
                  continue
            
              avg_vague_lvls[-1].append(sum_vague_lvls / avg_i)
              success_rates[-1].append(sum_success_rates / avg_i)
              print(f" => avg_vague_lvls = {sum_vague_lvls / avg_i}")
              print(f" => success_rates = {sum_success_rates / avg_i}")
  
  gen_extended_graph(avg_vague_lvls, success_rates)

run()