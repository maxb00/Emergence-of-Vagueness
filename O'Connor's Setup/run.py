from main import main
from display import gen_graph

def run():

  # gen_graph([
  #   [0.02972002328828833, 0.029756293761484674, 0.029752217632661032, 0.030110657440503907, 0.034134998435105354, 0.06841502983041481],
  #   [0.09205175042280934, 0.11598154535049943, 0.14435088012378508, 0.24258834640687452, 0.5398414013028098, 0.5014354111264238],
  #   [0.7782322114818868, 0.7769392448599689, 0.8470317274604855, 0.8133006032100287, 0.6823790274900483, 0.5007409157135698]
  # ])
  # exit(0)

  n_lst = [1000]
  k_lst = [4]
  c = 1
  d_lst = [100]
  w_lst = [0, 2, 10, 20]
  i_lst = [1000000] 
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
              for ii in range(1):
                print(f"python main.py {n} {k} {n} {c} {d} {w}{' -n' if null else ''} {i} -r {i} #{ii+1}", end="")
                try:
                  sr = main((n, k, n, c, d, w, null, i, i))
                  print(f" --> {sr}")
                  sum_success_rates += sr
                except ValueError:
                  print("==> Failed")
                  continue
            
              success_rates[-1].append(sum_success_rates / 1)
              print(f" => avg_vague_lvl = {sum_success_rates / 1}")
  
  gen_graph(success_rates)

run()