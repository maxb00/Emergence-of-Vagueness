from main import main
from display import gen_graph, gen_extended_graph, gen_extended_graph2, gen_comp_graph

def run():

  n_lst = [[20], [100], [200], [50], [1000]]
  k_lst = [[3], [5], [40], [10], [4]]
  c = 1
  d_lst = [[6], [10], [4], [4], [100]]
  w_lst = [
    [0, 2, 4, 6],
    [0, 4, 8, 12, 16],
    [0, 0.25, 0.5, 1, 2, 4],
    [0, 0.25, 0.5, 1, 2, 4],
    [0, 2, 10, 20]
  ]
  i_lst = [
    [1000000], [1000000], [1000000], [1000000], [1000000]
  ]
  avg_i = [2, 2, 1, 2, 1]
  r = 1000000

  for fig in range(5):
    avg_vague_lvls = [[] for _ in range(8)]
    success_rates = [[] for _ in range(8)]

    for n in n_lst[fig]:
      for k in k_lst[fig]:
        for ind, d in enumerate(d_lst[fig]):
          for i in i_lst[fig]:
            avg_vague_lvls.append([])
            success_rates.append([])
            for null in [False]:
              for w in w_lst[fig]:
                sum_vague_lvls1 = 0
                sum_success_rates1 = 0
                # sum_vague_lvls1n = 0
                # sum_success_rates1n = 0
                sum_vague_lvls2 = 0
                sum_success_rates2 = 0
                sum_vague_lvls3 = 0
                sum_success_rates3 = 0
                # sum_vague_lvls3n = 0
                # sum_success_rates3n = 0
                sum_vague_lvls4 = 0
                sum_success_rates4 = 0
                sum_vague_lvls5 = 0
                sum_success_rates5 = 0
                sum_vague_lvls6 = 0
                sum_success_rates6 = 0
                sum_vague_lvls7 = 0
                sum_success_rates7 = 0
                sum_vague_lvls8 = 0
                sum_success_rates8 = 0
                for ii in range(avg_i[fig]):
                  if ii == 0:
                    print(f"python main.py {n} {k} {n} {c} {d} {w}{' -n' if null else ''} {i} -r {i}")
                  try:
                    print(f"Setup 1-{ii+1}:")
                    vl1, sr1 = main((n, k, n, c, d, w, null, 100000, 100000, True, True, True, False)) #linear-exp-stimgen-pos

                    sum_vague_lvls1 += vl1
                    sum_success_rates1 += sr1
                    
                  except ValueError:
                    print("==> Failed Setup 1")
                    exit(0)

                  # try:
                  #   print(f"Setup 1n-{ii+1}:")
                  #   vl1n, sr1n = main((n, k, n, c, d, w, null, 100000, 100000, True, True, True, True)) #linear-exp-stimgen-neg

                  #   sum_vague_lvls1n += vl1n
                  #   sum_success_rates1n += sr1n
                    
                  # except ValueError:
                  #   print("==> Failed Setup 1n")
                  #   exit(0)

                  try:
                    print(f"Setup 2-{ii+1}:")
                    vl2, sr2 = main((n, k, n, c, d, w, null, 100000, 100000, True, True, False, False)) #linear-exp-no stimgen

                    sum_vague_lvls2 += vl2
                    sum_success_rates2 += sr2
                    
                  except ValueError:
                    print("==> Failed Setup 2")
                    exit(0)

                  try:
                    print(f"Setup 3-{ii+1}:")
                    vl3, sr3 = main((n, k, n, c, d, w, null, i, i, True, False, True, False)) #linear-linear-stimgen-pos

                    sum_vague_lvls3 += vl3
                    sum_success_rates3 += sr3
                    
                  except ValueError:
                    print("==> Failed Setup 3")
                    exit(0)

                  # try:
                  #   print(f"Setup 3n-{ii+1}:")
                  #   vl3n, sr3n = main((n, k, n, c, d, w, null, i, i, True, False, True, True)) #linear-linear-stimgen-neg

                  #   sum_vague_lvls3n += vl3n
                  #   sum_success_rates3n += sr3n
                    
                  # except ValueError:
                  #   print("==> Failed Setup 3")
                  #   exit(0)

                  try:
                    print(f"Setup 4-{ii+1}:")
                    vl4, sr4 = main((n, k, n, c, d, w, null, i, i, True, False, False, False)) #linear-linear-no stimgen

                    sum_vague_lvls4 += vl4
                    sum_success_rates4 += sr4
                    
                  except ValueError:
                    print("==> Failed Setup 4")
                    exit(0)

                  try:
                    print(f"Setup 5-{ii+1}:")
                    vl5, sr5 = main((n, k, n, c, d, w, null, 100000, 100000, False, True, True, False)) #gauss-exp-stimgen

                    sum_vague_lvls5 += vl5
                    sum_success_rates5 += sr5
                    
                  except ValueError:
                    print("==> Failed Setup 5")
                    exit(0)

                  try:
                    print(f"Setup 6-{ii+1}:")
                    vl6, sr6 = main((n, k, n, c, d, w, null, 100000, 100000, False, True, False, False)) #gauss-exp-no stimgen

                    sum_vague_lvls6 += vl6
                    sum_success_rates6 += sr6
                    
                  except ValueError:
                    print("==> Failed Setup 6")
                    exit(0)

                  try:
                    print(f"Setup 7-{ii+1}:")
                    vl7, sr7 = main((n, k, n, c, d, w, null, i, i, False, False, True, False)) #gauss-linear-stimgen

                    sum_vague_lvls7 += vl7
                    sum_success_rates7 += sr7
                    
                  except ValueError:
                    print("==> Failed Setup 7")
                    exit(0)

                  try:
                    print(f"Setup 8-{ii+1}:")
                    vl8, sr8 = main((n, k, n, c, d, w, null, i, i, False, False, False, False)) #gauss-linear-no stimgen

                    sum_vague_lvls8 += vl8
                    sum_success_rates8 += sr8

                  except ValueError:
                    print("==> Failed Setup 8")
                    exit(0)

                avg_vague_lvls[0].append(sum_vague_lvls1 / avg_i[fig])
                success_rates[0].append(sum_success_rates1 / avg_i[fig])
                print(f" => avg_vague_lvls1 = {sum_vague_lvls1 / avg_i[fig]}")
                print(f" => success_rates1 = {sum_success_rates1 / avg_i[fig]}")

                # avg_vague_lvls[8].append(sum_vague_lvls1n / avg_i[fig])
                # success_rates[8].append(sum_success_rates1n / avg_i[fig])
                # print(f" => avg_vague_lvls1n = {sum_vague_lvls1n / avg_i[fig]}")
                # print(f" => success_rates1n = {sum_success_rates1n / avg_i[fig]}")

                avg_vague_lvls[1].append(sum_vague_lvls2 / avg_i[fig])
                success_rates[1].append(sum_success_rates2 / avg_i[fig])
                print(f" => avg_vague_lvls2 = {sum_vague_lvls2 / avg_i[fig]}")
                print(f" => success_rates2 = {sum_success_rates2 / avg_i[fig]}")
                
                avg_vague_lvls[2].append(sum_vague_lvls3 / avg_i[fig])
                success_rates[2].append(sum_success_rates3 / avg_i[fig])
                print(f" => avg_vague_lvls3 = {sum_vague_lvls3 / avg_i[fig]}")
                print(f" => success_rates3 = {sum_success_rates3 / avg_i[fig]}")

                # avg_vague_lvls[9].append(sum_vague_lvls3n / avg_i[fig])
                # success_rates[9].append(sum_success_rates3n / avg_i[fig])
                # print(f" => avg_vague_lvls3n = {sum_vague_lvls3n / avg_i[fig]}")
                # print(f" => success_rates3n = {sum_success_rates3n / avg_i[fig]}")
                
                avg_vague_lvls[3].append(sum_vague_lvls4 / avg_i[fig])
                success_rates[3].append(sum_success_rates4 / avg_i[fig])
                print(f" => avg_vague_lvls4 = {sum_vague_lvls4 / avg_i[fig]}")
                print(f" => success_rates4 = {sum_success_rates4 / avg_i[fig]}")
                
                avg_vague_lvls[4].append(sum_vague_lvls5 / avg_i[fig])
                success_rates[4].append(sum_success_rates5 / avg_i[fig])
                print(f" => avg_vague_lvls5 = {sum_vague_lvls5 / avg_i[fig]}")
                print(f" => success_rates5 = {sum_success_rates5 / avg_i[fig]}")
                
                avg_vague_lvls[5].append(sum_vague_lvls6 / avg_i[fig])
                success_rates[5].append(sum_success_rates6 / avg_i[fig])
                print(f" => avg_vague_lvls6 = {sum_vague_lvls6 / avg_i[fig]}")
                print(f" => success_rates6 = {sum_success_rates6 / avg_i[fig]}")
                
                avg_vague_lvls[6].append(sum_vague_lvls7 / avg_i[fig])
                success_rates[6].append(sum_success_rates7 / avg_i[fig])
                print(f" => avg_vague_lvls7 = {sum_vague_lvls7 / avg_i[fig]}")
                print(f" => success_rates7 = {sum_success_rates7 / avg_i[fig]}")
                
                avg_vague_lvls[7].append(sum_vague_lvls8 / avg_i[fig])
                success_rates[7].append(sum_success_rates8 / avg_i[fig])
                print(f" => avg_vague_lvls8 = {sum_vague_lvls8 / avg_i[fig]}")
                print(f" => success_rates8 = {sum_success_rates8 / avg_i[fig]}")
                
    gen_comp_graph(fig+6, w_lst[fig], avg_vague_lvls, success_rates)

  # n_lst = [200]
  # k_lst = [40]
  # c = 1
  # d_lst = [4]
  # w_lst = [0, 0.25, 0.5, 1, 2, 4]
  # i_lst = [1000, 5000, 10000, 50000, 100000]
  # avg_i = 10
  # r = 1000000

  # avg_vague_lvls = []
  # success_rates = []

  # for n in n_lst:
  #   for k in k_lst:
  #     for ind, d in enumerate(d_lst):
  #       for i in i_lst:
  #         avg_vague_lvls.append([])
  #         success_rates.append([])
  #         for null in [False]:
  #           for w in w_lst:
  #             sum_vague_lvls = 0
  #             sum_success_rates = 0
  #             for ii in range(avg_i):
  #               if ii == 0:
  #                 print(f"python main.py {n} {k} {n} {c} {d} {w}{' -n' if null else ''} {i} -r {i}")
  #               try:
  #                 vl, sr = main((n, k, n, c, d, w, null, i, i))
  #                 # print(f"#{ii+1} --> {vl}--{sr}")
  #                 sum_vague_lvls += vl
  #                 sum_success_rates += sr
  #               except ValueError:
  #                 print("==> Failed")
  #                 continue
            
  #             avg_vague_lvls[-1].append(sum_vague_lvls / avg_i)
  #             success_rates[-1].append(sum_success_rates / avg_i)
  #             print(f" => avg_vague_lvls = {sum_vague_lvls / avg_i}")
  #             print(f" => success_rates = {sum_success_rates / avg_i}")
  
  # gen_extended_graph2(avg_vague_lvls, success_rates)

run()