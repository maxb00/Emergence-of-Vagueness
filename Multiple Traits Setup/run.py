from main import main
import numpy as np
import math

def run():

  # res = [[0 for _ in range(27)] for _ in range(27)]
  # for x in range(27):
  #   for y in range(27):
  #     diff1 = abs(x // 9 - y // 9)
  #     diff2 = abs((x % 9) // 3 - (y % 9) // 3)
  #     diff3 = abs(x % 3 - y % 3)

  #     total_diff = diff1 + diff2 + diff3

  #     res[x][y] = 1 - 0.5 * total_diff
  
  # for i in range(27):
  #   for j in range(27):
  #     print(f"{res[i][j]:>6}", end='')
  #   print()

  res = [[0 for _ in range(9)] for _ in range(9)]
  for x in range(9):
    for y in range(9):
      diff1 = abs(x // 3 - y // 3)
      diff2 = abs(x % 3 - y % 3)

      total_diff = diff1 + diff2

      res[x][y] = 1 - 0.5 * total_diff
  
  for i in range(9):
    print(f"{i+1:>6}", end='')
    for j in range(9):
      print(f"{res[i][j]:>6}", end='')
    print()
  
  exit(0)



  t = 2
  n_lst = [5]
  k_lst = [4]
  c = 1
  d_lst = [0.4]
  i = 8000
  r = 200
  repeat = 5

  # info_by_state_t1_sum = [0] * 3
  # info_by_state_t2_sum = [0] * 3
  # info_by_state_t3_sum = [0] * 3
  # w_info_by_state_t1_sum = [0] * 3
  # w_info_by_state_t2_sum = [0] * 3
  # w_info_by_state_t3_sum = [0] * 3


  for n in n_lst:
    for k in k_lst:
      for ind, d in enumerate(d_lst):
        for null in [False]:
          for repeat_i in range(repeat):
            print(f"python main.py {t} {n} {k} {n} {c} {d}{' -n' if null else ''} {i} -r {r} #{repeat_i}")
            try:
              main((t, n, k, n, c, d, null, i, r))

              # info_by_state_t1, info_by_state_t2, info_by_state_t3, w_info_by_state_t1, w_info_by_state_t2, w_info_by_state_t3 = main((3, n, k, n, c, d, null, i, r))
              # info_by_state_t1, info_by_state_t2, w_info_by_state_t1, w_info_by_state_t2, = main((2, n, k, n, c, d, null, i, r))

              # info_by_state_t1_sum = [info_by_state_t1_sum[i] + info_by_state_t1[i] for i in range(3)]
              # info_by_state_t2_sum = [info_by_state_t2_sum[i] + info_by_state_t2[i] for i in range(3)]
              # info_by_state_t3_sum = [info_by_state_t3_sum[i] + info_by_state_t3[i] for i in range(3)]
              # w_info_by_state_t1_sum = [w_info_by_state_t1_sum[i] + w_info_by_state_t1[i] for i in range(3)]
              # w_info_by_state_t2_sum = [w_info_by_state_t2_sum[i] + w_info_by_state_t2[i] for i in range(3)]
              # w_info_by_state_t3_sum = [w_info_by_state_t3_sum[i] + w_info_by_state_t3[i] for i in range(3)]
            except ValueError:
              print("==> Failed")
              continue
  
  # print("\n\nAVERAGE")
  # print(f"Unweighted trait 1's low|medium|high: {info_by_state_t1_sum[0]/repeat:.5f} | {info_by_state_t1_sum[1]/repeat:.5f} | {info_by_state_t1_sum[2]/repeat:.5f}")
  # print(f"Unweighted trait 2's low|medium|high: {info_by_state_t2_sum[0]/repeat:.5f} | {info_by_state_t2_sum[1]/repeat:.5f} | {info_by_state_t2_sum[2]/repeat:.5f}")
  # print(f"Unweighted trait 3's low|medium|high: {info_by_state_t3_sum[0]/repeat:.5f} | {info_by_state_t3_sum[1]/repeat:.5f} | {info_by_state_t3_sum[2]/repeat:.5f}")
  # print(f"Weighted trait 1's low|medium|high: {w_info_by_state_t1_sum[0]/repeat:.5f} | {w_info_by_state_t1_sum[1]/repeat:.5f} | {w_info_by_state_t1_sum[2]/repeat:.5f}")
  # print(f"Weighted trait 2's low|medium|high: {w_info_by_state_t2_sum[0]/repeat:.5f} | {w_info_by_state_t2_sum[1]/repeat:.5f} | {w_info_by_state_t2_sum[2]/repeat:.5f}")
  # print(f"Weighted trait 3's low|medium|high: {w_info_by_state_t3_sum[0]/repeat:.5f} | {w_info_by_state_t3_sum[1]/repeat:.5f} | {w_info_by_state_t3_sum[2]/repeat:.5f}")

run()