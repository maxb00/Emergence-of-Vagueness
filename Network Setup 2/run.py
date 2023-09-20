from main import main

def analyze():
  const = open("results/circular/const_reward_circ.txt", "r")
  small = open("results/circular/small_reward_circ.txt", "r")
  same = open("results/circular/same_reward_circ.txt", "r")

  for (const_ln, small_ln, same_ln) in zip(const, small, same):
    if const_ln[0] == 'p':
      print(const_ln, end="")
    elif const_ln[0] == 'a':
      const_wrds = const_ln.split()
      small_wrds = small_ln.split()
      same_wrds = same_ln.split()

      const_avg1 = float(const_wrds[1].split("=")[1][:-1])
      const_avg2 = float(const_wrds[2].split("=")[1])
      small_avg1 = float(small_wrds[1].split("=")[1][:-1])
      small_avg2 = float(small_wrds[2].split("=")[1])
      same_avg1 = float(same_wrds[1].split("=")[1][:-1])
      same_avg2 = float(same_wrds[2].split("=")[1])

      if const_avg1+const_avg2 >= small_avg1+small_avg2:
        if const_avg1+const_avg2 >= same_avg1+same_avg2:
          print(f"const: avg1={const_avg1}, avg2={const_avg2}")
          if small_avg1+small_avg2 >= same_avg1+same_avg2:
            print(f"small: avg1={small_avg1}, avg2={small_avg2}")
            print(f"same: avg1={same_avg1}, avg2={same_avg2}")
            print()
          else:
            print(f"same: avg1={same_avg1}, avg2={same_avg2}")
            print(f"small: avg1={small_avg1}, avg2={small_avg2}")
            print()
        else:
          print(f"same: avg1={same_avg1}, avg2={same_avg2}")
          print(f"const: avg1={const_avg1}, avg2={const_avg2}")
          print(f"small: avg1={small_avg1}, avg2={small_avg2}")
          print()
      else:
        if small_avg1+small_avg2 >= same_avg1+same_avg2:
          print(f"small: avg1={small_avg1}, avg2={small_avg2}")
          if const_avg1+const_avg2 >= same_avg1+same_avg2:
            print(f"const: avg1={const_avg1}, avg2={const_avg2}")
            print(f"same: avg1={same_avg1}, avg2={same_avg2}")
            print()
          else:
            print(f"same: avg1={same_avg1}, avg2={same_avg2}")
            print(f"const: avg1={const_avg1}, avg2={const_avg2}")
            print()
        else:
          print(f"same: avg1={same_avg1}, avg2={same_avg2}")
          print(f"small: avg1={small_avg1}, avg2={small_avg2}")
          print(f"const: avg1={const_avg1}, avg2={const_avg2}")
          print()


def run():
  n_lst = [20, 30]
  k_lst = [2, 3, 4]
  c_lst = [1.5, 1.75, 2]
  d = 0.5
  i = 4500
  r = 150

  avg = 50

  for n in n_lst:
    for k in k_lst:
      for c in c_lst:
        for null in [False, True]:
          # sum1 = 0
          # sum2 = 0
          print(f"python main.py {n} {k} {n} {c} {d}{' -n' if null else ''} {i} -r {r}")
          try:
            main((n, k, n, c, d, null, i, r))
          except RuntimeError:
            continue
          # for iter in range(avg):
          #   ep1, ep2 = main((n, k, n, c, d, null, i, r))
          #   print(f"run {iter}: ep1={ep1:.3f}, ep2={ep2:.3f}")
          #   sum1 += ep1
          #   sum2 += ep2
          
          # avgp1 = sum1 / avg
          # avgp2 = sum2 / avg
          # print(f"average: avg1={avgp1:.3f}, avg2={avgp2:.3f}")

run()
# analyze()
