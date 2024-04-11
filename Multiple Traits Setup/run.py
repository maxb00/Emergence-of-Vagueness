from main import main
import numpy as np

def info_measure(self, signal_prob) -> float:
  shape = signal_prob.shape
  signal_prob = signal_prob.reshape(shape[0], shape[1] * shape[2])

  print(signal_prob)
  exit(0)

  prob = (signal_prob.T / np.sum(signal_prob, axis=1)).T

  inf = 0
  for i in range(self.num_signals):
    if self.null_signal and i == self.num_signals:
      break
    inf_sig = 0
    for j in range(self.num_states):
      inf_sig += prob[i, j] * np.log(prob[i, j] * self.num_states)

    inf += (np.sum(signal_prob[i]) / self.num_states) * inf_sig

  return inf

def run():
  # a = np.array([
  #   [
  #     [1, 2, 3],
  #     [3, 4, 5],
  #     [1, 2, 4]
  #   ],
  #   [
  #     [5, 6, 7],
  #     [7, 8, 9],
  #     [9, 10, 11]
  #   ]
  # ])

  # print(info_measure(0, a))
  n_lst = [3, 4, 5, 6, 7, 8]
  k_lst = [4, 5]
  c = 1
  d_lst = [0.3, 0.5, 0.7]
  i = 10000
  r = 125

  for n in n_lst:
    for k in k_lst:
      for ind, d in enumerate(d_lst):
        for null in [False]:
          print(f"python main.py {2} {n} {k} {n} {c} {d}{' -n' if null else ''} {i} -r {r}")
          try:
            main((2, n, k, n, c, d, null, i, r))
          except ValueError:
            print("==> Failed")
            continue

run()