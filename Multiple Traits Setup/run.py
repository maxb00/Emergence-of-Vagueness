from main import main
import numpy as np
import math

class Test:
  def __init__(self):
    self.num_states = 3
    self.num_traits = 2
    self.num_signals = 2
    self.state_prob = gen_state_prob(3)
    self.null_signal = False

def normpdf(x, mean, std):
  var = float(std)**2
  denom = (2*math.pi*var)**.5
  num = math.exp(-(float(x)-float(mean))**2/(2*var))
  return num/denom

def gen_state_prob(num_states):
  mean = (num_states-1) / 2
  std = mean / 1.5

  state_prob = np.array([[0] * num_states] * num_states, dtype=np.float64)
  for i in range(num_states):
    for j in range(num_states):
      state_prob[i, j] = float(normpdf(i, mean, std) * normpdf(j, mean, std))

  state_prob = state_prob / np.sum(state_prob)

  return state_prob.flatten()

def info_measure_by_trait(self, signal_prob) -> float:
  total_states = self.num_states**self.num_traits
  signal_prob = signal_prob.reshape(self.num_signals, total_states)

  prob_t1 = np.zeros((self.num_signals, self.num_states))
  prob_t2 = np.zeros((self.num_signals, self.num_states))
  state_prob_t = np.zeros(self.num_states)
  for i in range(self.num_signals):
    for j in range(total_states):
      prob_t1[i, j%self.num_states] += signal_prob[i, j] * self.state_prob[j]
      prob_t2[i, j//self.num_states] += signal_prob[i, j] * self.state_prob[j]
      state_prob_t[j%self.num_states] += self.state_prob[j]
  prob_t1 = (prob_t1.T / np.sum(prob_t1, axis=1)).T
  prob_t2 = (prob_t2.T / np.sum(prob_t2, axis=1)).T
  state_prob_t = state_prob_t / np.sum(state_prob_t)

  print(prob_t1)
  print(prob_t2)
  print(state_prob_t)
  exit(0)

  inf_by_trait = [[], []]
  for i in range(self.num_signals):
    if self.null_signal and i == self.num_signals:
      break
    inf_sig_t1 = 0
    inf_sig_t2 = 0
    for j in range(self.num_states):
      inf_sig_t1 += prob_t1[i, j] * np.log(prob_t1[i, j]/state_prob_t[j])
      inf_sig_t2 += prob_t2[i, j] * np.log(prob_t2[i, j]/state_prob_t[j])

      print(f"inf_sig_t1 += {prob_t1[i, j]/state_prob_t[j]}")
      print(f"inf_sig_t2 += {prob_t2[i, j]/state_prob_t[j]}")

    inf_sig_t1 = (np.sum(signal_prob[i]) / (total_states)) * inf_sig_t1
    inf_sig_t2 = (np.sum(signal_prob[i]) / (total_states)) * inf_sig_t2

    inf_by_trait[0].append(inf_sig_t1)
    inf_by_trait[1].append(inf_sig_t2)

  return inf_by_trait

def run():
  _self = Test()

  signal_prob = np.array([
    [[1, 1, 1],
    [0, 0, 0],
    [0, 0, 0]],
    [[0, 0, 0],
     [1, 1, 1],
     [1, 1, 1]]
  ])

  res = info_measure_by_trait(_self, signal_prob)

  print(res)

  # n_lst = [3, 4, 5, 6, 7, 8]
  # k_lst = [4, 5]
  # c = 1
  # d_lst = [0.3, 0.5, 0.7]
  # i = 10000
  # r = 125

  # for n in n_lst:
  #   for k in k_lst:
  #     for ind, d in enumerate(d_lst):
  #       for null in [False]:
  #         print(f"python main.py {2} {n} {k} {n} {c} {d}{' -n' if null else ''} {i} -r {r}")
  #         try:
  #           main((2, n, k, n, c, d, null, i, r))
  #         except ValueError:
  #           print("==> Failed")
  #           continue

run()