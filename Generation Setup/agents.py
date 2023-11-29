import numpy as np
from display import gen_image

class Agent:
  def __init__(self, num_states, num_signals, num_actions, null_signal=False):
    self.num_states = num_states
    self.num_signals = num_signals
    self.num_actions = num_actions
    self.null_signal = null_signal

    self.curr_state = np.zeros((self.num_signals, self.num_states))

    self.state_history = []

  def set_opt_strat(self):
    self.curr_state = np.zeros((self.num_signals, self.num_states))

    signal_perm = np.random.permutation(self.num_signals)
    size = self.num_states // self.num_signals
    remainder = self.num_states % self.num_signals
    index = 0
    for signal in signal_perm:
      if signal < remainder:
        bucket_size = size + 1
      else:
        bucket_size = size
      
      for i in range(bucket_size):
        self.curr_state[signal, index + i] = 100
      
      index += bucket_size

  def gen_state_signal_pairs(self, num_pair):
    avail_states = [i for i in range(self.num_states)]
    pairs = []
    for _ in range(num_pair):
      state = np.random.choice(avail_states)
      while max(self.curr_state[:, state]) == 0:
        avail_states.remove(state)
        state = np.random.choice(avail_states)
      
      signal = np.argmax(self.curr_state[:, state])

      pairs.append((state, signal))
      avail_states.remove(state)

    return pairs
  
  def learn(self, pairs):
    buckets = [[self.num_states, -1] for _ in range(self.num_signals)]
    for (state, signal) in pairs:
      if signal == -1:
        continue
      if state < buckets[signal][0]:
        buckets[signal][0] = state
      if state > buckets[signal][1]:
        buckets[signal][1] = state

    missing = False
    min_bucket = -1
    min_state = self.num_states
    max_bucket = -1
    max_state = -1
    for i, (bmin, bmax) in enumerate(buckets):
      print(f"({i}, {bmin}-{bmax})")
      if bmin > bmax:
        missing = True
      for j in range(bmin, bmax+1):
        self.curr_state[i, j] = 1

      if bmin < min_state:
        min_state = bmin
        min_bucket = i
      if bmax > max_state:
        max_state = bmax
        max_bucket = i

    if not missing:
      for i in range(min_state):
        self.curr_state[min_bucket, i] = 1
      for i in range(max_state+1, self.num_states):
        self.curr_state[max_bucket, i] = 1

n = 50
k = 2
c = 1
d = 0.5
l = 10
iter = 1000

agent1 = Agent(n, k, n)
agent1.set_opt_strat()
pairs = agent1.gen_state_signal_pairs(l)
print(pairs)

agent2 = Agent(n, k, n)
agent2.learn(pairs)
print(agent2.curr_state)
pairs = agent2.gen_state_signal_pairs(l)
print(pairs)

agent3 = Agent(n, k, n)
agent3.learn(pairs)
print(agent3.curr_state)

gen_image([agent1.curr_state, agent2.curr_state, agent3.curr_state])

