import numpy as np

def norm(arr):
  exp = np.exp(arr)
  exp_sum = np.sum(exp)
  return exp / exp_sum * 100

def stimgen(n: int) -> float:
  """Stimulus generalization function
  
  Args:
    n (int): the distance from the peak

  Returns:
    float: the coefficient of the contiguous reward
  """
  return 1 / 2**(n**2)

class Agent:
  """An agent of a signaling game
  
  Can become the Sender or Receiver, based on the specific game

  Attributes:
    num_states, num_signals, num_actions (int): the number of (world) states,
      signals, and actions
    null_signal (boolean): indicates the use of null signals
    signal_weights, action_weights (np.ndarray): the signal/action weights
    signal_history, action_history (list): history of probability matrix for
      signals/actions
  """
  def __init__(self, num_states: int, num_signals: int, 
               num_actions: int, null_signal=False):
    """Initializes the instances to set up an agent
    
    Args:
      num_states (int): the number of (world) states
      num_signals (int): the number of signals
      num_action (int): the number of actions
      null_signal (boolean): indicates the use of null signals
    """
    self.num_states = num_states
    self.num_signals = num_signals + (1 if null_signal else 0)
    self.num_actions = num_actions

    self.null_signal = null_signal

    self.signal_weights = np.zeros((self.num_signals, num_states))
    self.action_weights = np.zeros(
      (self.num_signals - (1 if null_signal else 0), 
      num_actions))

    self.signal_history = []
    self.action_history = []

    self.positive_reward = 1 # to update

  def gen_signal(self, state: int, record=False) -> int:
    """Generates a signal based on the state (hard-coded for now)
    
    Args:
      state (int): the current state

    Returns:
      int: a signal. -1 indicates a null signal
    """
    exp = np.exp(self.signal_weights)
    sum_exp = np.sum(exp, axis=0)
    prob = exp / sum_exp
    signal = np.random.choice(self.num_signals, p=prob.T[state])
    if self.null_signal and signal == self.num_signals-1:
      signal = -1
    self.curr_signal = signal

    if record:
      self.signal_history.append(prob)

    return signal
  
  def gen_action(self, signal: int, record=False) -> int:
    """Generates an action based on a signal

    Args:
      signal (int): the signal, -1 if null signal
      record (boolean): indicates whether the probability will be added into the history

    Returns:
      int: an action
    """
    exp = np.exp(self.action_weights)
    sum_exp = np.sum(exp, axis=1)
    prob = exp.T / sum_exp
    if signal == -1:
      action = -1
    else:
      action = np.random.choice(self.num_actions, p=prob.T[signal])
    self.curr_action = action

    if record:
      self.action_history.append(prob.T)

    return action
  
  def sender_update(self, curr_game: dict):
    """Updates the signal/action weights using stimulus generalization as a sender

    Args:
      curr_game (dict): information about the current game
    """
    state, signal = curr_game["state"], curr_game["signal"]
    if signal == -1:
      return
    reward = curr_game["reward"]
    self.signal_weights[signal, state] += reward

    l = r = state
    for i in range(1,4):
      stimgen_reward = stimgen(i) * reward

      r += 1
      if r < self.num_states:
        self.signal_weights[signal, r] += stimgen_reward

      l -= 1
      if l >= 0:
        self.signal_weights[signal, l] += stimgen_reward

    reward = self.positive_reward
    self.action_weights[signal, state] += reward
    l = r = state
    for i in range(1,4):
      stimgen_reward = stimgen(i) * reward

      r += 1
      if r < self.num_states:
        self.action_weights[signal, r] += stimgen_reward

      l -= 1
      if l >= 0:
        self.action_weights[signal, l] += stimgen_reward

  def receiver_update(self, curr_game: dict):
    """Updates the signal/action weights using stimulus generalization as a receiver

    Args:
      curr_game (dict): information about the current game
    """
    signal, action = curr_game["signal"], curr_game["action"]
    if signal == -1:
      return
    reward = curr_game["reward"]
    self.action_weights[signal, action] += reward

    l = r = action
    for i in range(1,4):
      stimgen_reward = stimgen(i) * reward

      r += 1
      if r < self.num_actions:
        self.action_weights[signal, r] += stimgen_reward

      l -= 1
      if l >= 0:
        self.action_weights[signal, l] += stimgen_reward

    reward = self.positive_reward
    self.signal_weights[signal, action] += reward
    l = r = action
    for i in range(1,4):
      stimgen_reward = stimgen(i) * reward

      r += 1
      if r < self.num_states:
        self.signal_weights[signal, r] += stimgen_reward

      l -= 1
      if l >= 0:
        self.signal_weights[signal, l] += stimgen_reward

  def print_signal_prob(self):
    """Prints the current signal probabilities"""
    prob = np.zeros_like(self.signal_weights)
    for i in range(self.num_states):
      prob[:, i] = norm(self.signal_weights[:, i])

    print('m|s', end=' ')
    for i in range(self.num_states):
      print(f'{i:3}', end=' ')
    print()
    for i in range(self.num_signals):
      print(f'{i:3}', end=' ')
      for j in range(self.num_states):
        print(f'{int(prob[i, j]):3}', end=' ')
      print()

  def print_action_prob(self):
    """Prints the current action probabilities"""
    prob = np.zeros_like(self.action_weights)
    for i in range(self.num_signals - (1 if self.null_signal else 0)):
      prob[i, :] = norm(self.action_weights[i, :])

    print('m|a', end=' ')
    for i in range(self.num_actions):
      print(f'{i:3}', end=' ')
    print()
    for i in range(self.num_signals - (1 if self.null_signal else 0)):
      print(f'{i:3}', end=' ')
      for j in range(self.num_actions):
        print(f'{int(prob[i, j]):3}', end=' ')
      print()
