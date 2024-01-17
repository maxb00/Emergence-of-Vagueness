import numpy as np

def norm(arr):
  return arr / np.sum(arr) * 100

  # exp = np.exp(arr)
  # exp_sum = np.sum(exp)
  # return exp / exp_sum * 100

def stimgen(width: int, n: int) -> float:
  """Stimulus generalization function
  
  Args:
    n (int): the distance from the peak

  Returns:
    float: the coefficient of the contiguous reward
  """
  if width == 0:
    return 0
  return 16**(-n**2/width**2)

class Sender:
  """The sender of a signaling game
  
  Generates a signal given a world state.

  Attributes:
    num_states, num_signals (int): the number of (world) states, signals
    null_signal (boolean): indicates the use of null signals
    signal_weights (np.ndarray): the signal weights
    signal_history (list): history of probability matrix for
      signals
  """
  def __init__(self, num_states: int, num_signals: int, stimgen_width: int, null_signal=False):
    """Initializes the instances to set up a sender
    
    Args:
      num_states (int): the number of (world) states
      num_signals (int): the number of signals
      null_signal (boolean): indicates the use of null signals
    """
    self.num_states = num_states
    self.num_signals = num_signals + (1 if null_signal else 0)

    self.stimgen_width = stimgen_width

    self.null_signal = null_signal

    self.signal_weights = np.ones((self.num_signals, num_states))

    self.signal_history = []

  def gen_signal(self, state: int, record=False) -> int:
    """Generates a signal based on the state (hard-coded for now)
    
    Args:
      state (int): the current state

    Returns:
      int: a signal. -1 indicates a null signal
    """
    if record:
      prob = self.signal_weights / np.sum(self.signal_weights, axis=0)
      signal = np.random.choice(self.num_signals, p=prob.T[state])
    else:
      prob = self.signal_weights[:, state] / np.sum(self.signal_weights[:, state])
      signal = np.random.choice(self.num_signals, p=prob)
    if self.null_signal and signal == self.num_signals-1:
      signal = -1
    self.curr_signal = signal

    if record:
      self.signal_history.append(prob)

    # exp = np.exp(self.signal_weights)
    # sum_exp = np.sum(exp, axis=0)
    # prob = exp / sum_exp
    # signal = np.random.choice(self.num_signals, p=prob.T[state])
    # if self.null_signal and signal == self.num_signals-1:
    #   signal = -1
    # self.curr_signal = signal

    # if record:
    #   self.signal_history.append(prob)

    return signal
  
  def update(self, curr_game: dict):
    """Updates the signal weights using stimulus generalization

    Args:
      curr_game (dict): information about the current game
    """
    state, signal = curr_game["state"], curr_game["signal"]
    reward = curr_game["reward"]
    # if reward < 0:
    #   self.signal_weights[:, state] += -reward

    self.signal_weights[signal, state] += reward
    if self.signal_weights[signal, state] < 1:
      self.signal_weights[signal, state] = 1

    l = r = state
    for i in range(1, self.num_states//2+1):
      stimgen_reward = stimgen(self.stimgen_width, i) * reward

      if stimgen_reward == 0:
        break

      r += 1
      if r < self.num_states:
        self.signal_weights[signal, r] += stimgen_reward

        if self.signal_weights[signal, r] < 1:
          self.signal_weights[signal, r] = 1

      l -= 1
      if l >= 0:
        self.signal_weights[signal, l] += stimgen_reward
      
        if self.signal_weights[signal, l] < 1:
          self.signal_weights[signal, l] = 1

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
  
class Receiver:
  """The receiver of a signaling game

  Generates an action given a signal using an exponential function to generate the probabilities
  Updates the probabilities using stimulus generalization

  Attributes:
    num_signals, num_actions (int): the number of signals, actions
    action_weights (np.ndarray): the action weights
    action_history (list): history of probability matrix for
      actions
  """
  def __init__(self, num_signals: int, num_actions: int, stimgen_width: int):
    """Initializes the instances to set up a receiver

    Args:
      num_signals (int): the number of signals
      num_action (int): the number of actions
    """
    self.num_signals = num_signals
    self.num_actions = num_actions

    self.stimgen_width = stimgen_width

    self.action_weights = np.ones((num_signals, num_actions))

    self.action_history = []

  def gen_action(self, signal: int, record=False) -> int:
    """Generates an action based on a signal

    Args:
      signal (int): the signal, -1 if null signal
      record (boolean): indicates whether the probability will be added into the history

    Returns:
      int: an action
    """
    if record:
      prob = self.action_weights.T / np.sum(self.action_weights, axis=1)
    else:
      prob = self.action_weights[signal] / np.sum(self.action_weights[signal])
    if signal == -1:
      action = -1
    else:
      if record:
        action = np.random.choice(self.num_actions, p=prob.T[signal])
      else:
        action = np.random.choice(self.num_actions, p=prob)
    self.curr_action = action

    if record:
      self.action_history.append(prob.T)

    # exp = np.exp(self.action_weights)
    # sum_exp = np.sum(exp, axis=1)
    # prob = exp.T / sum_exp
    # if signal == -1:
    #   action = -1
    # else:
    #   action = np.random.choice(self.num_actions, p=prob.T[signal])
    # self.curr_action = action

    # if record:
    #   self.action_history.append(prob.T)

    return action
  
  def update(self, curr_game: dict):
    """Updates the action weights using stimulus generalization

    Args:
      curr_game (dict): information about the current game
    """
    signal, action = curr_game["signal"], curr_game["action"]
    reward = curr_game["reward"]
    # if reward < 0:
    #   self.action_weights[signal, :] += -reward
    self.action_weights[signal, action] += reward
    if self.action_weights[signal, action] < 1:
      self.action_weights[signal, action] = 1

    l = r = action
    for i in range(1, self.num_actions//2+1):
      stimgen_reward = stimgen(self.stimgen_width, i) * reward

      if stimgen_reward == 0:
        break

      r += 1
      if r < self.num_actions:
        self.action_weights[signal, r] += stimgen_reward

        if self.action_weights[signal, r] < 1:
          self.action_weights[signal, r] = 1

      l -= 1
      if l >= 0:
        self.action_weights[signal, l] += stimgen_reward

        if self.action_weights[signal, l] < 1:
          self.action_weights[signal, l] = 1

  def print_action_prob(self):
    """Prints the current action probabilities"""
    prob = np.zeros_like(self.action_weights)
    for i in range(self.num_signals):
      prob[i, :] = norm(self.action_weights[i, :])

    print('m|a', end=' ')
    for i in range(self.num_actions):
      print(f'{i:3}', end=' ')
    print()
    for i in range(self.num_signals):
      print(f'{i:3}', end=' ')
      for j in range(self.num_actions):
        print(f'{int(prob[i, j]):3}', end=' ')
      print()


