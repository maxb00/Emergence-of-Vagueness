import numpy as np

def norm(arr):
  """Normalizes the weights into probabilities"""
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

class Sender:
  """The sender of a signaling game
  
  Generates a signal given a world state.

  Attributes:
    num_traits, num_states, num_signals (int): the number of traits, (world) states per trait, signals
    total_states: the number of total states over all traits
    null_signal (boolean): indicates the use of null signals
    signal_weights (np.ndarray): the signal weights
    signal_history (list): history of probability matrix for
      signals
  """
  def __init__(self, num_traits: int, num_states: int, num_signals: int, null_signal=False):
    """Initializes the instances to set up a sender
    
    Args:
      num_traits (int): the number of traits
      num_states (int): the number of (world) states per trait
      num_signals (int): the number of signals
      null_signal (boolean): indicates the use of null signals
    """
    self.num_traits = num_traits
    self.num_states = num_states
    self.num_signals = num_signals + (1 if null_signal else 0)

    self.total_states = num_states**num_traits

    self.null_signal = null_signal

    self.signal_weights = np.zeros((self.num_signals, self.total_states))

    self.signal_history = []

  def gen_signal(self, state: int, record=False) -> int:
    """Generates a signal based on the state 
    
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
      new_size = [self.num_signals]
      new_size.extend([self.num_states] * self.num_traits)
      self.signal_history.append(np.resize(prob, tuple(new_size)))

    return signal
  
  def update(self, curr_game: dict):
    """Updates the signal weights using stimulus generalization

    Args:
      curr_game (dict): information about the current game
    """
    state, signal = curr_game["fstate"], curr_game["signal"]
    reward = curr_game["reward"]
    self.signal_weights[signal, state] += reward

    maxw = np.max(self.signal_weights[:, state])
    minw = np.min(self.signal_weights[:, state])

    # Setting the range of weights from -300 to 300
    if maxw - minw > 600:
      self.signal_weights[:, state] = (self.signal_weights[:, state] - minw) * 600 / (maxw - minw) + minw
    elif maxw > 300:
      self.signal_weights[:, state] -= maxw - 300
    elif minw < -300:
      self.signal_weights[:, state] += -300 - minw

    # l = r = state
    # for i in range(1,4):
    #   stimgen_reward = stimgen(i) * reward

    #   r += 1
    #   if r < self.num_states:
    #     self.signal_weights[signal, r] += stimgen_reward

    #   l -= 1
    #   if l >= 0:
    #     self.signal_weights[signal, l] += stimgen_reward

  def print_signal_prob(self):
    """Prints the current signal probabilities"""
    prob = np.zeros_like(self.signal_weights)
    for i in range(self.total_states):
      prob[:, i] = norm(self.signal_weights[:, i])

    print('m|s', end=' ')
    for i in range(self.total_states):
      print(f'{i:3}', end=' ')
    print()
    for i in range(self.num_signals):
      print(f'{i:3}', end=' ')
      for j in range(self.total_states):
        print(f'{int(prob[i, j]):3}', end=' ')
      print()
  
class Receiver:
  """The receiver of a signaling game

  Generates an action given a signal using an exponential function to generate the probabilities
  Updates the probabilities using stimulus generalization

  Attributes:
    num_traits, num_signals, num_actions (int): the number of traits, signals, actions per trait
    total_actions (int): the number of total actions over all traits
    action_weights (np.ndarray): the action weights
    action_history (list): history of probability matrix for
      actions
  """
  def __init__(self, num_traits: int, num_signals: int, num_actions: int):
    """Initializes the instances to set up a receiver

    Args:
      num_traits (int): the number of traits
      num_signals (int): the number of signals
      num_action (int): the number of actions per trait
    """
    self.num_traits = num_traits
    self.num_signals = num_signals
    self.num_actions = num_actions

    self.total_actions = num_actions**num_traits

    self.action_weights = np.zeros((num_signals, self.total_actions))

    self.action_history = []

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
      action = np.random.choice(self.total_actions, p=prob.T[signal])
    self.curr_action = action

    if record:
      new_size = [self.num_signals]
      new_size.extend([self.num_actions] * self.num_traits)
      self.action_history.append(np.resize(prob.T, tuple(new_size)))

    return action
  
  def update(self, curr_game: dict):
    """Updates the action weights using stimulus generalization

    Args:
      curr_game (dict): information about the current game
    """
    signal, action = curr_game["signal"], curr_game["faction"]
    reward = curr_game["reward"]
    self.action_weights[signal, action] += reward

    maxw = np.max(self.action_weights[signal])
    minw = np.min(self.action_weights[signal])

    # Setting the range of weights from -300 to 300
    if maxw - minw > 600:
      self.action_weights[signal] = (self.action_weights[signal] - minw) * 600 / (maxw - minw) + minw
    elif maxw > 300:
      self.action_weights[signal] -= maxw - 300
    elif minw < -300:
      self.action_weights[signal] += -300 - minw

    # l = r = action
    # for i in range(1,4):
    #   stimgen_reward = stimgen(i) * reward

    #   r += 1
    #   if r < self.num_actions:
    #     self.action_weights[signal, r] += stimgen_reward

    #   l -= 1
    #   if l >= 0:
    #     self.action_weights[signal, l] += stimgen_reward

  def print_action_prob(self):
    """Prints the current action probabilities"""
    prob = np.zeros_like(self.action_weights)
    for i in range(self.num_signals):
      prob[i, :] = norm(self.action_weights[i, :])

    print('m|a', end=' ')
    for i in range(self.total_actions):
      print(f'{i:3}', end=' ')
    print()
    for i in range(self.num_signals):
      print(f'{i:3}', end=' ')
      for j in range(self.total_actions):
        print(f'{int(prob[i, j]):3}', end=' ')
      print()


