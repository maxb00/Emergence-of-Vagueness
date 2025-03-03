from copy import deepcopy
import pdb
import numpy as np
from warnings import catch_warnings

def RaiseWarning(func):
    def wrapper(*args, **kwargs):
        with catch_warnings(action="error"):
            return func(*args, **kwargs)
    return wrapper

def transform(value):
  if value == 0:
    return 1.0
  elif value < 1:
    return 1.0/((value-1.0)**2)
  return (value+1.0)**2

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
  def __init__(self, num_states: int, num_signals: int, null_signal=False):
    """Initializes the instances to set up a sender
    
    Args:
      num_states (int): the number of (world) states
      num_signals (int): the number of signals
      null_signal (boolean): indicates the use of null signals
    """
    self.num_states = num_states
    self.num_signals = num_signals + (1 if null_signal else 0)

    self.null_signal = null_signal

    self.signal_weights = np.zeros((self.num_signals, num_states))

    self.signal_history = []

  @RaiseWarning
  def gen_signal(self, state: int, record=False) -> int:
    """Generates a signal based on the state (hard-coded for now)
    
    Args:
      state (int): the current state

    Returns:
      int: a signal. -1 indicates a null signal
    """
    try:
      transformation_vector = np.vectorize(transform, otypes=[float])
      transformed_weights = transformation_vector(self.signal_weights)
      col_sums = np.sum(transformed_weights, axis=0)
      prob = transformed_weights / col_sums
    except RuntimeWarning:
      pdb.set_trace()
    try:
      signal = np.random.choice(self.num_signals, p=prob.T[state])
    except ValueError:
      pdb.set_trace()
    if self.null_signal and signal == self.num_signals-1:
      signal = -1
    self.curr_signal = signal

    if record:
      self.signal_history.append(prob)

    return signal
  
  def update(self, curr_game: dict):
    """Updates the signal weights using stimulus generalization

    Args:
      curr_game (dict): information about the current game
    """
    state, signal = curr_game["state"], curr_game["signal"]
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
  def __init__(self, num_signals: int, num_actions: int):
    """Initializes the instances to set up a receiver

    Args:
      num_signals (int): the number of signals
      num_action (int): the number of actions
    """
    self.num_signals = num_signals
    self.num_actions = num_actions

    self.action_weights = np.zeros((num_signals, num_actions))

    self.action_history = []

  @RaiseWarning
  def gen_action(self, signal: int, record=False) -> int:
    """Generates an action based on a signal

    Args:
      signal (int): the signal, -1 if null signal
      record (boolean): indicates whether the probability will be added into the history

    Returns:
      int: an action
    """
    try:
      transformation_vector = np.vectorize(transform, otypes=[float])
      transformed_weights = transformation_vector(self.action_weights)
      row_sums = np.sum(transformed_weights, axis=1)
      prob = transformed_weights.T / row_sums
    except RuntimeWarning:
      pdb.set_trace()
    if signal == -1:
      action = -1
    else:
      action = np.random.choice(self.num_actions, p=prob.T[signal])
    self.curr_action = action

    if record:
      self.action_history.append(prob.T)

    return action
  
  def update(self, curr_game: dict):
    """Updates the action weights using stimulus generalization

    Args:
      curr_game (dict): information about the current game
    """
    signal, action = curr_game["signal"], curr_game["action"]
    reward = curr_game["reward"]
    self.action_weights[signal, action] += reward

    # 3/3/25 - removed reci stimgen
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
    for i in range(self.num_actions):
      print(f'{i:3}', end=' ')
    print()
    for i in range(self.num_signals):
      print(f'{i:3}', end=' ')
      for j in range(self.num_actions):
        print(f'{int(prob[i, j]):3}', end=' ')
      print()


class SKSender(Sender):
  def __init__(self, *args, **kwargs):
    Sender.__init__(self, *args, **kwargs)
    self.classifier = None
    
    if self.null_signal:
      assert self.num_signals == 3, f"n_signals should be 3, found {self.num_signals} and {self.null_signal}"
      # otherwise, we have to do more work 
      # to determine and send the correct training label in self.update()
    else:
      assert self.num_signals == 2, f"n_signals should be 2, found {self.num_signals} and {self.null_signal}"

    self.all_signals = np.array(range(self.num_signals))

  def set_classifier(self, clf):
    self.classifier = clf
    # to "hack" fit requirements, partial fit now on a random sample
    state = np.random.randint(0, self.num_states)
    signal = np.random.randint(0, self.num_signals-(1 if self.null_signal else 0))
    training_sample = np.array([[state]])
    training_label = np.array([signal])
    self.classifier.partial_fit(training_sample, training_label, classes=self.all_signals)

  def gen_signal_direct(self, state, record=False) -> int:
    # probe classifier for prediction directly.
    assert self.classifier is not None
    self.update_probs()
    if record:
      self.signal_history.append(self.signal_weights)
    pred = self.classifier.predict(np.array([[state]]))
    prediction = pred[0]
    if self.null_signal and prediction == self.num_signals - 1:
      return -1
    return prediction
  
  def gen_signal(self, state, record=False):
    self.update_probs()
    try:
      signal = np.random.choice(self.num_signals, p=self.signal_weights.T[state])
    except ValueError as e:
      import pdb; pdb.set_trace()
    if self.null_signal and signal == self.num_signals-1:
      signal = -1
    self.curr_signal = signal

    if record:
      self.signal_history.append(deepcopy(self.signal_weights))

    return signal
  
  def update(self, curr_game: dict):
    # update weights of self.classifer partial_fit()
    assert self.classifier is not None
    
    state, signal = curr_game["state"], curr_game["signal"]
    reward = curr_game["reward"]
    training_sample = np.array([[state]])

    training_label = np.array([signal])
    # if there is a negative reward on a non-null signal,
    if reward < 0 and signal != -1:
      # mark the "correct" signal for training as the opposite to what was sent.
      if signal == 0:
        training_label[0] = 1
      else:
        training_label[0] = 0
    # if we have the null signal,
    elif signal == -1:
      # send it no matter what.
      training_label[0] = 2 # this is just an indexing switch.

    # partial fit our classifier
    try:
      self.classifier.partial_fit(training_sample, training_label)
    except ValueError as e:
      # partial fitting didn't work
      import pdb; pdb.set_trace()

    self.update_probs()
    
  def update_probs(self):
    # update "weights" with probability predictions
    for state in range(self.num_states):
      preds = self.classifier.predict_proba(np.array([[state]]))
      self.signal_weights[:, state] = preds


class SKReciever(Receiver):
  def __init__(self, *args, **kwargs):
    Receiver.__init__(self, *args, **kwargs)
    self.classifier = None
    self.all_actions = np.array(range(self.num_actions))

  def set_classifier(self, clf):
    # set classifier
    self.classifier = clf
    # random seed
    
    action1 = np.random.randint(0, self.num_actions)
    action2 = np.random.randint(0, self.num_actions)
    training_sample = np.array([[0], [1]])
    training_label = np.array([action1, action2])
    self.classifier.partial_fit(training_sample, training_label, classes=self.all_actions)
    pass

  def gen_action_direct(self, signal, record=False):
    assert self.classifier is not None
    self.update_probs()
    if record:
      self.action_history.append(self.action_weights)
    pred = self.classifier.predict(np.array([[signal]]))
    prediction = pred[0]
    return prediction
  
  def gen_action(self, signal, record=False):
    self.update_probs()
    if signal == -1:
      action = -1
    else:
      try:
        action = np.random.choice(self.num_actions, p=self.action_weights[signal])
      except ValueError as e:
        import pdb; pdb.set_trace()
    self.curr_action = action

    if record:
      self.action_history.append(deepcopy(self.action_weights))

    return action

  def update(self, curr_game: dict):
    assert self.classifier is not None
    signal, action = curr_game["signal"], curr_game["action"]
    reward = curr_game["reward"]
    world_state = curr_game["state"]
    training_sample = np.array([[signal]])

    if signal == -1:
      return

    training_label = np.array([action])
    if reward < 0:
      # TODO: Debate this
      # what should we code the training label to when we're wrong?
      # this is like giving a peek to the Reciever.
      training_label[0] = world_state

    try:
      self.classifier.partial_fit(training_sample, training_label)
    except ValueError as e:
      import pdb; pdb.set_trace()

  def update_probs(self):
    for signal in range(self.num_signals):
      preds = self.classifier.predict_proba(np.array([[signal]]))
      self.action_weights[signal, :] = preds
