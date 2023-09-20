import numpy as np

from agents import Sender, Receiver
from display import gen_gif

def linear_reward_fn(param: tuple[float, float], null_signal=False):
  """Returns a linear reward function based on the distance between the state and the action

  Args:
    param (tuple[float, float]): the two parameters to be used as constants in the
      reward function
    null_signal (boolean): null signal case

  Returns:
    function: a linear reward function
  """
  def get_reward(state, action):
    if null_signal and action == -1:
      return 0
    
    # min_dist = min([abs(state - a) for a in action])
    return param[0] - param[1] * abs(state - action)
  
  return get_reward


class SignalingGame:
  """A signaling game between a sender and a receiver

  Simulates the result of a repeated signaling game between a sender and a receiver, given the number of (world) states, the number of signals for the sender, and the number of actions for the receiver.
  Implements a linear reward function based on the distance between the current state and the current action.

  Attributes:
    num_states, num_signals, num_actions (int): the number of (world) states,
      signals, and actions
    reward_param (tuple[float, float]): the reward parameters
    reward_fn (function): the reward function
    null_signal (boolean): null signal case
    random (np.random.Generator): the random generator for the states
    receiver (Receiver): the receiver
    sender (Sender): the sender
    curr_state, curr_signal, curr_action (int): the current (world) state,
      signal, and action
    history (list): the history of the repeated games
  """
  def __init__(self, num_states: int, num_signals: int, num_actions: int, 
               reward_param: tuple[float, float], null_signal=False):
    """Initalizes the instances to set up the game

    Args:
      num_states (int): the number of (world) states
      num_signals (int): the number of signals
      num_actions (int): the number of actions
      reward_param (tuple[float, float]): the necessary parameters for the reward
        function
      null_signal (boolean): null signal case
    """
    self.num_states = num_states
    self.num_signals = num_signals
    self.num_actions = num_actions

    self.reward_param = reward_param

    self.reward_fn = linear_reward_fn(reward_param, null_signal)

    self.null_signal = null_signal

    self.random = np.random.default_rng(0) # default seed = 0. Can be changed with set_random_seed()

    self.sender = Sender(self.num_states, self.num_signals, null_signal)
    self.receiver = Receiver(self.num_signals, self.num_actions)

    self.curr_state = None
    self.curr_signal = None
    self.curr_action = None

    self.history = []

  def set_random_seed(self, seed: int):
    """Creates a new np.random.Generator with the given seed

    Args:
      seed (int): the seed for the np.random.Generator
    """
    self.random = np.random.default_rng(seed)

  def evaluate(self, state: int, action: int) -> float:
    """Calculates the reward to the agents
    
    Args:
      state (int): the given (world) state
      action (int): the given action
        
    Returns:
      float: the reward based on the state and the action
    """
    return self.reward_fn(state, action)
  
  def expected_payoff(self, signal_prob, action_prob) -> float:
    """Calculates the expected payoff given the probabilities of the Sender and the Receiver
    
    Args:
      signal_prob (np.ndarray): signal probabilities
      action_prob (np.ndarray): action probabilities

    Returns:
      float: the expected payoff
    """
    ep = 0
    for w in range(self.num_states):
      epw = 0
      for m in range(self.num_signals + (1 if self.null_signal else 0)):
        eps = 0
        for a in range(self.num_actions):
          if not self.null_signal or m != self.num_signals:
            eps += action_prob[m, a] * self.evaluate(w, a)

        epw += signal_prob[m, w] * eps

      ep += epw

    return ep / self.num_states

  def gen_state(self) -> int:
    """Generates a random (world) state

    Returns:
      int: a new current state
    """
    return self.random.integers(self.num_states)
  
  def update_history(self, reward: int):
    """Updates the history of simulations
    
    Args:
      reward (int): the reward of the current simulation
    """
    self.history.append({"state": self.curr_state,
                         "signal": self.curr_signal,
                         "action": self.curr_action,
                         "reward": reward})
  
  def __call__(self, num_iter: int, record_interval=-1):
    """Runs the simulation

    Args:
      num_iter (int): number of iterations (simulations)
      record_interval (int): the simulations to be recorded and made into an
        image to display. -1 implies no image/gif will be displayed
    """
    for i in range(num_iter):
      state = self.gen_state()
      self.curr_state = state
      try:
        if (record_interval > 0 and (i+1) % record_interval == 0) \
            or (record_interval == -1 and i == num_iter-1):
          signal = self.sender.gen_signal(state, True)
          action = self.receiver.gen_action(signal, True)
        else:
          signal = self.sender.gen_signal(state)
          action = self.receiver.gen_action(signal)
      except ValueError:
        print(i)
        exit(0)
      self.curr_signal = signal
      self.curr_action = action

      reward = self.evaluate(state, action)
      self.update_history(reward)
      self.sender.update(self.history[-1])
      self.receiver.update(self.history[-1])

      if i == num_iter - 1:
        print(f"game={self.history[-1]}")
        print("Signal weights & probs:")
        print(self.sender.signal_weights)
        self.sender.print_signal_prob()
        print("Action weights & probs:")
        print(self.receiver.action_weights)
        self.receiver.print_action_prob()
        # ep = self.expected_payoff(self.sender.signal_history[-1], self.receiver.action_history[-1])
        # print(ep)

    if record_interval == -1:
      return
    
    gif_filename = f"./simulations/{self.num_states}_{self.num_signals}_{self.num_actions}/{self.reward_param}{'_null' if self.null_signal else ''}_{num_iter}.gif"
    
    gen_gif(self.sender.signal_history, self.receiver.action_history, self.expected_payoff, num_iter, record_interval, 100, gif_filename)
  