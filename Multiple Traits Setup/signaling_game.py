import numpy as np
import math

from agents import Sender, Receiver
from display import gen_gif

def linear_reward_fn(param: tuple[float, float], null_signal=False):
  """Returns a linear reward function based on the distance between the state and the action

  Args:
    param (tuple[float, float]): the two parameters to be used as constants in the reward function
    null_signal (boolean): null signal case

  Returns:
    function: a linear reward function
  """
  def get_reward(state, action):
    if null_signal and action == -1:
      return 0
    
    l2dist = 0
    for s, a in zip(state, action):
      l2dist += abs(s - a)

    return param[0] - param[1] * l2dist
  
  return get_reward

def normpdf(x: float, mean: float, std: float) -> float:
  """Returns the probability of a variable in a normal distribution given the mean and standard deviation

  Args:
    x (float): the variable
    mean (float): the mean
    std (float): the standard deviation

  Returns:
    float: the probability
  """
  var = float(std)**2
  denom = (2*math.pi*var)**.5
  num = math.exp(-(float(x)-float(mean))**2/(2*var))
  return num/denom

def gen_state_prob(num_traits: int, num_states: int):
  """Returns the probability matrix of the states of the game

  Args:
    num_traits (int): the number of traits
    num_states (int): the number of states per trait
  
  Returns:
    np.ndarray: the probability matrix (flattened)
  """
  mean = (num_states-1) / 2
  std = mean / 1.25

  state_prob = np.zeros(tuple([num_states] * num_traits), dtype=np.float64)
  for inds in np.ndindex(state_prob.shape):
    prob = 1
    for ind in inds:
      prob *= normpdf(ind, mean, std)
    
    state_prob[inds] = prob

  state_prob = state_prob / np.sum(state_prob)

  return state_prob.flatten()


class SignalingGame:
  """A signaling game between a sender and a receiver

  Simulates the result of a repeated signaling game between a sender and a receiver, given the number of (world) states, the number of signals for the sender, and the number of actions for the receiver.
  Implements a linear reward function based on the distance between the current state and the current action.

  Attributes:
    num_traits, num_states, num_signals, num_actions (int): the number of       
      traits, (world) states, signals, and actions
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
  def __init__(self, num_traits: int, num_states: int, num_signals: int, 
               num_actions: int, reward_param: tuple[float, float],
               null_signal=False):
    """Initalizes the instances to set up the game

    Args:
      num_traits (int): the number of traits
      num_states (int): the number of (world) states
      num_signals (int): the number of signals
      num_actions (int): the number of actions
      reward_param (tuple[float, float]): the necessary parameters for the reward
        function
      null_signal (boolean): null signal case
    """
    self.num_traits = num_traits
    self.num_states = num_states
    self.num_signals = num_signals
    self.num_actions = num_actions

    self.state_prob = gen_state_prob(num_traits, num_states)

    self.reward_param = reward_param

    self.reward_fn = linear_reward_fn(reward_param, null_signal)

    self.null_signal = null_signal

    self.random = np.random.default_rng() 

    self.sender = Sender(self.num_traits, self.num_states, self.num_signals, null_signal)
    self.receiver = Receiver(self.num_traits, self.num_signals, self.num_actions)

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
  
  # HARD_CODED
  def floor_strat(self):
    """Floor strategies for comparison purposes"""
    sv = 1e-7
    
    if self.num_states == 6:
      siglv = 1 - 3*sv
      actlv = 1 - 35*sv

      flr_sig = np.array([
        [[sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv]],
        [[sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv]],
        [[sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv]],
        [[sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv]],
      ])

      for i in range(self.num_states):
        for j in range(self.num_states):
          rand_sig = self.random.integers(self.num_signals)
          flr_sig[rand_sig][i][j] = siglv

      flr_act = np.array([
        [[sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, actlv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv]],
        [[sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, actlv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv]],
        [[sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, actlv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv]],
        [[sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, actlv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv]],
      ])
    elif self.num_signals == 4:
      siglv = 1 - 3*sv
      actlv = 1 - 8*sv

      flr_sig = np.array([
        [[sv, sv, sv],
        [sv, sv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, sv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, sv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, sv, sv],
        [sv, sv, sv]],
      ])

      for i in range(self.num_states):
        for j in range(self.num_states):
          rand_sig = self.random.integers(self.num_signals)
          flr_sig[rand_sig][i][j] = siglv

      flr_act = np.array([
        [[sv, sv, sv],
        [sv, actlv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, actlv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, actlv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, actlv, sv],
        [sv, sv, sv]],
      ])
    elif self.num_signals == 5:
      siglv = 1 - 4*sv
      actlv = 1 - 8*sv

      flr_sig = np.array([
        [[sv, sv, sv],
        [sv, sv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, sv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, sv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, sv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, sv, sv],
        [sv, sv, sv]]
      ])

      for i in range(self.num_states):
        for j in range(self.num_states):
          rand_sig = self.random.integers(self.num_signals)
          flr_sig[rand_sig][i][j] = siglv

      flr_act = np.array([
        [[sv, sv, sv],
        [sv, actlv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, actlv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, actlv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, actlv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, actlv, sv],
        [sv, sv, sv]]
      ])

    return flr_sig, flr_act
  
  # HARD_CODED
  def optimal_strat(self):
    """Optimal strategies for comparison purposes"""
    sv = 1e-7
    
    if self.num_states == 6:
      siglv = 1 - 3*sv
      actlv = 1 - 35*sv

      opt_sig = np.array([
        [[siglv, siglv, siglv, sv, sv, sv],
        [siglv, siglv, siglv, sv, sv, sv],
        [siglv, siglv, siglv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv]],
        [[sv, sv, sv, siglv, siglv, siglv],
        [sv, sv, sv, siglv, siglv, siglv],
        [sv, sv, sv, siglv, siglv, siglv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv]],
        [[sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [siglv, siglv, siglv, sv, sv, sv],
        [siglv, siglv, siglv, sv, sv, sv],
        [siglv, siglv, siglv, sv, sv, sv]],
        [[sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, siglv, siglv, siglv],
        [sv, sv, sv, siglv, siglv, siglv],
        [sv, sv, sv, siglv, siglv, siglv]],
      ])

      opt_act = np.array([
        [[sv, sv, sv, sv, sv, sv],
        [sv, actlv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv]],
        [[sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, actlv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv]],
        [[sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, actlv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv]],
        [[sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, sv, sv],
        [sv, sv, sv, sv, actlv, sv],
        [sv, sv, sv, sv, sv, sv]],
      ])
    elif self.num_signals == 4:
      siglv = 1 - 3*sv
      actlv = 1 - 8*sv

      opt_sig = np.array([
        [[siglv, siglv, siglv],
        [sv, sv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, sv, siglv],
        [sv, sv, siglv]],
        [[sv, sv, sv],
        [sv, siglv, sv],
        [sv, siglv, sv]],
        [[sv, sv, sv],
        [siglv, sv, sv],
        [siglv, sv, sv]],
      ])

      opt_act = np.array([
        [[sv, actlv, sv],
        [sv, sv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, sv, actlv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, actlv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [actlv, sv, sv],
        [sv, sv, sv]],
      ])
    elif self.num_signals == 5:
      siglv = 1 - 4*sv
      actlv = 1 - 8*sv

      opt_sig = np.array([
        [[siglv, siglv, sv],
        [sv, sv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, sv, siglv],
        [sv, sv, siglv]],
        [[sv, sv, sv],
        [sv, sv, sv],
        [sv, siglv, siglv]],
        [[siglv, sv, sv],
        [siglv, sv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, siglv, sv],
        [sv, sv, sv]]
      ])

      opt_act = np.array([
        [[sv, actlv, sv],
        [sv, sv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, sv, actlv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, sv, sv],
        [sv, actlv, sv]],
        [[sv, sv, sv],
        [actlv, sv, sv],
        [sv, sv, sv]],
        [[sv, sv, sv],
        [sv, actlv, sv],
        [sv, sv, sv]]
      ])

    return opt_sig, opt_act
  
  def expected_payoff(self, signal_prob, action_prob) -> float:
    """Calculates the expected payoff given the probabilities of the Sender and the Receiver
    
    Args:
      signal_prob (np.ndarray): signal probabilities
      action_prob (np.ndarray): action probabilities

    Returns:
      float: the expected payoff
    """
    total_states = self.num_states**self.num_traits
    signal_prob = signal_prob.reshape(self.num_signals, total_states)
    action_prob = action_prob.reshape(self.num_signals, total_states)

    ep = 0
    for w in range(total_states):
      epw = 0
      for m in range(self.num_signals + (1 if self.null_signal else 0)):
        eps = 0
        for a in range(total_states):
          if not self.null_signal or m != self.num_signals:
            state = self.unnumerize(w)
            action = self.unnumerize(a)
            eps += action_prob[m, a] * self.evaluate(state, action)

        epw += signal_prob[m, w] * eps

      ep += self.state_prob[w] * epw

    return ep / total_states

  def optimal_payoff(self) -> float:
    """Calculates the optimal payoff"""

    opt_sig, opt_act = self.optimal_strat()

    return self.expected_payoff(opt_sig, opt_act)
  
  def floor_payoff(self):
    """Calculates the floor payoff"""

    floor_sig, floor_act = self.floor_strat()

    return self.expected_payoff(floor_sig, floor_act)
  
  def info_measure(self, signal_prob, weighted=True) -> float:
    """Calculates the information content of the signals

    Args:
      signal_prob (np.ndarray): the probabilities of the signals
      weighted (boolean): weighted/unweighted options

    Returns:
      inf (float): total information content measure
      inf_sigs (list): information content by signal
      inf_states (list): information content by state
    """
    total_states = self.num_states**self.num_traits
    signal_prob = signal_prob.reshape(self.num_signals, total_states)

    prob = np.zeros_like(signal_prob)
    for i in range(self.num_signals):
      for j in range(total_states):
        prob[i, j] = signal_prob[i, j] * self.state_prob[j]
    prob_sig = [np.sum(prob[i]) for i in range(self.num_signals)]
    prob = (prob.T / np.sum(prob, axis=1)).T

    inf = 0
    inf_sigs = []
    inf_states = []
    for i in range(self.num_signals):
      if self.null_signal and i == self.num_signals:
        break
      inf_sig = 0
      inf_states.append([])
      for j in range(total_states):
        inf_state = prob[i, j] * np.log(prob[i, j]/self.state_prob[j])
        inf_sig += inf_state
        
        if weighted:
          inf_state = prob_sig[i] * inf_state

        inf_states[i].append(inf_state)

      if weighted:
        inf_sig = prob_sig[i] * inf_sig

      inf_sigs.append(inf_sig)
      inf += inf_sig

    new_size = [self.num_signals]
    new_size.extend([self.num_states] * self.num_traits)
    inf_states = np.resize(np.array(inf_states), tuple(new_size))

    return inf, inf_sigs, inf_states
  
  # HARD_CODED FOR 2 TRAITS
  def info_measure_by_trait(self, signal_prob, weighted=True) -> float:
    """Calculates the information content by trait

    Args:
      signal_prob (np.ndarray): the probabilities of the signals
      weighted (boolean): weighted/unweighted options

    Returns:
      inf_by_trait (list): information content by trait
    """
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
    prob_sig = [np.sum(prob_t1[i]) for i in range(self.num_signals)]
    prob_t1 = (prob_t1.T / np.sum(prob_t1, axis=1)).T
    prob_t2 = (prob_t2.T / np.sum(prob_t2, axis=1)).T
    state_prob_t = state_prob_t / np.sum(state_prob_t)

    inf_by_trait = [[], []]
    for i in range(self.num_signals):
      if self.null_signal and i == self.num_signals:
        break
      inf_sig_t1 = 0
      inf_sig_t2 = 0
      for j in range(self.num_states):
        inf_sig_t1 += prob_t1[i, j] * np.log(prob_t1[i, j]/state_prob_t[j])
        inf_sig_t2 += prob_t2[i, j] * np.log(prob_t2[i, j]/state_prob_t[j])

      if weighted:
        inf_sig_t1 = prob_sig[i] * inf_sig_t1
        inf_sig_t2 = prob_sig[i] * inf_sig_t2

      inf_by_trait[0].append(inf_sig_t1)
      inf_by_trait[1].append(inf_sig_t2)

    return inf_by_trait
  
  def optimal_info(self, weighted=True) -> float:
    """Calculates the total information content for the optimal strategy"""

    opt_sig, _ = self.optimal_strat()

    return self.info_measure(opt_sig, weighted)[0]
  
  def floor_info(self, weighted=True) -> float:
    """Calculates the total information content for the floor strategy"""

    floor_sig, _ = self.floor_strat()

    return self.info_measure(floor_sig, weighted)[0]

  def gen_state(self) -> int:
    """Generates a random (world) state

    Returns:
      int: a new current state (unflattened)
    """
    state = self.random.choice(self.num_states**self.num_traits, p=self.state_prob)

    return self.unnumerize(state)
  
  def update_history(self, reward: int):
    """Updates the history of simulations
    
    Args:
      reward (int): the reward of the current simulation
    """
    self.history.append({"state": self.curr_state,
                         "fstate": self.numerize(self.curr_state),
                         "signal": self.curr_signal,
                         "action": self.curr_action,
                         "faction": self.numerize(self.curr_action),
                         "reward": reward})
    
  def numerize(self, state) -> int:
    """Turn a state/action from an nd array to a number"""
    fstate = 0
    for i, s in enumerate(reversed(state)):
      fstate += s * (self.num_states**i)

    return fstate
  
  def unnumerize(self, action: int):
    """Turn a numerized state/action into an nd array"""
    ufaction = []
    while action > 0:
      ufaction.insert(0, action % self.num_states)
      action = action // self.num_states

    while len(ufaction) < self.num_traits:
      ufaction.insert(0, 0)

    return ufaction
  
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
      state = self.numerize(state)
      if record_interval > 0 and (i+1) % record_interval == 0:
        signal = self.sender.gen_signal(state, True)
        action = self.receiver.gen_action(signal, True)
      else:
        signal = self.sender.gen_signal(state)
        action = self.receiver.gen_action(signal)
      action = self.unnumerize(action)
      self.curr_signal = signal
      self.curr_action = action

      reward = self.evaluate(self.curr_state, self.curr_action)
      self.update_history(reward)
      self.sender.update(self.history[-1])
      self.receiver.update(self.history[-1])

      # if i == num_iter - 1:
      #   print(f"game={self.history[-1]}")
      #   print("Signal weights & probs:")
      #   print(self.sender.signal_weights)
      #   self.sender.print_signal_prob()
      #   print("Action weights & probs:")
      #   print(self.receiver.action_weights)
      #   self.receiver.print_action_prob()

    # if record_interval == -1:
    #   return self.info_measure(self.sender.signal_history[-1])

    """ TESTING: Info content by rows/columns in 2D/3D """
    # _, _, info_state = self.info_measure(self.sender.signal_history[-1], False)
    # _, _, w_info_state = self.info_measure(self.sender.signal_history[-1])

    # info_by_state_t1 = [np.sum(info_state[:, :, i]) for i in range(self.num_states)]
    # info_by_state_t2 = [np.sum(info_state[:, i, :]) for i in range(self.num_states)]
    # info_by_state_t3 = [np.sum(info_state[:, i, :, :]) for i in range(self.num_states)]

    # print(f"Unweighted trait 1's low|medium|high: {info_by_state_t1[0]:.5f} | {info_by_state_t1[1]:.5f} | {info_by_state_t1[2]:.5f}")
    # print(f"Unweighted trait 2's low|medium|high: {info_by_state_t2[0]:.5f} | {info_by_state_t2[1]:.5f} | {info_by_state_t2[2]:.5f}")
    # print(f"Unweighted trait 3's low|medium|high: {info_by_state_t3[0]:.5f} | {info_by_state_t3[1]:.5f} | {info_by_state_t3[2]:.5f}")

    # w_info_by_state_t1 = [np.sum(w_info_state[:, :, i]) for i in range(self.num_states)]
    # w_info_by_state_t2 = [np.sum(w_info_state[:, i, :]) for i in range(self.num_states)]
    # w_info_by_state_t3 = [np.sum(w_info_state[:, i, :, :]) for i in range(self.num_states)]

    # print(f"Weighted trait 1's low|medium|high: {w_info_by_state_t1[0]:.5f} | {w_info_by_state_t1[1]:.5f} | {w_info_by_state_t1[2]:.5f}")
    # print(f"Weighted trait 2's low|medium|high: {w_info_by_state_t2[0]:.5f} | {w_info_by_state_t2[1]:.5f} | {w_info_by_state_t2[2]:.5f}")
    # print(f"Weighted trait 3's low|medium|high: {w_info_by_state_t3[0]:.5f} | {w_info_by_state_t3[1]:.5f} | {w_info_by_state_t3[2]:.5f}")

    gif_filename = f"./simulations/v7/{self.num_states}_{self.num_signals}_{self.num_actions}/{self.reward_param}{'_null' if self.null_signal else ''}_{num_iter}" #HARD-CODED
  
    gen_gif(self, num_iter, record_interval, 100, gif_filename)

    # return (info_by_state_t1, info_by_state_t2, w_info_by_state_t1, w_info_by_state_t2)

    # return (info_by_state_t1, info_by_state_t2, info_by_state_t3, w_info_by_state_t1, w_info_by_state_t2, w_info_by_state_t3)
  
  