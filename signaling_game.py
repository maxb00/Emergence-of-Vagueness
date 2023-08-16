import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import seaborn as sns
import os
from IPython.display import HTML, display

from agents import Sender, Receiver

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
    return param[0] - param[1] * abs(state - action)
  
  return get_reward


class SignalingGame:
  """A signaling game between a sender and a receiver

  Simulates the result of a repeated signaling game between a sender and a receiver, given the number of (world) states, the number of signals for the sender, and the number of actions for the receiver.
  Implements a linear reward function based on the distance between the current state and the current action.

  Attributes:
    num_states, num_signals, num_actions (int): the number of (world) states,
      signals, and actions
    reward_fn (function): the reward function
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

    self.sender = Sender(self.num_states, self.num_signals, self, null_signal)
    self.receiver = Receiver(self.num_signals, self.num_actions, self)

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
    
  def gen_gif(self, num_iter: int, record_interval: int, duration: int):
    """Generates a heatmap gif of the whole simulation and saves it into 
    test.gif

    Args:
      num_images (int): the number of images in the gif
      record_interval (int): number of simulations between each image
      duration (int): the duration an image is shown in the gif
    """
    num_images = num_iter // record_interval

    if not os.path.exists("./images"):
      os.mkdir("images")

    epx = []
    epy = []

    for i in range(num_images):
      fig, axs = plt.subplots(3, 1, figsize=(8, 6))
      plt.tight_layout(pad=3)

      epx.append((i+1)*record_interval)
      epy.append(self.expected_payoff(self.sender.signal_history[i], self.receiver.action_history[i]))

      sns.heatmap(self.sender.signal_history[i], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, 
      fmt=".1f", ax=axs[0])
      axs[0].set_xlabel("states")
      axs[0].set_ylabel("messages")
      axs[0].set_title("Sender\'s weights")

      sns.heatmap(self.receiver.action_history[i], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, 
      fmt=".1f", ax=axs[1])
      axs[1].set_xlabel("actions")
      axs[1].set_ylabel("messages")
      axs[1].set_title("Receiver\'s weights")

      axs[2].plot(epx, epy)
      axs[2].set_xlabel("rollout")
      axs[2].set_ylabel("expected payoff")
      axs[2].set_title("Expected payoff by rollout")

      fig.suptitle(f"Rollout {(i+1)*record_interval}")
      plt.savefig(f"./images/game_{(i+1)*record_interval}.png")
      plt.close(fig)

    images = []
    for filename in [f"./images/game_{(j+1)*record_interval}.png" for j in range(num_images)]:
      images.append(imageio.imread(filename))
    imageio.mimsave(f"{self.num_states}_{self.num_signals}_{self.num_actions}_{self.reward_param}{'_null' if self.null_signal else ''}_{num_iter}.gif", images, duration=duration)
    display(HTML('<img src="test.gif">'))
  
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
      if record_interval > 0 and (i+1) % record_interval == 0:
        signal = self.sender.gen_signal(state, True)
        action = self.receiver.gen_action(signal, True)
      else:
        signal = self.sender.gen_signal(state)
        action = self.receiver.gen_action(signal)
      self.curr_signal = signal
      self.curr_action = action

      reward = self.evaluate(state, action)
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

    if record_interval == -1:
      return
    
    self.gen_gif(num_iter, record_interval, 100)
  

def main():
  game = SignalingGame(20, 2, 20, (2,0.5), null_signal=True)
  game(2000, 25)

if __name__ == '__main__':
  main()