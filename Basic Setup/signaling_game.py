import numpy as np
import csv  # Added for saving null-signal usage results
from sklearn.neural_network import MLPClassifier
from agents import Sender, Receiver, SKSender, SKReciever
from display import gen_gif
import os
import csv


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

    Simulates the result of a repeated signaling game between a sender and a receiver, 
    given the number of (world) states, the number of signals for the sender, and the 
    number of actions for the receiver.
    Implements a linear reward function based on the distance between the current state 
    and the current action.

    Attributes:
        num_states (int): number of (world) states
        num_signals (int): number of signals
        num_actions (int): number of actions
        reward_param (tuple[float, float]): the reward parameters
        reward_fn (function): the reward function
        null_signal (boolean): whether null signals are in use
        random (np.random.Generator): random generator for states
        receiver (Receiver): the receiver agent
        sender (Sender): the sender agent
        curr_state, curr_signal, curr_action (int): current (world) state, signal, and action
        history (list): list of dicts (state, signal, action, reward) per round
    """
    def __init__(self, num_states: int, num_signals: int, num_actions: int,
                 reward_param: tuple[float, float], null_signal=False):
        """Initializes the game

        Args:
            num_states (int): number of (world) states
            num_signals (int): number of signals
            num_actions (int): number of actions
            reward_param (tuple[float, float]): parameters for the reward function
            null_signal (bool): if True, enable null-signal logic
        """
        self.num_states = num_states
        self.num_signals = num_signals
        self.num_actions = num_actions

        self.reward_param = reward_param
        self.reward_fn = linear_reward_fn(reward_param, null_signal)

        self.null_signal = null_signal
        self.random = np.random.default_rng()

        # By default, using plain Sender and Receiver
        # (Alternatively, you can enable the scikit-learn versions)
        self.sender = Sender(self.num_states, self.num_signals, null_signal)
        # self.sender = SKSender(self.num_states, self.num_signals, null_signal)
        # self.sender.set_classifier(MLPClassifier(
        #     hidden_layer_sizes=(3,3),
        #     random_state=1,
        #     solver='adam',
        #     alpha=0.1
        # ))

        self.reciever = Receiver(self.num_signals, self.num_actions)
        # self.receiver = SKReciever(self.num_signals, self.num_actions)
        # self.receiver.set_classifier(MLPClassifier(
        #     hidden_layer_sizes=(3,3),
        #     random_state=1,
        #     solver='adam',
        #     alpha=0.1
        # ))

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
            # add 1 signal slot if null_signal == True
            for m in range(self.num_signals + (1 if self.null_signal else 0)):
                eps = 0
                for a in range(self.num_actions):
                    # skip if null signal
                    if not self.null_signal or m != self.num_signals:
                        eps += action_prob[m, a] * self.evaluate(w, a)
                epw += signal_prob[m, w] * eps
            ep += epw
        return ep / self.num_states

    def optimal_payoff(self) -> float:
        """An approximate or simplified formula for the 'optimal' payoff."""
        opt_bucket = 2 * (self.reward_param[0] // self.reward_param[1]) + 1

        if self.null_signal and opt_bucket < self.num_states // self.num_signals:
            return (self.reward_param[0]*opt_bucket
                    - self.reward_param[1]*(opt_bucket**2-1)/4) \
                   * self.num_signals / self.num_states
        else:
            m = self.num_states // self.num_signals
            z = self.num_states % self.num_signals
            if m % 2 == 0:
                return self.reward_param[0] - self.reward_param[1] * m * (self.num_states+z)/(4*self.num_states)
            else:
                return self.reward_param[0] - self.reward_param[1] * (m+1) * (self.num_states+z-self.num_signals)/(4*self.num_states)

    def info_measure(self, signal_prob) -> float:
        """Calculates a simple information measure (approx. mutual info)."""
        # Normalize each row to sum to 1
        prob = (signal_prob.T / np.sum(signal_prob, axis=1)).T

        inf = 0
        for i in range(self.num_signals):
            if self.null_signal and i == self.num_signals:
                break
            inf_sig = 0
            for j in range(self.num_states):
                if prob[i, j] > 0:
                    inf_sig += prob[i, j] * np.log(prob[i, j] * self.num_states)
            inf += (np.sum(signal_prob[i]) / self.num_states) * inf_sig
        return inf

    def optimal_info(self) -> float:
        """A rough estimate of the 'optimal' info measure for comparison."""
        opt_m = 2 * (self.reward_param[0] // self.reward_param[1]) + 1
        m_null = self.num_states - self.num_signals * opt_m
        m = self.num_states // self.num_signals
        z = self.num_states % self.num_signals

        if self.null_signal and m_null > 0:
            opt_info = opt_m/self.num_states * self.num_signals * np.log(self.num_states/opt_m)
        else:
            opt_info = np.log(self.num_states) \
                       - (z/self.num_signals)*np.log(m+1) \
                       - (1-z/self.num_signals)*np.log(m)
        return opt_info

    def gen_state(self) -> int:
        """Generates a random (world) state

        Returns:
            int: a new current state
        """
        return self.random.integers(self.num_states)

    def update_history(self, reward: float):
        """Updates the history of simulations

        Args:
            reward (float): the reward of the current simulation
        """
        self.history.append({
            "state": self.curr_state,
            "signal": self.curr_signal,
            "action": self.curr_action,
            "reward": reward
        })

    def __call__(self, num_iter: int, record_interval=-1, repeat_num: int = None):
        """Runs the simulation

        Args:
            num_iter (int): number of iterations (simulations)
            record_interval (int): if > 0, record snapshots for a GIF every N steps
            repeat_num (int): optional index for repeated runs
        """
        # Main simulation loop
        for i in range(num_iter):
            state = self.gen_state()
            self.curr_state = state

            # If we want to record, pass record=True to gen_signal/gen_action
            if record_interval > 0 and (i+1) % record_interval == 0:
                signal = self.sender.gen_signal(state, record=True)
                action = self.reciever.gen_action(signal, record=True)
            else:
                signal = self.sender.gen_signal(state)
                action = self.reciever.gen_action(signal)

            self.curr_signal = signal
            self.curr_action = action

            reward = self.evaluate(state, action)
            self.update_history(reward)
            self.sender.update(self.history[-1])
            self.reciever.update(self.history[-1])

        # Generate a GIF if requested
        if record_interval != -1:
            if repeat_num is None:
                gif_filename = f"./simulations/{self.num_states}_{self.num_signals}_{self.num_actions}/{self.reward_param}{'_null' if self.null_signal else ''}_{num_iter}.gif"
            else:
                gif_filename = f"./simulations/{self.num_states}_{self.num_signals}_{self.num_actions}/{self.reward_param}{'_null' if self.null_signal else ''}_{num_iter}_{repeat_num}.gif"

            gen_gif(
                self.sender.signal_history,
                self.reciever.action_history,
                self.expected_payoff,
                self.optimal_payoff(),
                self.info_measure,
                self.optimal_info(),
                num_iter,
                record_interval,
                duration=100,
                output_file=gif_filename
            )

        # Count and store how often the null signal was used 
        if self.null_signal:
            null_usage_by_state = np.zeros(self.num_states, dtype=int)
            state_counts = np.zeros(self.num_states, dtype=int)

            for h in self.history:
                s = h["state"]
                sig = h["signal"]
                state_counts[s] += 1
                if sig == -1:
                    null_usage_by_state[s] += 1

            print("\nNull-signal usage")
            print("State : count_of_null : total_occurrences : fraction_null")
            for s in range(self.num_states):
                if state_counts[s] > 0:
                    frac = null_usage_by_state[s] / state_counts[s]
                else:
                    frac = 0
                print(f"{s:5d} : {null_usage_by_state[s]:13d} : {state_counts[s]:17d} : {frac:12.3f}")

            # Store the usage stats in a CSV
            run_id = repeat_num if repeat_num is not None else 'none'
            csv_dir = f"./simulations/{self.num_states}_{self.num_signals}_{self.num_actions}"
            os.makedirs(csv_dir, exist_ok=True)

            csv_filename = (
                f"{csv_dir}/null_usage_{int(self.reward_param[0])}_{self.reward_param[1]}_{run_id}.csv"
            )
            with open(csv_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["state", "null_count", "total_count", "null_fraction"])
                for s in range(self.num_states):
                    if state_counts[s] > 0:
                        frac = null_usage_by_state[s] / state_counts[s]
                    else:
                        frac = 0
                    writer.writerow([s, null_usage_by_state[s], state_counts[s], frac])
