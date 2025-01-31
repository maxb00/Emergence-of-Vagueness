from typing import List, Tuple, Type
import numpy as np
import gc
import matplotlib
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numpy.typing import NDArray
from warnings import warn, catch_warnings
from collections import Counter, defaultdict
from random import sample
from matplotlib import colormaps
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from math import inf, exp
from pdb import set_trace

matplotlib.use("TKAgg")

def SuppressWarning(func):
    def wrapper(*args, **kwargs):
        with catch_warnings(action='ignore'):
            return func(*args, **kwargs)
    return wrapper

"""
Basic Player. Learning algorithm: K-Nearest Neighbors
"""
class Player:
    def __init__(
            self, policy: NDArray = None, 
            n_signals: int = 2, n_states: int = 100, 
            reward_struct: Tuple[float, float, float] = (1., 0.5, 0.)
    ):
        self.policy = policy
        self.signals = n_signals
        self.states = n_states
        self.trained_neigbors = 5
        self.threshold = 0.5
        self.reward = reward_struct
        if policy is not None:
            self.given_examples = []
        else:
            self.given_examples = None

    def poll(self, n_samples: int = -1, sampling_strat="random"):
        # randomly polls policy across all certain states
        assert self.policy is not None

        # iron out how many samples to take
        if n_samples == -1:
            # try to take 20%
            n_samples = len(self.policy) // 5
            # if 20% can't possibly sample each bucket, try.
            if n_samples < self.signals:
                n_samples = self.signals

        valid_moves = []
        groups = defaultdict(list)
        for i in range(self.states):
            group = self.policy[i]
            if group == 0:
                continue
            move = (i, group)
            valid_moves.append(move)
            groups[group].append(move)

        if len(valid_moves) == 0:
            return [(i,0) for i in range(self.states)]

        if sampling_strat == "random":
            if n_samples > len(valid_moves):
                return valid_moves
            else:
                return sample(valid_moves, n_samples)
        else:
            # sampling_strat == "uniform"
            per_group = n_samples // self.signals
            smallest_group = inf
            for group in groups:
                group_size = len(groups[group])
                if group_size != 0 and group_size < smallest_group:
                    smallest_group = group_size

            if smallest_group < per_group:
                per_group = smallest_group

            examples = []
            for group in groups:
                if len(groups[group]) >= per_group:
                    examples.extend(sample(groups[group], per_group))
                else:
                    examples.extend(groups[group])

            return examples


    def learn(self, examples, k_neighbors, vagueness=True, threshold=0.7):
        # learns policy from examples
        # ideally, we extend Player and override this method 
        # for each learning algorithm we want to implement
        if self.policy is not None:
            warn("This player already has a policy! Be aware, this will overwrite it.")

        if vagueness:
            one_above_half = math.floor(k_neighbors / 2) + 1
            min_threshold = one_above_half / k_neighbors
            assert threshold > min_threshold, "Threshold too low for vagueness"

        # do the learning
        working_policy = np.zeros(self.states, dtype=np.int64)
        n_examples = len(examples)
        k_neighbors = k_neighbors if n_examples > k_neighbors else n_examples
        for state in range(self.states):
            if vagueness == False:
                working_policy[state] = self.predict(examples, state, k_neighbors, False)
            else:
                pred, prob = self.predict(examples, state, k_neighbors, True)
                if prob >= threshold:
                    working_policy[state] = pred

        self.policy = working_policy
        self.given_examples = examples
        self.trained_neigbors = k_neighbors
        self.threshold = threshold


    def predict(self, examples, state, k_neighbors=5, vagueness=True):
        # Basic KNN
        neighbors = self.get_sorted_neighbors(examples, state)[:k_neighbors]

        # make prediction
        if vagueness == False:
            c = Counter()
            for n in neighbors:
                c[n[1]] += 1
            # return winner
            return c.most_common(1)[0][0]
        else:
            c = np.asarray([0] * self.signals)
            for n in neighbors:
                if n[1] - 1 < self.signals:
                    c[int(n[1])-1] += 1
            probs = c / k_neighbors
            if np.sum(probs) == 0:
                return 0, 0  # Return no prediction if no confidence
            most_confident = np.argmax(probs)
            prediction = most_confident + 1
            confidence = probs[most_confident]
            return prediction, confidence


    def get_sorted_neighbors(self, examples, state):
        # get distances
        distance = []
        for point_group in examples:
            point, group = point_group
            if isinstance(point, np.int64) or isinstance(point, int):
                dist = abs(state - point)
            else:
                # implement distance function for cartesian points
                dist = 0.5
            distance.append((dist, group)) 

        # sort distances, take k best
        return sorted(distance, key=lambda x: x[0])  # Modified to sort by distance


    def utility(self, init_policy):
        # compares (self) player's policy to given policy
        # returns utility score as ratio (0-1)
        assert self.policy is not None

        score = 0
        for state in range(self.states):
            if self.policy[state] == 0:
                score += self.reward[1]
            elif self.policy[state] == init_policy[state]:
                score += self.reward[0]
            else:
                # default: 0. Punishment for changing singals.
                score += self.reward[2]

        return score / self.states


    def plot_strategy(self, ax):
        # Ensure the policy and examples are available
        assert self.policy is not None
        assert self.given_examples is not None

        cmp = colormaps["Pastel1"].colors
        # Change givens to a dict mapping from state to signal
        givens = {state: signal for state, signal in self.given_examples}
        
        # Draw rectangles for all states based on the policy
        for i in range(self.states):
            signal = self.policy[i]
            if signal == 0:
                color = "white"
            else:
                color = cmp[signal-1]
            
            rect = patches.Rectangle((i, 0), 1, 1, linewidth=1, edgecolor="black", facecolor=color)
            ax.add_patch(rect)
            
        # Plot numbers at the sampled positions
        for i in givens:
            # Plot the actual signal number at the sampled position
            signal = givens[i]
            # Move the numbers down a little bit for readability
            ax.text(i + 0.5, -0.2, str(signal), ha='center', va='center', fontsize=8, color='red')
        
        # Adjust plot limits to show the numbers
        ax.set_xlim(0, self.states)
        ax.set_ylim(-0.5, 1)
        ax.set_yticks([])

        return ax

    def get_predictions(self):
        predictions = np.zeros((self.states, self.signals))
        for state in range(self.states):
            neighbors = self.get_sorted_neighbors(self.given_examples, state)[:self.trained_neigbors]
            cnt = np.asarray([0] * self.signals)
            for ne in neighbors:
                if ne[1] - 1 < self.signals:
                    cnt[ne[1]-1] += 1
            probs = cnt / self.trained_neigbors
            predictions[state] = probs
        return predictions

    def graph_preds(self, filename=None):
        assert self.given_examples is not None

        predictions = self.get_predictions()
        givens = [state for state, _ in self.given_examples]
        colors = [signal for _, signal in self.given_examples]
        givens_y = [0.2] * len(givens)
        # create legend
        labels = [f"Signal {i+1}" for i in range(self.signals)]
        labels.append("Threshold")

        plt.figure(figsize=(7,5))
        for sig in range(self.signals):
            plt.plot(range(1, self.states+1), predictions[:, sig], label=labels[sig])
        plt.plot(range(1, self.states+1), [self.threshold] * self.states, label=labels[-1], linestyle='--')
        plt.scatter(givens, givens_y, marker="|", cmap="Set1", c=colors)
        plt.xlim(1, self.states)
        plt.ylim(0, 1)
        plt.xlabel("State")
        plt.ylabel("Confidence")
        plt.legend()
        plt.grid(True)

        if isinstance(filename, str):
            plt.savefig(filename)
            plt.clf()
        else:
            plt.show()
        plt.close()
        gc.collect()


class LinearFunctionPlayer(Player):
    def __init__(self, *args, **kwargs):
        Player.__init__(self, *args, **kwargs)
        self.functions = None

    def learn(self, examples, threshold=0.5):
        # learn bounds from examples
        bounds = {0: (-1,-1)}
        for signal in range(1, self.signals+1):
            low = inf
            high = -inf
            for ex in examples:
                state, sig = ex
                if sig == signal:
                    if low > state:
                        low = state
                    if high < state:
                        high = state
            bounds[signal] = (low, high)
        bounds[self.signals+1] = (self.states, self.states)

        # learn linear piecewise functions from bounds
        functions = {}
        for signal in range(1, self.signals+1):
            l = bounds[signal][0]
            h = bounds[signal][1]
            s_l = bounds[signal-1][1]
            s_h = bounds[signal+1][0]
            def wrapper(low, high, super_low, super_high):
                def signal_prob(state):
                    if state < low:
                        # function between low and super low
                        # points: (low, 1), (super_low, 0)
                        if super_low == -1:
                            return 1
                        slope = 1.0 / (low - super_low)
                        intercept = -(slope * low - 1)
                        return max((slope * state) + intercept, 0)
                    elif state > high:
                        # function between high and super
                        # points: (high, 1), (super_high, 0)
                        if super_high == self.states:
                            return 1 
                        slope = -1.0 / (super_high - high)
                        intercept = -(slope * high - 1)
                        return max((slope * state) + intercept, 0)
                    else:
                        return 1
                return signal_prob
            functions[signal] = wrapper(l, h, s_l, s_h)

        # predict 
        working_policy = np.zeros(self.states, dtype=np.int64)
        for state in range(self.states):
            pred, prob = self.predict(state, functions)
            if prob >= threshold:
                working_policy[state] = pred

        self.policy = working_policy
        self.functions = functions
        self.threshold = threshold
        self.given_examples = examples


    def predict(self, state, functions):
        predictions = np.zeros(self.signals, dtype=np.float64)
        for signal in range(self.signals):
            confidence = functions[signal+1](state)
            predictions[signal] = confidence
        best_choice = np.argmax(predictions)
        final_prediction = best_choice + 1
        final_confidence = predictions[best_choice]
        return final_prediction, final_confidence

    def get_predictions(self):
        predictions = np.zeros((self.states, self.signals))
        for state in range(self.states):
            preds = np.zeros(self.signals, dtype=np.float64)
            for signal in range(self.signals):
                confidence = self.functions[signal+1](state)
                preds[signal] = confidence
            predictions[state] = preds
        return predictions
    
    def graph_preds(self, filename=None):
        assert self.functions is not None

        predictions = self.get_predictions()
        givens = [state for state, _ in self.given_examples]
        colors = [signal for _, signal in self.given_examples]
        givens_y = [0.2] * len(givens)
        

        # create legend
        labels = [f"Signal {i+1}" for i in range(self.signals)]
        labels.append("Threshold")

        plt.figure(figsize=(7,5))
        for sig in range(self.signals):
            plt.plot(range(1, self.states+1), predictions[:, sig], label=labels[sig])
        plt.plot(range(1, self.states+1), [self.threshold] * self.states, label=labels[-1], linestyle='--')
        plt.scatter(givens, givens_y, marker="|", cmap="Set1", c=colors)
        plt.xlim(0, self.states + 1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel("State")
        plt.ylabel("Confidence")
        plt.legend()
        plt.grid(True)

        if isinstance(filename, str):
            plt.savefig(filename)
            plt.clf()
        else:
            plt.show()
        plt.close()
        gc.collect()


class SigmoidPlayer(LinearFunctionPlayer):
    def learn(self, examples, threshold):
        # learn bounds from examples
        bounds = {0: (-1,-1)}
        for signal in range(1, self.signals+1):
            low = inf
            high = -inf
            for ex in examples:
                state, sig = ex
                if sig == signal:
                    if low > state:
                        low = state
                    if high < state:
                        high = state
            bounds[signal] = (low, high)
        bounds[self.signals+1] = (self.states, self.states)

        # learn sigmoid piecewise functions from bounds
        functions = {}
        for signal in range(1, self.signals+1):
            l = bounds[signal][0]
            h = bounds[signal][1]
            s_l = bounds[signal-1][1]
            s_h = bounds[signal+1][0]
            def wrapper(low, high, super_low, super_high):
                def signal_prob(state):
                    if state < low:
                        # function between low and super low
                        # points: (low, 1), (super_low, 0)
                        if super_low == -1:
                            return 1
                        a = 8 / (low - super_low)
                        b = (low - ((low - super_low) // 2)) * a
                        try:
                            return 1 / (1 + exp((-a * state) + b))
                        except OverflowError:
                            return 0
                    elif state > high:
                        # function between high and super
                        # points: (high, 1), (super_high, 0)
                        if super_high == self.states:
                            return 1
                        a = 8 / (super_high - high)
                        b = (super_high - ((super_high - high) // 2)) * a
                        try:
                            return 1 / (1 + exp((a * state) - b))
                        except OverflowError:
                            return 0
                    else:
                        return 1
                return signal_prob
            functions[signal] = wrapper(l, h, s_l, s_h)

        # predict 
        working_policy = np.zeros(self.states, dtype=np.int64)
        for state in range(self.states):
            pred, prob = self.predict(state, functions)
            if prob >= threshold:
                working_policy[state] = pred

        self.policy = working_policy
        self.functions = functions
        self.threshold = threshold
        self.given_examples = examples


class StrictPlayer(LinearFunctionPlayer):
    def learn(self, examples):
        # learn bounds
        bounds = {}
        for signal in range(1, self.signals+1):
            low = inf
            high = -inf
            for ex in examples:
                state, sig = ex
                if sig == signal:
                    if low > state:
                        low = state
                    if high < state:
                        high = state
            bounds[signal] = (low, high)
        
        # predict from bounds
        working_policy = np.zeros(self.states, dtype=np.int64)
        for state in range(self.states):
            pred = self.predict(state, bounds)
            working_policy[state] = pred

        self.policy = working_policy
        self.functions = bounds
        self.given_examples = examples

    def predict(self, state, bounds):
        if self.signals > 2:
            for signal in bounds:
                if state >= bounds[signal][0] and state <= bounds[signal][1]:
                    return signal
            return 0 
        else:
            if state <= bounds[1][1]:
                return 1
            elif state >= bounds[2][0]:
                return 2
            return 0 

    def get_predictions(self):
        predictions = np.zeros((self.states, self.signals))
        for state in range(self.states):
            strict_pred = self.predict(state, self.functions)
            prediction_vector = np.array([0, 0])
            if strict_pred != 0:
                prediction_vector[strict_pred-1] = 1
            predictions[state] = prediction_vector
        return predictions



"""
Intermediate class for Players using an SK-Learn algorithm with an underlying classifier.
"""
class SKLearnPlayer(Player):
    def __init__(self, *args, **kwargs):
        Player.__init__(self, *args, **kwargs)
        self.classifier = None

    def predict(self, model, state: int):
        probabilities = model.predict_proba(np.array([[state]]))
        most_confident = np.argmax(probabilities[0])
        prediction = most_confident + 1
        confidence = probabilities[0, most_confident]
        return prediction, confidence
    
    def graph_preds(self, filename=None):
        assert self.classifier is not None

        predictions = np.zeros((self.states, self.signals))
        for state in range(self.states):
            preds = self.classifier.predict_proba(np.array([[state]]))
            predictions[state] = preds

        givens = [state for state, _ in self.given_examples]
        colors = [signal for _, signal in self.given_examples]
        givens_y = [0.2] * len(givens)
        

        # create legend
        labels = [f"Signal {i+1}" for i in range(self.signals)]
        labels.append("Threshold")

        plt.figure(figsize=(7,5))
        for sig in range(self.signals):
            plt.plot(range(1, self.states+1), predictions[:, sig], label=labels[sig])
        plt.plot(range(1, self.states+1), [self.threshold] * self.states, label=labels[-1], linestyle='--')
        plt.scatter(givens, givens_y, marker="|", cmap="Set1", c=colors)
        plt.xlim(1, self.states)
        plt.ylim(0, 1)
        plt.xlabel("State")
        plt.ylabel("Confidence")
        plt.legend()
        plt.grid(True)

        if isinstance(filename, str):
            plt.savefig(filename)
            plt.clf()
        else:
            plt.show()
        plt.close()
        gc.collect()


"""
SK-learn player implementing a multi-layer perceptron classifier. Hidden shape: (3,3)
"""
class MLPPlayer(SKLearnPlayer):
    @SuppressWarning
    def learn(self, examples, threshold=0.5):
        # train classifier
        clf = MLPClassifier(
            hidden_layer_sizes=(3,3), 
            random_state=1, 
            solver='lbfgs', 
            max_iter=300,
            alpha=0.1
            )
        # (state, signal) - e.g., (0, 1), (24, 1), (59, 2)
        X = np.asarray([x[0] for x in examples]).reshape(-1, 1)
        y = np.asarray([x[1] for x in examples])
        
        clf.fit(X, y)

        # predict states for new policy
        working_policy = np.zeros(self.states, dtype=np.int64)
        for i in range(self.states):
            pred, prob = self.predict(clf, i)
            if prob >= threshold:
                working_policy[i] = pred
                
        # set policy
        self.policy = working_policy
        self.classifier = clf
        self.threshold = threshold
        self.given_examples = examples


"""
SK-learn Player implementing Logistic Regression
"""
class LRPlayer(SKLearnPlayer):
    def learn(self, examples, threshold=0.5):
        # Format data
        X = np.asarray([x[0] for x in examples]).reshape(-1, 1)
        y = np.asarray([x[1] for x in examples])

        # train classifier
        clf = LogisticRegression()
        clf.fit(X, y)

        # predict policy
        working_policy = np.zeros(self.states, dtype=np.int64)
        for i in range(self.states):
            pred, prob = self.predict(clf, i)
            if prob >= threshold:
                working_policy[i] = pred

        # set policy
        self.policy = working_policy
        self.classifier = clf
        self.threshold = threshold
        self.given_examples = examples


"""
SK-Learn player implementing naive bayes
Doc: https://scikit-learn.org/1.5/modules/naive_bayes.html
'although naive Bayes is known as a decent classifier, it is known to be a bad estimator, 
so the probability outputs from predict_proba are not to be taken too seriously.'
"""
class NaiveBayesPlayer(SKLearnPlayer):
    def learn(self, examples, threshold=0.5):
        X = np.asarray([x[0] for x in examples]).reshape(-1,1)
        y = np.asarray([x[1] for x in examples])

        clf = GaussianNB()
        clf.fit(X, y)

        working_policy = np.zeros(self.states, dtype=np.int64)
        for i in range(self.states):
            pred, prob = self.predict(clf, i)
            if prob >= threshold:
                working_policy[i] = pred

        self.policy = working_policy
        self.classifier = clf
        self.threshold = threshold
        self.given_examples = examples

        # set_trace()


def show_history(player_stack, filename=None):
    num_plots = len(player_stack)
    num_states = len(player_stack[0].policy)
    fig_height = max(2, num_plots * 1.0)
    fig_width = max(10, num_states * 0.1)
    fig, axes = plt.subplots(
        nrows=num_plots,
        sharex=True,
        figsize=(fig_width, fig_height),
    )

    # Ensure axes is iterable
    if num_plots == 1:
        axes = [axes]

    for i, player in enumerate(player_stack):
        if player.policy is not None:
            axes[i] = player.plot_strategy(axes[i])
            axes[i].set_ylabel(f"Gen {i}", fontsize=12)
            axes[i].set_ylim(-0.5, 1)  # Ensure y-limits match plot_strategy
        else:
            axes[i].set_visible(False)

    plt.xlabel("States", fontsize=12)
    plt.subplots_adjust(hspace=0.3)  # Adjust vertical spacing
    if filename is not None:
        plt.savefig(filename, dpi=50, bbox_inches="tight", pad_inches=1)
        plt.clf()
    else: 
        plt.show()
    plt.close()
    gc.collect()


def plot_utility(utils, filename=None, subtitle=None):
    plt.figure(figsize=(7,5))
    plt.plot(utils, marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Utility")
    if subtitle is not None:
        plt.title(f"Utility Over Generations: {subtitle}")
    else:
        plt.title("Utility Over Generations")
    if filename is not None:
        plt.savefig(filename)
        plt.clf()
    else: 
        plt.show()
    plt.close()
    gc.collect()

def single_policy_utility(player_stack, policy):
    return [x.utility(policy) for x in player_stack if x.policy is not None]


def generation_utility(player_stack):  # Added: New function to calculate utility between consecutive generations
    utilities = []
    for i in range(1, len(player_stack)):
        prev_player = player_stack[i-1]
        curr_player = player_stack[i]
        
        # Skip if either policy is None
        if prev_player.policy is None or curr_player.policy is None:
            continue
        
        prev_policy = prev_player.policy

        util = curr_player.utility(prev_policy)

        utilities.append(util)
    return utilities


def analyze_signal_gaps(player):  # Added: New function to analyze signal gaps
    signals = defaultdict(list)
    for state in range(len(player.policy)):
        signal = player.policy[state]
        if signal > 0:
            signals[signal].append(state)

    results = {}
    sorted_signals = sorted(signals.keys())
    for i in range(len(sorted_signals) - 1):
        current_signal = sorted_signals[i]
        next_signal = sorted_signals[i + 1]
        max_signal_i = max(signals[current_signal]) if signals[current_signal] else -1
        min_signal_next = min(signals[next_signal]) if signals[next_signal] else len(player.policy)
        results[f"Signal {current_signal} to {next_signal}"] = (max_signal_i, min_signal_next)

    return results


def run_game(
    initial_policy: NDArray, signals: int, 
    states: int, generations: int, 
    samples: int, threshold: float, 
    strat: str = "random", 
    reward: Tuple[float, float, float] = [1.0,0.5,0.0],
    player: Type[Player] = LinearFunctionPlayer
):
    p0 = player(policy=initial_policy, n_signals=signals, n_states=states, reward_struct=reward)  
    p1 = player(n_signals=signals, n_states=states, reward_struct=reward)  
    player_stack = [p0, p1]

    # Run generations
    for _ in range(generations):
        learner = player_stack.pop()
        teacher = player_stack.pop()

        examples = teacher.poll(samples, sampling_strat=strat)
        if player == StrictPlayer:
            learner.learn(examples)
        elif player == Player:
            learner.learn(examples, 5, True, threshold)
        else:
            learner.learn(examples, threshold) 

        next_gen = player(n_signals=signals, n_states=states, reward_struct=reward) 
        player_stack.append(teacher)
        player_stack.append(learner)
        player_stack.append(next_gen)

    return player_stack

def predict_interpretability(examples: List, player: Type[Player], info: Tuple[int,int,int,float], n_children: int = 100):
    signals, states, samples, threshold = info
    child_list = []
    for ex in examples:
        child = player(n_signals=signals,n_states=states)
        if player == StrictPlayer:
            child.learn(ex)
        elif player == Player:
            child.learn(ex, 5, True, threshold)
        else:
            child.learn(ex, threshold)

        child_list.append(child)

    scores = []
    lo_score = inf
    lo_pair = (-1, -1)
    for i in range(n_children):
        for j in range(i+1, n_children):
            child_a = child_list[i]
            child_b = child_list[j]
            score = child_a.utility(child_b.policy)
            scores.append(score)

            if score < lo_score:
                lo_score = score
                lo_pair = (child_a, child_b)
    
    return (sum(scores) / len(scores), lo_score, lo_pair)


def get_sample_bounds(player: Player):
    assert player.given_examples is not None
    bounds = {}
    for signal in range(1, player.signals+1):
        low = inf
        high = -inf
        for ex in player.given_examples:
            state, sig = ex
            if sig == signal:
                if low > state:
                    low = state
                if high < state:
                    high = state
        bounds[signal] = (low, high)
    return bounds