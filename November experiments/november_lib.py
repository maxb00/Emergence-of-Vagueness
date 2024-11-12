import numpy as np
from numpy.typing import NDArray
from warnings import warn, catch_warnings
from collections import Counter, defaultdict
from random import sample
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.patches as patches
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from math import inf, exp
from pdb import set_trace

def SuppressWarning(func):
    def wrapper(*args):
        with catch_warnings(action='ignore'):
            func(*args)
    return wrapper

"""
Basic Player. Learning algorithm: K-Nearest Neighbors
"""
class Player:
    def __init__(self, policy: NDArray = None, n_signals: int = 2, n_states: int = 100):
        self.policy = policy
        self.signals = n_signals
        self.states = n_states
        if policy is not None:
            self.given_examples = []
        else:
            self.given_examples = None

    def poll(self, n_samples: int = -1, sampling_strat="random"):
        # randomly polls policy across all "certain" states
        assert self.policy is not None

        # iron out how many samples to take
        if n_samples == -1:
            # try to take 20%
            n_samples = len(self.policy) // 5
            # if 20% can't possible sample each bucket, try.
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
                if group_size < smallest_group:
                    smallest_group = group_size
            
            if smallest_group < per_group:
                per_group = smallest_group

            examples = []
            for group in groups:
                examples.extend(sample(groups[group], per_group))

            return examples

    
    def learn(self, examples, k_neighbors):
        # learns policy from examples
        # ideally, we extend Player and override this method 
        # for each learning algorithm we want to implement
        if self.policy is not None:
            warn("This player already has a policy! Be aware, this will overwrite it.")

        # do the learning
        working_policy = np.zeros(self.states, dtype=np.int64)
        n_examples = len(examples)
        for state in range(self.states):
            k_neighbors = k_neighbors if n_examples > k_neighbors else n_examples
            working_policy[state] = self.predict(examples, state, k_neighbors)

        self.policy = working_policy
        self.given_examples = examples


    def predict(self, examples, state, k_neighbors=5):
        # I'm going to implement a basic KNN
        distance = []
        for point_group in examples:
            point, group = point_group
            if isinstance(point, np.int64) or isinstance(point, int):
                dist = abs(state-point)
            else:
                # implement distance function for cartesian points
                dist = 0.5
            distance.append((dist, group))
        neighbors = sorted(distance)[:k_neighbors]
        c = Counter()
        for n in neighbors:
            c[n[1]] += 1
        return c.most_common(1)[0][0]
    

    def utility(self, init_policy):
        # compares (self) player's policy to given policy
        # returns utility score as ratio (0-1)
        assert self.policy is not None
        
        score = 0
        for state in range(self.states):
            if self.policy[state] == init_policy[state]:
                score += 1

        return score / self.states
    

    def plot_strategy(self, ax):
        # given an axis, plot our strategy
        assert self.policy is not None
        assert self.given_examples is not None

        cmp = colormaps["Pastel1"].colors
        givens = set([i for i, _ in self.given_examples])
        
        for i in range(self.states):
            signal = self.policy[i]
            if signal == 0:
                color = "white"
            else:
                color = cmp[signal-1]
            
            if i in givens:
                rect = patches.Rectangle((i, 0), 1, 1, linewidth=1, edgecolor="black", facecolor="plum")
            else:
                rect = patches.Rectangle((i, 0), 1, 1, linewidth=1, edgecolor="black", facecolor=color)
                
            ax.add_patch(rect)

        ax.set_xlim(0, self.states)
        ax.set_ylim(0, 1)
        ax.set_yticks([])

        return ax
    

class LinearFunctionPlayer(Player):
    def __init__(self, *args, **kwargs):
        Player.__init__(self, *args, **kwargs)
        self.functions = None
        self.threshold = 0.5

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
                        slope = 1.0 / (low - super_low)
                        intercept = -(slope*low - 1)
                        return max((slope*state) + intercept, 0)
                    elif state > high:
                        # function between high and super
                        # points: (high, 1), (super_high, 0) 
                        slope = -1.0 / (super_high - high)
                        intercept = -(slope*high - 1)
                        return max((slope*state) + intercept, 0)
                    else:
                        return 1
                return signal_prob
            functions[signal] = wrapper(l,h,s_l,s_h)

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
        # set_trace()
        final_prediction = best_choice + 1
        final_confidence = predictions[best_choice]
        return final_prediction, final_confidence
    

    def graph_preds(self, filename=None):
        assert self.functions is not None

        predictions = np.zeros((self.states, self.signals))
        for state in range(self.states):
            preds = np.zeros(self.signals, dtype=np.float64)
            for signal in range(self.signals):
                confidence = self.functions[signal+1](state)
                preds[signal] = confidence
            predictions[state] = preds

        # create legend
        labels = [f"Signal {i+1}" for i in range(self.signals)]
        labels.append("Threshold")

        plt.plot(range(1, self.states+1), predictions)
        plt.plot(range(1, self.states+1), [self.threshold] * self.states)
        plt.xlim(0, 101)
        plt.ylim(-0.1, 1.1)
        plt.xlabel("State")
        plt.ylabel("Confidence")
        plt.legend(labels=labels)
        plt.grid(True)

        if isinstance(filename, str):
            plt.savefig(filename)
            plt.clf()
        else:
            plt.show()


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
                        a = 8 / (low-super_low)
                        b = (low - ((low - super_low) // 2)) * a
                        return 1 / (1 + exp((-a * state)+b))
                    elif state > high:
                        # function between high and super
                        # points: (high, 1), (super_high, 0) 
                        a = 8 / (super_high - high)
                        b = (super_high - ((super_high - high) // 2)) * a
                        return 1 / (1 + exp((a*state)-b))
                    else:
                        return 1
                return signal_prob
            functions[signal] = wrapper(l,h,s_l,s_h)

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

"""
Intermediate class for Players using an SK-Learn algorithm with an underlying classifier.
"""
class SKLearnPlayer(Player):
    def __init__(self, *args, **kwargs):
        Player.__init__(self, *args, **kwargs)
        self.classifier = None
        self.threshold = 0.5

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

        # create legend
        labels = [f"Signal {i+1}" for i in range(self.signals)]
        labels.append("Threshold")

        plt.plot(range(1, self.states+1), predictions)
        plt.plot(range(1, self.states+1), [self.threshold] * self.states)
        plt.xlim(1, 100)
        plt.ylim(0, 1)
        plt.xlabel("State")
        plt.ylabel("Confidence")
        plt.legend(labels=labels)
        plt.grid(True)

        if isinstance(filename, str):
            plt.savefig(filename)
            plt.clf()
        else:
            plt.show()


"""
SK-learn player implementing a multi-layer perceptron classifier. Hidden shape: (3,3)
"""
class MLPPlayer(SKLearnPlayer):
    @SuppressWarning
    def learn(self, examples, threshold=0.5):
        # train classifier
        clf = MLPClassifier((3,3), max_iter=10000)
        # (state, signal) - (0, 1), (24, 1), (59, 2)
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


def show_history(player_stack, filename=None):
    _, axes = plt.subplots(nrows=len(player_stack)-1, sharex=True,
                           figsize=(15, len(player_stack) * 0.5))

    for i, player in enumerate(player_stack[:-1]):
        axes[i] = player.plot_strategy(axes[i])
        axes[i].set_ylabel(i)

    plt.xlabel("States")
    if filename is not None:
        plt.savefig(filename)
        plt.clf()
    else: 
        plt.show()


def plot_utility(player_stack, policy, filename=None):
    player_stack = player_stack[:-1]
    utils = [x.utility(policy) for x in player_stack]
    plt.plot(utils)
    plt.xlabel("Generation")
    plt.ylabel("Utility")
    if filename is not None:
        plt.savefig(filename)
        plt.clf()
    else: plt.show()