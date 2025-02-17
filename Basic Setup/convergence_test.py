# how many steps does it take to reach convergence?
# try a couple values of d.
# reward = c - (d * dist)
# the lower d is (in relation to c), the more states
# recieve a positive reward.
# main thing to track: when and where are null signals sent?
from signaling_game import SignalingGame
import pdb
from tqdm import tqdm
spreads = [0.75, 0.6, 0.5, 0.3, 0.25, 0.1, 0.01]

# constants
states = 20
signals = 2
actions = 20
reward_magnitude = 1 # c
iterations = 100_000
repeats = 5

for d in tqdm(spreads):
    for repeat in tqdm(range(repeats)):
        game = SignalingGame(
            states, signals, actions, 
            (reward_magnitude, d), True
        )

        game(iterations, 100, repeat)
