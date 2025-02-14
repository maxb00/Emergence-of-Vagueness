# how many steps does it take to reach convergence?
# try a couple values of d.
# reward = c - (d * dist)
# the lower d is (in relation to c), the more states
# recieve a positive reward.
# main thing to track: when and where are null signals sent?
from signaling_game import SignalingGame
import pdb
spreads = [0.75, 0.6, 0.5, 0.3, 0.25, 0.1, 0.01]

# constants
states = 20
signals = 2
actions = 20
reward_magnitude = 1 # c
iterations = 40_000

game = SignalingGame(
    states, signals, actions, 
    (reward_magnitude, spreads[3]), True
)

game(iterations, 100)

pdb.set_trace()