from argparse import ArgumentParser
from signaling_game import SignalingGame
from agents import Sender, Receiver

def get_parser():
  parser = ArgumentParser()
  parser.add_argument("num_states", type=int, help="number of states")
  parser.add_argument("num_signals", type=int, help="number of signals")
  parser.add_argument("num_actions", type=int, help="number of actions")
  parser.add_argument("reward_param_1", type=float, 
                      help="first parameter for reward function")
  parser.add_argument("reward_param_2", type=float, 
                      help="second parameter for reward function")
  parser.add_argument("-n", "--null", help="null signal", action="store_true")
  parser.add_argument("num_iter", type=int, 
                      help="number of iterations for simulation")
  parser.add_argument("-r", "--record", type=int, default=-1, help="record interval")

  return parser

def main():
  parser = get_parser()
  args = parser.parse_args()

  game = SignalingGame(args.num_states, args.num_signals, args.num_actions, (args.reward_param_1, args.reward_param_2), null_signal=args.null)
  game(args.num_iter, args.record)

if __name__ == '__main__':
  main()