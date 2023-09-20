from argparse import ArgumentParser
import numpy as np
from signaling_game import SignalingGame
from agents import Agent
from display import gen_network_gif

def get_args():
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

  args = parser.parse_args()

  return (args.num_states, args.num_signals, args.num_actions, 
          args.reward_param_1, args.reward_param_2, args.null,
          args.num_iter, args.record)

def main(info=None):
  if info == None:
    (nstates, nsignals, nactions, rp1, rp2, null, niter, record) = get_args()
  else:
    (nstates, nsignals, nactions, rp1, rp2, null, niter, record) = info

  network_size = 5 # to update

  agent_lst1 = [Agent(nstates, nsignals, nactions, null) 
             for _ in range(network_size)]
  agent_lst2 = [Agent(nstates, nsignals, nactions, null) 
               for _ in range(network_size)]

  game = SignalingGame(nstates, nsignals, nactions, 
                       (rp1, rp2), null_signal=null)

  for iter in range(niter):
    random_senders = np.random.permutation(agent_lst1)
    random_receivers = np.random.permutation(agent_lst2)

    for i in range(network_size):
      if record > 0 and (iter+1) % record == 0:
        game(random_senders[i], random_receivers[i], 1, 1)
      else:
        game(random_senders[i], random_receivers[i], 1)

    random_senders = np.random.permutation(agent_lst2)
    random_receivers = np.random.permutation(agent_lst1)

    for i in range(network_size):
      if record > 0 and (iter+1) % record == 0:
        game(random_senders[i], random_receivers[i], 1, 1)
      else:
        game(random_senders[i], random_receivers[i], 1)
    
  for i in range(network_size):
    print(f"Agent {i} of list 1")
    print(agent_lst1[i].signal_weights)
    agent_lst1[i].print_signal_prob()
    print(agent_lst1[i].action_weights)
    agent_lst1[i].print_action_prob()
    print()
    print(f"Agent {i} of list 2")
    print(agent_lst2[i].signal_weights)
    agent_lst2[i].print_signal_prob()
    print(agent_lst2[i].action_weights)
    agent_lst2[i].print_action_prob()
    print()

  if record < 0:
    return
  
  gif_filename = f"./simulations/{nstates}_{nsignals}_{nactions}/{(rp1, rp2)}{'_null' if null else ''}_{niter}A.gif"

  gen_network_gif(agent_lst1, niter, record, 200, gif_filename)

  gif_filename = f"./simulations/{nstates}_{nsignals}_{nactions}/{(rp1, rp2)}{'_null' if null else ''}_{niter}B.gif"

  gen_network_gif(agent_lst2, niter, record, 200, gif_filename)

if __name__ == '__main__':
  main()