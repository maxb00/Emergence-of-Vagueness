import gurobipy as gp
from gurobipy import GRB
import tomllib as tml
import numpy as np
import display_helper as dh
import datetime
import argparse

parser = argparse.ArgumentParser(
    prog="Gurobi payoff solve automation",
    description="Calculates all payoffs between 1 and n.",
    epilog="Contact barlowm15@gmail.com with questions."
)

parser.add_argument("t", type=int, help="number of traits")
parser.add_argument("n_exp", type=int, help="number of expressions per trait")


# Functions for probability matrix
def normpdf(x: float, mean: float, std: float) -> float:
  var = float(std)**2
  denom = (2*np.pi*var)**.5
  num = np.exp(-(float(x)-float(mean))**2/(2*var))
  return num/denom


def highlow(ind):
  if ind == 0 or ind == 2:
    return 0.4
  else:
    return 0.2


def gen_state_prob(num_traits: int, num_states: int):
  mean = (num_states-1) / 2
  std = mean / 1.25

  state_prob = np.zeros(tuple([num_states] * num_traits), dtype=np.float64)
  for inds in np.ndindex(state_prob.shape):
    prob = 1
    for ind in inds:
      prob *= normpdf(ind, mean, std)
      # prob *= highlow(ind)
    
    state_prob[inds] = prob

  state_prob = state_prob / np.sum(state_prob)

  return state_prob.flatten()

# functions for reward matrix
def unnumerize(num_traits: int, num_states: int, action: int):
  ufaction = []
  while action > 0:
    ufaction.insert(0, action % num_states)
    action = action // num_states

  while len(ufaction) < num_traits:
    ufaction.insert(0, 0)

  return ufaction


def reward_fn(param: tuple[float, float], state, action):
  
  l1dist = 0
  for s, a in zip(state, action):
    l1dist += abs(s - a)
          
  return param[0] - param[1] * l1dist


def reward_matrix(num_traits: int, num_states: int, reward_param: tuple[float, float]):
  total_states = num_states**num_traits
  res = np.array([[0 for _ in range(total_states)] for _ in range(total_states)], dtype=np.float64)
  for x in range(total_states):
    for y in range(total_states):
      s1 = unnumerize(num_traits, num_states, x)
      s2 = unnumerize(num_traits, num_states, y)

      res[x, y] = reward_fn(reward_param, s1, s2)

  return res

# generate neighbor list (graph connections)
def is_adj(num_traits: int, num_states: int, i: int, j: int):
  s1 = unnumerize(num_traits, num_states, i)
  s2 = unnumerize(num_traits, num_states, j)

  l1dist = 0
  for s, a in zip(s1, s2):
    l1dist += abs(s - a)

  return l1dist == 1

def neighbor_lst(num_traits: int, num_states: int):
  total_states = num_states**num_traits
  res = [[] for _ in range(total_states)]
  for x in range(total_states):
    for y in range(x, total_states):
      if is_adj(num_traits, num_states, x, y):
        res[x].append(y)
        res[y].append(x)

  return res
  
def convert(num_traits, num_states, num_signals, strat):
  total_states = num_states**num_traits
  small_val = 1e-7
  large_val = 1 - (small_val * (num_signals-1))

  signal_strat = np.full((num_signals, total_states), small_val, dtype=np.float64)

  for i, bucket in enumerate(strat):
    for state in bucket:
      signal_strat[i, state] = large_val

  return signal_strat

def info_measure(num_traits, num_states, num_signals, signal_prob, state_prob, weighted=True) -> float:
    """Calculates the information content of the signals

    Args:
      signal_prob (np.ndarray): the probabilities of the signals
      weighted (boolean): weighted/unweighted options

    Returns:
      inf (float): total information content measure
      inf_sigs (list): information content by signal
      inf_states (list): information content by state
    """
    total_states = num_states**num_traits
    signal_prob = signal_prob.reshape(num_signals, total_states)

    prob = np.zeros_like(signal_prob)
    for i in range(num_signals):
      for j in range(total_states):
        prob[i, j] = signal_prob[i, j] * state_prob[j]
    prob_sig = [np.sum(prob[i]) for i in range(num_signals)]
    prob = (prob.T / np.sum(prob, axis=1)).T

    inf = 0
    inf_sigs = []
    inf_states = []
    for i in range(num_signals):
      inf_sig = 0
      inf_states.append([])
      for j in range(total_states):
        inf_state = prob[i, j] * np.log(prob[i, j]/state_prob[j])
        inf_sig += inf_state
        
        if weighted:
          inf_state = prob_sig[i] * inf_state

        inf_states[i].append(inf_state)

      if weighted:
        inf_sig = prob_sig[i] * inf_sig

      inf_sigs.append(inf_sig)
      inf += inf_sig

    new_size = [num_signals]
    new_size.extend([num_states] * num_traits)
    inf_states = np.resize(np.array(inf_states), tuple(new_size))

    return inf, inf_sigs, inf_states

def stats(inf, inf_sigstates, output_file):
  inf_states = np.sum(inf_sigstates, axis=0)

  print(f"Info measure = {inf}", file=output_file)
  print(f"Info measure by states:", file=output_file)

  with np.printoptions(precision=3, suppress=True, formatter={'float': '{:.3f}'.format}, linewidth=100):
    print(inf_states, file=output_file)

def defineX(model, V, k):
    x = model.addVars(V, V, vtype=GRB.BINARY, name="x")

    model.update()

    # /* there are exactly k buckets */
    # kBucketConstr: sum{j in V} x[j, j] = k;
    k_bucket = gp.quicksum( (x[j,j]) for j in V ) == k
    model.addConstr(k_bucket)

    # /* a state can only belong to one bucket */
    # uniqueBucketConstr{i in V}: sum{j in V} x[i, j] = 1;
    unique_bucket = ( gp.quicksum( (x[i,j]) for j in V ) == 1 for i in V )
    model.addConstrs(unique_bucket)

    # /* a state cannot belong to a non-existant bucket */
    # nonexBucketConstr{i in V, j in V}: x[i, j] <= x[j, j];
    nonex_bucket = ( (x[i,j] <= x[j,j]) for i in V for j in V )
    model.addConstrs(nonex_bucket)

    model.update()

    return x

def defineCutEdge(model, V, x, neighbor):
    # cut egde variables
    # y[i, j] = 1 iff edge {i, j} is cut
    y = model.addVars(V, V, vtype=GRB.BINARY)

    model.update()

    # /* cut edge constraints */
    # edge {i, j} is cut if i and j are not adjacent.
    cut_edge_not_adj = ( y[i, j] == 1 for i in V for j in np.setdiff1d(V, np.array(neighbor[i])) )
    model.addConstrs(cut_edge_not_adj)

    # edge {i, j} is cut if i and j are in different buckets.
    cut_edge_diff_bucket = ( y[i, j] >= x[i, l] - x[j, l] for i in V for j in V for l in V)
    model.addConstrs(cut_edge_diff_bucket)

    model.update()

    return y

def defineFlow(model, V, n, k, y, neighbor, x):
    # What does this mean?
    M = n - k + 1

    # flow variables
    # f[i, j] = the flow from state i to state j
    f = model.addVars(V, V)

    model.update()

    # do not send flow across cut edges
    cut_edge_flow = ( f[i, j] + f[j, i] <= M * (1 - y[i, j]) for i in V for j in V )
    model.addConstrs(cut_edge_flow)

    # /* flow constraint */
    # if not a root, consume some flow.
    # if a root, only send out (so much) flow.
    flow = ( gp.quicksum( f[j, i]- f[i, j] for j in neighbor[i] )
        >= 1 - M * x[i, i] for i in V )
    model.addConstrs(flow)

    model.update()

    return f

def main(args):
    # parameters
    t = args.t              # number of traits (?) (cube is n,n,n)
    n_per_t = args.n_exp    # number of expressions per trait  (?)
    n = n_per_t**t          # size of state space
    reward_param = (1, 0.5) # 1 - (0.5 * dist to correct)

    V = np.asarray([i for i in range(n)])

    state_prob = gen_state_prob(t, n_per_t)
    # state_prob = np.full(n, 1/n, dtype=np.float64)

    reward = reward_matrix(t, n_per_t, reward_param)

    neighbor = neighbor_lst(t, n_per_t)
    
    for k in range(1, n):
        # establish model (must close)
        model = gp.Model(env=env)

        timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")
        distrobution = "gaussian" # informal tag only
        output_filename = f"{t}_{n_per_t}_{k}_{distrobution}_{timestamp}"
        output_file = open(output_filename + ".txt", "a")
        
        print(f"traits: {t}\texpressions: {n_per_t}\tstates: {n}\tsignals: {k}", file=output_file)
        print(f"reward: {reward_param[0]}, {reward_param[1]}", file=output_file)

        x = defineX(model, V, k)
        y = defineCutEdge(model, V, x, neighbor)
        f = defineFlow(model, V, n, k, y, neighbor, x)

        objective = gp.quicksum( gp.quicksum( (state_prob[i] * x[i,j] * reward[i][j]) for j in V) for i in V )
        model.setObjective(objective, GRB.MAXIMIZE)

        model.Params.PoolSolutions = 1
        model.Params.PoolSearchMode = 2

        model.update()

        model.optimize()

        centers = [j for j in V if round(x[j,j].getAttr("x")) == 1]
        bucket_lookup = dict()

        for ind, j in enumerate(centers):
            print(f"Bucket {j+1}: ", end="", file=output_file)
            members = [i for i in V if round(x[i,j].getAttr("x")) == 1]
            for i in members:
                print(f"{i+1} ", end="", file=output_file)
                bucket_lookup[i] = ind
            print(file=output_file)

        # n solutions
        n_solutions = model.getAttr("SolCount")
        print(f"Number of solutions: {n_solutions}", file=output_file)

        new_size = [k]
        new_size.extend([n_per_t] * t)

        total_info = 0
        total_info_sigstates = np.zeros(tuple(new_size))
        total_w_info = 0
        total_w_info_sigstates = np.zeros(tuple(new_size))

        for sol in range(0, n_solutions):
            model.params.SolutionNumber = sol
            centers = [j for j in V if round(x[j,j].getAttr("Xn")) == 1]
            strat = []

            # # printing out the solution
            # print(f"Solution {sol+1}")

            for j in centers:
                members = [i for i in V if round(x[i,j].getAttr("Xn")) == 1]
                strat.append(members)

                # # printing out the buckets
                # print(f"Bucket {j+1}: ", end="")
                # for i in members:
                #     print(f"{i+1} ", end="")
                # print()

            converted_strat = convert(t, n_per_t, k, strat)
                
            inf, inf_sigs, inf_sigstates = info_measure(t, n_per_t, k, converted_strat, state_prob, False)

            # # printing the information content
            # print(f"Info measure={inf}")

            w_inf, w_inf_sigs, w_inf_sigstates = info_measure(t, n_per_t, k, converted_strat, state_prob)

            total_info += inf
            total_info_sigstates += inf_sigstates

            total_w_info += w_inf
            total_w_info_sigstates += w_inf_sigstates

            
        avg_info = total_info / n_solutions
        avg_info_sigstates = total_info_sigstates / n_solutions
        avg_w_info = total_w_info / n_solutions
        avg_w_info_sigstates = total_w_info_sigstates / n_solutions

        # printing the objective
        print(f"Objective = {model.ObjVal}\n", file=output_file)

        # printing the payoff of the first solution
        op = 0
        for i in V:
            for j in V:
                op += state_prob[i] * x[i,j].getAttr("X") * reward[i][j]
        print(f"Payoff = {op / n}", file=output_file)

        # printing the average information content
        print("UNWEIGHTED", file=output_file)
        stats(avg_info, avg_info_sigstates, output_file)
        print(file=output_file)
        print("WEIGHTED", file=output_file)
        stats(avg_w_info, avg_w_info_sigstates, output_file)

        # save cube
        og_cube, legend = dh.cube_from_lookup(bucket_lookup, n_per_t, centers)
        dh.show(og_cube, legend, True, output_filename + ".jpg")

        model.close()
        output_file.flush()
        output_file.close()

    return 0

if __name__ == "__main__":
    # get gurobi credentials
    options = tml.load(open("../../license.toml", "rb"))

    # establish env (must close)
    env = gp.Env(params=options)

    args = parser.parse_args()

    main(args)

    env.close()