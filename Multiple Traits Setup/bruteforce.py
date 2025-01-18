def custom_round(x):
  if x == 1.5 or x == 0.5:
    return 1
  
  return round(x)

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

  return state_prob

def unnumerize(num_traits, num_states, action):
  ufaction = []
  while action > 0:
    # Not sure why this isn't .insert(0, ...)?
    ufaction.insert(0, action % num_states)
    action = action // num_states

  while len(ufaction) < num_traits:
    # Same comment as above about using .insert(0, ...)?
    ufaction.insert(0, 0)

  return ufaction

def payoff(state, act):

  l2dist = 0
  for s, a in zip(state, act):
    l2dist += abs(s - a)

  return 1 - 0.5 * l2dist

def eval(signal, action):
  state_prob = gen_state_prob(2, 3)

  p = 0
  for i in range(3):
    for j in range(3):
      state = (i, j)
      sig = signal[state]
      # act = unnumerize(2, 3, action[sig])
      act = action[sig]

      p += payoff(state, act) * state_prob[state]
  
  return p

def run():

  opt_sig = np.array([
    [0, 0, 0],
    [1, 2, 3],
    [1, 2, 3]
  ])

  update = 0
  max_p = 0
  max_sig = []
  max_act = []
  signal = np.array([[0] * 3] * 3)
  # action = np.array([0] * 4)
  endsig = False
  bucket_sum = np.array([[9, 9], [0, 0], [0, 0], [0, 0]])
  bucket_size = np.array([9, 0, 0, 0])
  endi = 1755
  while not endsig and endi > 0:
    for i in range(3):
      for j in range(3):
        if signal[2-i, 2-j] < 3:
          bucket_sum[signal[2-i, 2-j]][0] -= 2-i
          bucket_sum[signal[2-i, 2-j]][1] -= 2-j
          bucket_size[signal[2-i, 2-j]] -= 1
          signal[2-i, 2-j] += 1
          bucket_size[signal[2-i, 2-j]] += 1
          bucket_sum[signal[2-i, 2-j]][0] += 2-i
          bucket_sum[signal[2-i, 2-j]][1] += 2-j
          break
        else:
          bucket_sum[signal[2-i, 2-j]][0] -= 2-i
          bucket_sum[signal[2-i, 2-j]][1] -= 2-j
          bucket_size[signal[2-i, 2-j]] -= 1
          signal[2-i, 2-j] = 0
          bucket_size[signal[2-i, 2-j]] += 1
          bucket_sum[signal[2-i, 2-j]][0] += 2-i
          bucket_sum[signal[2-i, 2-j]][1] += 2-j
          if 2-i == 0 and 2-j == 0:
            endsig = True
            break
      else:
        continue
      break

    # print(bucket_sum)
    # print(bucket_size)
    
    action = np.array([(0, 0)] * 4)
    for i in range(4):
      if bucket_size[i] == 0:
        continue

      action[i] = (custom_round(bucket_sum[i][0]/bucket_size[i]), custom_round(bucket_sum[i][1]/bucket_size[i]))
    
    # print(action)
    
    # endact = False
    # while not endact:
    #   for i in range(4):
    #     if action[i] < 8:
    #       action[i] += 1
    #       break
    #     else:
    #       action[i] = 0
    #       if i == 3:
    #         endact = True
    #         break
      
    ep = eval(signal, action)
    if ep > max_p:
      max_p = ep
      max_sig = signal.copy()
      max_act = action
    # print(f"sig={signal}\nact={action}\nep={ep}\n")

    if (endi == 1 or np.array_equal(signal, opt_sig)):
      print("OPTIMALLLL")
      print(f"sig={signal}\nact={action}\nep={ep}\n")
      exit(0)
    
    if update % 1000 == 0:
      print(f"{update}: {max_p}\n  sig={max_sig}\n  act={max_act}\n")
    update += 1

    # endi -= 1
  
  print(f"max_p={max_p}")

run()