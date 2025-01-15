import november_lib as lib
import numpy as np
from tqdm import tqdm

repeats = 100
generations = 100
states = 1000
signals = 2
samples = 200
thresholds = [i/100 for i in range(50, 100, 5)]
players = [lib.Player, lib.LinearFunctionPlayer, lib.SigmoidPlayer, lib.LRPlayer, lib.MLPPlayer]

initial_policy = np.zeros(states, dtype=np.int64)
step = states // signals
for i in range(states):
    initial_policy[i] = (i // step) + 1

# create my results csv
results_file = open("results.csv", "w+")
# print headers
print("algorithm,n_generations,n_states,n_signals,n_samples,threshold,util_gen2gen,util_gen2ground", file=results_file)

# one just for strict
for _ in tqdm(range(repeats), desc="Strict"):
    final_player_stack = lib.run_game(initial_policy, signals, states, generations, samples, 1.0, player=lib.StrictPlayer)
     # get the averages
    gen2gen = np.mean(lib.generation_utility(final_player_stack[:-1]))
    gen2ground = np.mean(lib.single_policy_utility(final_player_stack[:-1], initial_policy))
    # write to results csv
    result = f"Strict,{generations},{states},{signals},{samples},1.0,{gen2gen},{gen2ground}"
    print(result, file=results_file)


# for everyone else
for t in tqdm(thresholds, desc="Thresholds"):
    for p in tqdm(players, desc="Algorithm"):
        for _ in tqdm(range(repeats), desc="Repeats"):
            if p == lib.Player and t <= 0.6:
                continue
            # run the game
            try:
                final_player_stack = lib.run_game(initial_policy, signals, states, generations, samples, t, player=p)
            except ValueError as e:
                # we probably went extinct
                print(f"Extinction check - {p}, {t}")
                continue
            gaps = lib.analyze_signal_gaps(final_player_stack[-2])
            if len(gaps) == 0:
                # we probably went extinct
                print(f"Extinction check - {p}, {t}")
                continue
            # get the averages
            gen2gen = np.mean(lib.generation_utility(final_player_stack[:-1]))
            gen2ground = np.mean(lib.single_policy_utility(final_player_stack[:-1], initial_policy))
            # write to results csv
            result = f"{p.__name__},{generations},{states},{signals},{samples},{t},{gen2gen},{gen2ground}"
            print(result, file=results_file)

results_file.flush()
results_file.close()