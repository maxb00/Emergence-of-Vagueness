import november_lib as lib
import numpy as np
from tqdm import tqdm

repeats = 100
generations = 100
states = 1000
signals = 2
samples = 200
thresholds = [i/100 for i in range(50, 100, 5)]
players = [lib.SigmoidPlayer, lib.LRPlayer, lib.MLPPlayer, lib.Player, lib.LinearFunctionPlayer]

initial_policy = np.zeros(states, dtype=np.int64)
step = states // signals
for i in range(states):
    initial_policy[i] = (i // step) + 1

# create my results csv
results_file = open("results_vagueness.csv", "w+")
# print headers
print("algorithm,n_generations,n_states,n_signals,n_samples,threshold,1(g)max,2(g)min,V(g)min,V(g)max,dist", file=results_file)

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
            # get the gaps
            sample_bounds = lib.get_sample_bounds(final_player_stack[-2])
            bound_vals = list(sample_bounds.values())
            one_g_max = bound_vals[0][1] # highest sample
            two_g_min = bound_vals[1][0] # lowest sample
            v_g_min, v_g_max = list(gaps.values())[0]
            dist = v_g_max - v_g_min
            # write to results csv
            result = f"{p.__name__},{generations},{states},{signals},{samples},{t},{one_g_max},{two_g_min},{v_g_min},{v_g_max},{dist}"
            print(result, file=results_file)

results_file.flush()
results_file.close()