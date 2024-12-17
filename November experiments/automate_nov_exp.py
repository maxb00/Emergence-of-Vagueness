import numpy as np
import argparse
import datetime
import os
import november_lib as lib

# Argument Parser
parser = argparse.ArgumentParser(
    description="Automated November experiments"
)
parser.add_argument("gens", type=int, help="Number of generations")
parser.add_argument("signals", type=int, help="Number of signals")
parser.add_argument("states", type=int, help="Number of states")
parser.add_argument("threshold", type=float, help="Confidence threshold")
parser.add_argument("samples", type=int, help="Number of samples")
parser.add_argument("--sampling_strat", default="random", help="Sampling strategy")
parser.add_argument("--output_dir", default=".", help="Directory to save outputs")
parser.add_argument("--reward", nargs=3, type=float, help="Reward structure. Space-seperated floats ex. 1.0 0.5 -2.0", default=[1.0,0.5,0.0])
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Main function
def main(args):
    generations = args.gens
    signals = args.signals
    states = args.states
    threshold = args.threshold
    strat = args.sampling_strat
    samples = args.samples
    reward = args.reward

    assert strat in ["random", "uniform"], "Invalid sampling strategy"

    # Create the initial policy
    initial_policy = np.zeros(states, dtype=np.int64)
    step = states // signals
    for i in range(states):
        initial_policy[i] = (i // step) + 1

    # Initialize player stack
    p0 = lib.MLPPlayer(policy=initial_policy, n_signals=signals, n_states=states, reward_struct=reward)  
    p1 = lib.MLPPlayer(n_signals=signals, n_states=states, reward_struct=reward)  
    player_stack = [p0, p1]

    # Run generations
    for gen in range(generations):
        learner = player_stack.pop()
        teacher = player_stack.pop()

        examples = teacher.poll(samples, sampling_strat=strat)
        learner.learn(examples, threshold)  

        next_gen = lib.MLPPlayer(n_signals=signals, n_states=states, reward_struct=reward) 
        player_stack.append(teacher)
        player_stack.append(learner)
        player_stack.append(next_gen)

        print(f"Finished generation {gen + 1}")  

    # Compute generation-to-generation utilities 
    gen_utils = lib.generation_utility(player_stack[:-1])  # Added: Exclude the last player without a policy
    static_utils = lib.single_policy_utility(player_stack[:-1], initial_policy)
    print("Generation-to-generation utilities:", gen_utils)  # Added: Display utilities

    # Analyze signal gaps for each generation
    for i, player in enumerate(player_stack[:-1]):  # Added: Exclude the last player without a policy
        gaps = lib.analyze_signal_gaps(player)  # Added: Analyze signal gaps
        print(f"Generation {i} signal gaps:", gaps)  # Added: Display gaps

    # Timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    blurb = f"{generations}_{signals}_{states}_{threshold}_{samples}_{strat}_{timestamp}"

    # Save history and utility plots in the output directory
    history_filename = os.path.join(args.output_dir, f"history_{blurb}.jpg")
    static_utility_filename = os.path.join(args.output_dir, f"utility_static_{blurb}.jpg")
    gens_utility_filename = os.path.join(args.output_dir, f"utility_gens_{blurb}.jpg")


    lib.show_history(player_stack[:-1], history_filename)  # Added: Exclude the last player without a policy
    lib.plot_utility(static_utils, static_utility_filename, "Static")  # Added: Exclude the last player without a policy
    lib.plot_utility(gen_utils, gens_utility_filename, "Gen-to-gen")

    # Save generation predictions in the output directory
    for i in range(1, generations):
        predictions_filename = os.path.join(
            args.output_dir, f"gen{i}-preds_{blurb}.jpg"
        )
        player_stack[i].graph_preds(predictions_filename)

    # Threshold Optimization
    thresholds = np.arange(0.5, 1.0, 0.1)
    best_thresholds = {}
    for thresh in thresholds:
        temp_player_stack = [lib.MLPPlayer(policy=initial_policy.copy(), n_signals=signals, n_states=states)]
        temp_player_stack.append(lib.MLPPlayer(n_signals=signals, n_states=states))
        
        # Run generations with current threshold
        for gen in range(generations):
            learner = temp_player_stack.pop()
            teacher = temp_player_stack.pop()

            examples = teacher.poll(samples, sampling_strat=strat)
            learner.learn(examples, thresh)

            next_gen = lib.MLPPlayer(n_signals=signals, n_states=states)
            temp_player_stack.append(teacher)
            temp_player_stack.append(learner)
            temp_player_stack.append(next_gen)
        
        # Compute utility against initial policy (exclude the last player)
        utils = lib.single_policy_utility(temp_player_stack[:-1], initial_policy)
        # temp_gen_utils = lib.generation_utility(temp_player_stack[:-1])
        if utils:
            avg_utility = np.mean(utils)
            best_thresholds[thresh] = avg_utility
        else:
            best_thresholds[thresh] = 0  # Assign zero utility if no valid players

    # Determine the best threshold
    best_threshold = max(best_thresholds, key=best_thresholds.get)
    print(f"Best threshold: {best_threshold}, Average utility: {best_thresholds[best_threshold]}")

    # save the threshold optimization results
    threshold_filename = os.path.join(args.output_dir, f"threshold_optimization_{blurb}.txt")
    with open(threshold_filename, 'w') as f:
        for thresh, util in best_thresholds.items():
            f.write(f"Threshold: {thresh}, Average Utility: {util}\n")
    print(f"Threshold optimization results saved in {threshold_filename}")

    print(f"Results saved in {args.output_dir}")

if __name__ == "__main__":
    main(args)
