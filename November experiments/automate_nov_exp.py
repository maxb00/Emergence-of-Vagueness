import november_lib as lib
import argparse
import numpy as np
import datetime
from pdb import set_trace

parser = argparse.ArgumentParser(
    prog="Generational learning with imperfect data from the world.",
    description="-----",
    epilog="Contact barlowm15@gmail.com with questions."
)

parser.add_argument("gens", type=int, help="number of generations")
parser.add_argument("signals", type=int, help="number of signals")
parser.add_argument("states", type=int, help="number of states")
parser.add_argument("threshold", type=float, help="null signal confidence threshold")
parser.add_argument("samples", type=int, help="number of samples per generation")
parser.add_argument("--sampling_strat", default="random", help="sampling strategy. [random, uniform]")


def main(args):
    generations = args.gens
    signals = args.signals
    states = args.states
    threshold = args.threshold
    strat = args.sampling_strat
    samples = args.samples
    
    assert strat in ["random", "uniform"]

    inital_policy = np.zeros(states, dtype=np.int64)
    step = states // signals
    for i in range(states):
        inital_policy[i] = (i // step) + 1

    # This week: Investigate Naive Bayes Bernoilli 
    p0 = lib.MLPPlayer(policy=inital_policy, n_signals=signals, n_states=states)
    p1 = lib.MLPPlayer(n_signals=signals, n_states=states)
    test_player_stack = [p0, p1]
    for gen in range(generations):
        learner = test_player_stack.pop()
        teacher = test_player_stack.pop()

        examples = teacher.poll(samples, sampling_strat=strat)
        learner.learn(examples, threshold)

        next_gen = lib.MLPPlayer(n_signals=signals, n_states=states)
        test_player_stack.append(teacher)
        test_player_stack.append(learner)
        test_player_stack.append(next_gen)

        # set_trace()

        print(f"Finished gen {gen}")

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")
    blurb = f"{generations}_{signals}_{states}_{threshold}_{samples}_{strat}_{timestamp}"
    history_filename = "history_" + blurb + ".jpg"
    utility_filename = "utility_" + blurb + ".jpg"

    lib.show_history(test_player_stack, history_filename)
    lib.plot_utility(test_player_stack, inital_policy, utility_filename)

    for i in range(1, generations):
        predictions_filename = f"gen{i}-preds" + blurb + ".jpg"
        test_player_stack[i].graph_preds(predictions_filename)

    f = open("examples.txt", "w")

    for i, player in enumerate(test_player_stack):
        f.write(f"Generation {i}:")
        if player.given_examples is None:
            continue
        for ex in player.given_examples:
            f.write(f"({ex[0]}, {ex[1]}), ")
        f.write("\n")
    f.flush()
    f.close()
    
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)