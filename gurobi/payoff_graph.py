import matplotlib.pyplot as plt
import os
from pathlib import Path
import matplotlib
matplotlib.use('TKAgg')

def read_files(folder):
    path_generator = sorted(Path(folder).glob('*.txt'), key=os.path.getmtime)
    payoffs = []
    for path_obj in path_generator:
        path_str = str(path_obj)
        payoff = 0
        with open(path_str, 'r') as f:
            while True:
                line = f.readline().strip()
                if line.startswith("Objective ="):
                    payoff = float(line.split(" ")[2])
                    break
        payoffs.append(payoff)
    return payoffs

def main():
    payoffs = read_files(os.getcwd())

    plt.plot(payoffs)
    plt.title("3 traits, 5 expressions per trait")
    plt.xlabel("Number of Signals (k)")
    plt.ylabel("Expected Payoff")
    plt.ylim(-0.5,1.1)
    plt.grid(True)
    plt.savefig("3-5-payoff.jpg")

if __name__ == "__main__":
    main()