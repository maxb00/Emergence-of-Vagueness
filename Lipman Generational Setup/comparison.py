import pandas as pd
import matplotlib.pyplot as plt

file_path = 'results.csv'  
data = pd.read_csv(file_path)

average_results = data.groupby(['threshold', 'algorithm'])[['util_gen2gen', 'util_gen2ground']].mean().reset_index()

# Map new algorithm display names
algorithm_display_names = {
    'LRPlayer': 'Logistic Regression',
    'LinearFunctionPlayer': 'Linear',
    'MLPPlayer': 'MLP',
    'Player': 'KNN',
    'SigmoidPlayer': 'Sigmoid',
    'Strict': 'Strict'
}

# Replace algorithm names for display purposes
average_results['algorithm'] = average_results['algorithm'].map(algorithm_display_names)

# Pivot the data for plotting
pivoted_gen2gen = average_results.pivot(index='threshold', columns='algorithm', values='util_gen2gen')
pivoted_gen2ground = average_results.pivot(index='threshold', columns='algorithm', values='util_gen2ground')

# Plot the util_gen2gen graph
plt.figure(figsize=(10, 6))
for algorithm in pivoted_gen2gen.columns:
    plt.plot(pivoted_gen2gen.index, pivoted_gen2gen[algorithm], label=algorithm)

plt.title('Algorithm Comparison by Threshold (Gen-to-Gen)', fontsize=14)
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Average Utility (Gen-to-Gen)', fontsize=12)
plt.legend(title="Algorithm")
plt.grid(True)
plt.tight_layout()

plt.savefig('algorithm_comparison_gen2gen.png', dpi=300)
plt.show()

# Plot the util_gen2ground graph
plt.figure(figsize=(10, 6))
for algorithm in pivoted_gen2ground.columns:
    plt.plot(pivoted_gen2ground.index, pivoted_gen2ground[algorithm], label=algorithm)

plt.title('Algorithm Comparison by Threshold (Gen-to-Ground)', fontsize=14)
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Average Utility (Gen-to-Ground)', fontsize=12)
plt.legend(title="Algorithm")
plt.grid(True)
plt.tight_layout()

plt.savefig('algorithm_comparison_gen2ground.png', dpi=300)
plt.show()
