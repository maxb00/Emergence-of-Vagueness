import matplotlib.pyplot as plt
import imageio.v2 as imageio
import seaborn as sns
import os

def gen_image(agents_state):
  fig, axs = plt.subplots(3, 1, figsize=(15, 6))

  sns.heatmap(agents_state[0], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, fmt=".1f", ax=axs[0])

  sns.heatmap(agents_state[1], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, fmt=".1f", ax=axs[1])

  sns.heatmap(agents_state[2], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, fmt=".1f", ax=axs[2])

  plt.savefig("image.png")
  plt.close(fig)