import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import seaborn as sns
import os

def gen_gif(signal_history: list, action_history: list, ep_fn, num_iter: int, record_interval: int, duration: int, output_file: str):
  """Generates a heatmap gif of the whole simulation and saves it into 
  test.gif

  Args:
    num_images (int): the number of images in the gif
    record_interval (int): number of simulations between each image
    duration (int): the duration an image is shown in the gif
  """
  num_images = num_iter // record_interval

  if not os.path.exists("./images"):
    os.mkdir("images")

  epx = []
  epy = []

  for i in range(num_images):
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    plt.tight_layout(pad=3)

    epx.append((i+1)*record_interval)
    epy.append(ep_fn(signal_history[i], action_history[i]))

    sns.heatmap(signal_history[i], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, 
    fmt=".1f", ax=axs[0])
    axs[0].set_xlabel("states")
    axs[0].set_ylabel("messages")
    axs[0].set_title("Sender\'s weights")

    sns.heatmap(action_history[i], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, 
    fmt=".1f", ax=axs[1])
    axs[1].set_xlabel("actions")
    axs[1].set_ylabel("messages")
    axs[1].set_title("Receiver\'s weights")

    axs[2].plot(epx, epy)
    axs[2].set_xlabel("rollout")
    axs[2].set_ylabel("expected payoff")
    axs[2].set_title("Expected payoff by rollout")

    fig.suptitle(f"Rollout {(i+1)*record_interval}")
    plt.savefig(f"./images/game_{(i+1)*record_interval}.png")
    plt.close(fig)

  images = []
  for filename in [f"./images/game_{(j+1)*record_interval}.png" for j in range(num_images)]:
    images.append(imageio.imread(filename))

  if not os.path.exists("./simulations"):
    os.mkdir("simulations")
  
  subfolder = f"{len(signal_history[0][0])}_{len(signal_history[0])}_{len(action_history[0][0])}"
  if not os.path.exists(f"./simulations/{subfolder}"):
    os.makedirs(f"simulations/{subfolder}/")

  imageio.mimsave(output_file, images, duration=duration)

# f"./simulations/{self.num_states}_{self.num_signals}_{self.num_actions}/{self.reward_param}{'_null' if self.null_signal else ''}_{num_iter}.gif"

def gen_network_gif(agents: list, num_iter: int, record_interval: int, duration: int, output_file: str):

  num_images = num_iter // record_interval

  if not os.path.exists("./images"):
    os.mkdir("images")

  for i in range(num_images):
    fig, axs = plt.subplots(len(agents), 2, figsize=(17, 10))
    plt.tight_layout(pad=3)

    for j in range(len(agents)):
      sns.heatmap(agents[j].signal_history[i], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, 
      fmt=".1f", ax=axs[j, 0])
      axs[j, 0].set_xlabel("states")
      axs[j, 0].set_ylabel("messages")
      axs[j, 0].set_title(f"Agent {j+1}\'s weights")

      sns.heatmap(agents[j].action_history[i], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, 
      fmt=".1f", ax=axs[j, 1])
      axs[j, 1].set_xlabel("actions")
      axs[j, 1].set_ylabel("messages")
      axs[j, 1].set_title(f"Agent {j+1}\'s weights")

    fig.suptitle(f"Rollout {(i+1)*record_interval}")
    plt.savefig(f"./images/network_{(i+1)*record_interval}.png")
    plt.close(fig)

  images = []
  for filename in [f"./images/network_{(j+1)*record_interval}.png" for j in range(num_images)]:
    images.append(imageio.imread(filename))

  if not os.path.exists("./simulations"):
    os.mkdir("simulations")
  
  subfolder = f"{len(agents[0].signal_history[0][0])}_{len(agents[0].action_history[0])}_{len(agents[0].action_history[0][0])}"
  if not os.path.exists(f"./simulations/{subfolder}"):
    os.makedirs(f"simulations/{subfolder}/")

  imageio.mimsave(output_file, images, duration=duration)


def distFromOptimal(signal_prob) -> float:
  choices = np.zeros_like(signal_prob)
  for i in range(len(choices[0])):
    signal = np.argmax(signal_prob[:, i])
    choices[signal, i] = 1

  optimal_size = len(choices[0]) / len(choices)

  optimal = np.zeros_like(choices)
  used = []
  for i in range(len(choices)):
    j = 0
    max_score = 0
    max_j = 0
    for _ in range(len(choices)):
      if j in used:
        j += 1
        continue
      score = 0
      for _ in range(optimal_size):
        score += choices[i, j]
        j += 1
      if score >= max_score:
        max_score = score
        max_j = j - optimal_size + 1
    
    for k in range(optimal_size):
      optimal[i, max_j + k] = 1
      used.append(max_j + k)
    



