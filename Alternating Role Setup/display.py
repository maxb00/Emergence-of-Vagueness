import matplotlib.pyplot as plt
import imageio.v2 as imageio
import seaborn as sns
import os

def gen_gif(signal_histories: list, action_histories: list, ep_fn, num_iter: int, record_interval: int, duration: int, output_file: str, figsize=(15, 6), height_ratios=None):
  """Generates a heatmap gif and saves it into 
  a GIF file

  Args:
    num_images (int): the number of images in the gif
    record_interval (int): number of simulations between each image
    duration (int): the duration an image is shown in the gif
  """
  num_images = num_iter // record_interval

  if not os.path.exists("./images"):
    os.mkdir("images")

  epx1 = []
  epy1 = []
  epx2 = []
  epy2 = []

  for i in range(num_images):
    if height_ratios == None:
      fig, axs = plt.subplots(3, 2, figsize=figsize)
    else:
      fig, axs = plt.subplots(3, 2, figsize=figsize, height_ratios=height_ratios)
    plt.tight_layout(pad=5)

    epx1.append((i+1)*record_interval)
    epy1.append(ep_fn(signal_histories[0][i], action_histories[0][i]))
    epx2.append((i+1)*record_interval)
    epy2.append(ep_fn(signal_histories[1][i], action_histories[1][i]))

    sns.heatmap(signal_histories[0][i], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, 
    fmt=".1f", ax=axs[0, 0])
    axs[0, 0].set_xlabel("states")
    axs[0, 0].set_ylabel("messages")
    axs[0, 0].set_title("Agent 1\'s signal weights")

    sns.heatmap(action_histories[0][i], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, 
    fmt=".1f", ax=axs[1, 0])
    axs[1, 0].set_xlabel("actions")
    axs[1, 0].set_ylabel("messages")
    axs[1, 0].set_title("Agent 2\'s action weights")

    axs[2, 0].plot(epx1, epy1)
    axs[2, 0].set_xlabel("rollout")
    axs[2, 0].set_ylabel("expected payoff")
    axs[2, 0].set_title("Expected payoff by rollout")

    sns.heatmap(signal_histories[1][i], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, 
    fmt=".1f", ax=axs[0, 1])
    axs[0, 1].set_xlabel("states")
    axs[0, 1].set_ylabel("messages")
    axs[0, 1].set_title("Agent 2\'s signal weights")

    sns.heatmap(action_histories[1][i], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, 
    fmt=".1f", ax=axs[1, 1])
    axs[1, 1].set_xlabel("actions")
    axs[1, 1].set_ylabel("messages")
    axs[1, 1].set_title("Agent 1\'s action weights")

    axs[2, 1].plot(epx2, epy2)
    axs[2, 1].set_xlabel("rollout")
    axs[2, 1].set_ylabel("expected payoff")
    axs[2, 1].set_title("Expected payoff by rollout")

    fig.suptitle(f"Rollout {(i+1)*record_interval}")
    plt.savefig(f"./images/game_{(i+1)*record_interval}.png")
    plt.close(fig)

  images = []
  for filename in [f"./images/game_{(j+1)*record_interval}.png" for j in range(num_images)]:
    images.append(imageio.imread(filename))
  imageio.mimsave(output_file, images, duration=duration)

# f"./simulations/{self.num_states}_{self.num_signals}_{self.num_actions}/{self.reward_param}{'_null' if self.null_signal else ''}_{num_iter}.gif"