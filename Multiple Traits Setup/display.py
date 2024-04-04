import matplotlib.pyplot as plt
import imageio.v2 as imageio
import seaborn as sns
import os

def gen_gif(signal_history: list, action_history: list, num_iter: int, record_interval: int, duration: int, output_file: str):
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

  # ix = []
  # epy = []
  # optp_y = []
  # infoy = []
  # opti_y = []

  for i in range(num_images):
    width = len(signal_history[0][0])//2*4+8

    fig, axs = plt.subplots(2, 4, figsize=(width, width*3//4))
    plt.tight_layout(pad=3)

    # ix.append((i+1)*record_interval)
    # epy.append(ep_fn(signal_history[i], action_history[i]))
    # optp_y.append(opt_payoff)
    # infoy.append(info_measure(signal_history[i]))
    # opti_y.append(opt_info)

    for j in range(len(action_history[0])):

      sns.heatmap(signal_history[i][j], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, 
      fmt=".1f", ax=axs[0, j])
      axs[0, j].set_xlabel("states")
      axs[0, j].set_ylabel("messages")
      axs[0, j].set_title(f"Sender\'s Signal {j}")

      sns.heatmap(action_history[i][j], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, 
      fmt=".1f", ax=axs[1, j])
      axs[1, j].set_xlabel("actions")
      axs[1, j].set_ylabel("messages")
      axs[1, j].set_title(f"Receiver\'s Signal {j}")

    # axs[2].plot(ix, epy, label="expected")
    # axs[2].plot(ix, optp_y, label="optimal")
    # axs[2].legend(loc="upper left")
    # axs[2].set_xlabel("rollout")
    # axs[2].set_ylabel("expected payoff")
    # axs[2].set_title("Expected payoff by rollout")

    # axs[3].plot(ix, infoy, label="current")
    # axs[3].plot(ix, opti_y, label="optimal")
    # axs[3].legend(loc="upper left")
    # axs[3].set_xlabel("rollout")
    # axs[3].set_ylabel("info measure")
    # axs[3].set_title("Info measure by rollout")

    fig.suptitle(f"Rollout {(i+1)*record_interval}")
    plt.savefig(f"./images/game_{(i+1)*record_interval}.png")
    plt.close(fig)

  images = []
  for filename in [f"./images/game_{(j+1)*record_interval}.png" for j in range(num_images)]:
    images.append(imageio.imread(filename))

  if not os.path.exists("./simulations"):
    os.mkdir("simulations")
  
  subfolder = f"{len(signal_history[0][0])}_{len(action_history[0])//2}_{len(action_history[0][0])}"
  if not os.path.exists(f"./simulations/{subfolder}"):
    os.makedirs(f"simulations/{subfolder}/")
  
  imageio.mimsave(output_file, images, duration=duration)

# f"./simulations/{self.num_states}_{self.num_signals}_{self.num_actions}/{self.reward_param}{'_null' if self.null_signal else ''}_{num_iter}.gif"