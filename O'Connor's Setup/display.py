import matplotlib.pyplot as plt
import imageio.v2 as imageio
import seaborn as sns
import os

def gen_gif(signal_history: list, action_history: list, ep_fn, opt_payoff: float, info_measure, opt_info: float, num_iter: int, record_interval: int, duration: int, output_file: str):
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

  ix = []
  epy = []
  optp_y = []
  infoy = []
  opti_y = []

  for i in range(num_images):
    fig, axs = plt.subplots(4, 1, figsize=(8, 6))
    plt.tight_layout(pad=3)

    ix.append((i+1)*record_interval)
    epy.append(ep_fn(signal_history[i], action_history[i]))
    optp_y.append(opt_payoff)
    infoy.append(info_measure(signal_history[i]))
    opti_y.append(opt_info)

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

    axs[2].plot(ix, epy, label="expected")
    axs[2].plot(ix, optp_y, label="optimal")
    axs[2].legend(loc="upper left")
    axs[2].set_xlabel("rollout")
    axs[2].set_ylabel("expected payoff")
    axs[2].set_title("Expected payoff by rollout")

    axs[3].plot(ix, infoy, label="current")
    axs[3].plot(ix, opti_y, label="optimal")
    axs[3].legend(loc="upper left")
    axs[3].set_xlabel("rollout")
    axs[3].set_ylabel("info measure")
    axs[3].set_title("Info measure by rollout")

    fig.suptitle(f"Rollout {(i+1)*record_interval}")
    plt.savefig(f"./images/game_{(i+1)*record_interval}.png")
    plt.close(fig)

  images = []
  for filename in [f"./images/game_{(j+1)*record_interval}.png" for j in range(num_images)]:
    images.append(imageio.imread(filename))

  if not os.path.exists("./simulations"):
    os.mkdir("simulations")
  
  subfolder = f"{len(signal_history[0][0])}_{len(action_history[0])}_{len(action_history[0][0])}"
  if not os.path.exists(f"./simulations/{subfolder}"):
    os.makedirs(f"simulations/{subfolder}/")
  
  imageio.mimsave(output_file, images, duration=duration)

# f"./simulations/{self.num_states}_{self.num_signals}_{self.num_actions}/{self.reward_param}{'_null' if self.null_signal else ''}_{num_iter}.gif"

def gen_graph(avg_vague_lvls: list):
  if not os.path.exists("./images"):
    os.mkdir("images")

  rif_widths = [0, 2, 10, 20]

  fig, ax = plt.subplots(1, 1, figsize=(8, 6))
  plt.tight_layout(pad=3)

  ax.plot(rif_widths, avg_vague_lvls[0], 'o-')
  # ax.plot(rif_widths, avg_vague_lvls[1], 's-', label="20")
  # ax.plot(rif_widths, avg_vague_lvls[2], 'D-', label="10")
  # ax.legend(loc="upper right", title="Signals")
  ax.set_xlabel("Reinforcement Width")
  ax.set_ylabel("Success")
  ax.set_xlim(0, 20)
  ax.set_ylim(0, 1)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.set_title("Figure 11 - original")

  if not os.path.exists("./figures"):
    os.mkdir("figures")

  plt.savefig("./figures/Figure-11-orig.png")