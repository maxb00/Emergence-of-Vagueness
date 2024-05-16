import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio.v2 as imageio
import seaborn as sns
from math import ceil
import numpy as np
import os

def gen_gif(game, num_iter: int, record_interval: int, duration: int, output_file: str):
  """Generates a heatmap gif of the whole simulation and saves it into 
  test.gif

  Args:
    num_images (int): the number of images in the gif
    record_interval (int): number of simulations between each image
    duration (int): the duration an image is shown in the gif
  """
  num_images = num_iter // record_interval
  signal_history = game.sender.signal_history
  action_history = game.receiver.action_history

  if not os.path.exists("./images"):
    os.mkdir("images")

  num_states = len(signal_history[0][0])
  num_signals = len(action_history[0])

  flr_p = game.floor_payoff()
  flr_info = game.floor_info(False)
  w_flr_info = game.floor_info()

  opt_p = game.optimal_payoff()
  opt_info = game.optimal_info(False)
  w_opt_info = game.optimal_info()

  print(flr_info)
  print(opt_info)
  exit(0)

  ix = []
  exp_py = []
  flr_py = []
  opt_py = []
  infoy = []
  flr_infoy = []
  w_flr_infoy = []
  opt_infoy = []
  w_infoy = []
  w_opt_infoy = []

  for _ in range(num_signals):
    infoy.append([])
    w_infoy.append([])
  # opti_y = []

  for i in range(num_images):
    width = num_states//2*num_signals+28
    height = num_states*2+28

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    gs = fig.add_gridspec(nrows=5, ncols=num_signals * 3)

    ix.append((i+1)*record_interval)
    # epy.append(ep_fn(signal_history[i], action_history[i]))
    # optp_y.append(opt_payoff)
    exp_py.append(game.expected_payoff(signal_history[i], action_history[i]))
    flr_py.append(flr_p)
    opt_py.append(opt_p)

    total_info, info_sig, info_state = game.info_measure(signal_history[i], False)
    
    w_total_info, w_info_sig, w_info_state = game.info_measure(signal_history[i])
    

    accum_info = 0
    accum_w_info = 0
    for sig in range(num_signals):
      accum_info += info_sig[sig]
      infoy[sig].append(accum_info)
      accum_w_info += w_info_sig[sig]
      w_infoy[sig].append(accum_w_info)
    flr_infoy.append(flr_info)
    w_flr_infoy.append(w_flr_info)
    opt_infoy.append(opt_info)
    w_opt_infoy.append(w_opt_info)

    # opti_y.append(opt_info)

    sns.set_theme(font_scale=2)
    for j in range(num_signals):

      ax1 = fig.add_subplot(gs[0, j*3:j*3+3])
      sns.heatmap(signal_history[i][j], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, 
      fmt=".1f", ax=ax1)
      ax1.set_xlabel("Trait 1")
      ax1.set_ylabel("Trait 2")
      ax1.set_title(f"Sender\'s Signal {j}")

      ax6 = fig.add_subplot(gs[1, j*3:j*3+3])
      sns.heatmap(info_state[j], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, 
      fmt=".2f", ax=ax6)
      ax6.set_xlabel("Trait 1")
      ax6.set_ylabel("Trait 2")
      ax6.set_title(f"Signal {j}'s Info Measure")

      ax2 = fig.add_subplot(gs[2, j*3:j*3+3])
      sns.heatmap(action_history[i][j], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, 
      fmt=".1f", ax=ax2)
      ax2.set_xlabel("Trait 1")
      ax2.set_ylabel("Trait 2")
      ax2.set_title(f"Receiver\'s Signal {j}")

    # axs[2].plot(ix, epy, label="expected")
    # axs[2].plot(ix, optp_y, label="optimal")
    # axs[2].legend(loc="upper left")
    # axs[2].set_xlabel("rollout")
    # axs[2].set_ylabel("expected payoff")
    # axs[2].set_title("Expected payoff by rollout")

    ax3 = fig.add_subplot(gs[3, 0:num_signals])
    # for sig in reversed(range(num_signals)):
    #   ax3.plot(ix, w_infoy[sig], label=f"Signal {sig}")
    #   ax3.fill_between(ix, w_infoy[sig])
    ax3.plot(ix, infoy[-1], label="Unweighted")
    ax3.plot(ix, flr_infoy, label="Floor unweighted")
    ax3.plot(ix, opt_infoy, label="Optimal unweighted")
    ax3.legend(loc="upper left")
    ax3.set_xlabel("rollout")
    ax3.set_ylabel("info measure")
    ax3.set_title("Unweighted info measure by rollout")

    ax4 = fig.add_subplot(gs[3, num_signals:num_signals*2])
    # for sig in reversed(range(num_signals)):
    #   ax3.plot(ix, w_infoy[sig], label=f"Signal {sig}")
    #   ax3.fill_between(ix, w_infoy[sig])
    ax4.plot(ix, w_infoy[-1], label="Weighted")
    ax4.plot(ix, w_flr_infoy, label="Floor weighted")
    ax4.plot(ix, w_opt_infoy, label="Optimal weighted")
    ax4.legend(loc="upper left")
    ax4.set_xlabel("rollout")
    ax4.set_ylabel("info measure")
    ax4.set_title("Weighted info measure by rollout")

    ax7 = fig.add_subplot(gs[3, num_signals*2:num_signals*3])
    # for sig in reversed(range(num_signals)):
    #   ax3.plot(ix, w_infoy[sig], label=f"Signal {sig}")
    #   ax3.fill_between(ix, w_infoy[sig])
    ax7.plot(ix, exp_py, label="Payoff")
    ax7.plot(ix, flr_py, label="Floor payoff")
    ax7.plot(ix, opt_py, label="Optimal payoff")
    ax7.legend(loc="upper left")
    ax7.set_xlabel("rollout")
    ax7.set_ylabel("payoff")
    ax7.set_title("Payoff by rollout")
    
    text = f"Total unweighted/weighted info measure: [{total_info:.5f}, {w_total_info:.5f}]\n"
    text += "Info measure by signal: ["
    for sig in range(num_signals):
      text += f"{info_sig[sig]:.5f}, "
    text += "]\n"
    text += "Weighted info measure by signal: ["
    for sig in range(num_signals):
      text += f"{w_info_sig[sig]:.5f}, "
    text += "]\n"
    info_by_trait = game.info_measure_by_trait(signal_history[i], False)
    w_info_by_trait = game.info_measure_by_trait(signal_history[i])
    text += f"Trait 1's info measure by signal: ["
    for sig in range(num_signals):
      text += f"{info_by_trait[0][sig]:.5f}, "
    text += "]\n"
    text += f"Trait 1's weighted info measure by signal: ["
    for sig in range(num_signals):
      text += f"{w_info_by_trait[0][sig]:.5f}, "
    text += "]\n"
    text += f"Trait 2's info measure by signal: ["
    for sig in range(num_signals):
      text += f"{info_by_trait[1][sig]:.5f}, "
    text += "]\n"
    text += f"Trait 2's weighted info measure by signal: ["
    for sig in range(num_signals):
      text += f"{w_info_by_trait[1][sig]:.5f}, "
    text += "]\n\n"

    info_corner = (np.sum(info_state[:, 0, 0]) + np.sum(info_state[:, 0, 2]) + np.sum(info_state[:, 2, 0]) + np.sum(info_state[:, 2, 2]))/4
    info_middle = (np.sum(info_state[:, 0, 1]) + np.sum(info_state[:, 1, 2]) + np.sum(info_state[:, 1, 0]) + np.sum(info_state[:, 2, 1]))/4
    info_center = np.sum(info_state[:, 1, 1])

    text += f"Unweighted corner|middle|center: {info_corner:.5f} | {info_middle:.5f} | {info_center:.5f}\n"

    w_info_corner = (np.sum(w_info_state[:, 0, 0]) + np.sum(w_info_state[:, 0, 2]) + np.sum(w_info_state[:, 2, 0]) + np.sum(w_info_state[:, 2, 2]))/4
    w_info_middle = (np.sum(w_info_state[:, 0, 1]) + np.sum(w_info_state[:, 1, 2]) + np.sum(w_info_state[:, 1, 0]) + np.sum(w_info_state[:, 2, 1]))/4
    w_info_center = np.sum(w_info_state[:, 1, 1])

    text += f"Weighted corner|middle|center: {w_info_corner:.5f} | {w_info_middle:.5f} | {w_info_center:.5f}"

    ax5 = fig.add_subplot(gs[4, :])
    ax5.axis("off")
    ax5.annotate(text,
            xy=(0, 0.05), xytext=(0, -20),
            xycoords=('axes fraction', 'figure fraction'),
            textcoords='offset points',
            size=34, ha='left', va='bottom')

    fig.suptitle(f"Rollout {(i+1)*record_interval}")
    plt.savefig(f"./images/game_{(i+1)*record_interval}.png")
    plt.close(fig)

  images = []
  for filename in [f"./images/game_{(j+1)*record_interval}.png" for j in range(num_images)]:
    images.append(imageio.imread(filename))

  if not os.path.exists("./simulations"):
    os.mkdir("simulations")
  
  subfolder = f"{num_states}_{num_signals}_{num_states}"
  if not os.path.exists(f"./simulations/v5/{subfolder}"):
    os.makedirs(f"simulations/v5/{subfolder}/")
  
  version = 1
  final_output_file = f"{output_file}"
  while os.path.isfile(f"{final_output_file}.gif"):
    final_output_file = f"{output_file}#{version}"
    version += 1
  imageio.mimsave(f"{final_output_file}.gif", images, duration=duration)

# f"./simulations/{self.num_states}_{self.num_signals}_{self.num_actions}/{self.reward_param}{'_null' if self.null_signal else ''}_{num_iter}.gif"