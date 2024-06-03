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

  ix = [] # x coordinates (1, 2, ...)
  exp_py = [] # expected payoff
  flr_py = [] # floor payoff
  opt_py = [] # optimal payoff
  infoy = [] # info
  flr_infoy = [] # floor info
  opt_infoy = [] # optimal info
  w_infoy = [] # weighted info
  w_flr_infoy = [] # weighted floor info
  w_opt_infoy = [] # weighted optimal info

  for _ in range(num_signals): # used for old feature, could delete?
    infoy.append([])
    w_infoy.append([])

  for i in range(num_images):
    width = num_states//2*num_signals+32 # HARD-CODED
    height = num_states*2+32 # HARD-CODED

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    gs = fig.add_gridspec(nrows=5, ncols=num_signals * 3)

    ix.append((i+1)*record_interval)
    exp_py.append(game.expected_payoff(signal_history[i], action_history[i]))
    flr_py.append(flr_p)
    opt_py.append(opt_p)

    total_info, info_sig, info_state = game.info_measure(signal_history[i], False)
    
    w_total_info, w_info_sig, w_info_state = game.info_measure(signal_history[i])
    
    # used for old feature, could delete?
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

    sns.set_theme(font_scale=2)
    for j in range(num_signals):

      # sender strategy heatmap
      sender_heatmap_plot = fig.add_subplot(gs[0, j*3:j*3+3])
      sns.heatmap(signal_history[i][j], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, 
      fmt=".1f", ax=sender_heatmap_plot)
      sender_heatmap_plot.set_xlabel("Trait 1")
      sender_heatmap_plot.set_ylabel("Trait 2")
      sender_heatmap_plot.set_title(f"Sender\'s Signal {j}")

      # info by state heatmap
      info_by_state_plot = fig.add_subplot(gs[1, j*3:j*3+3])
      sns.heatmap(info_state[j], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, 
      fmt=".2f", ax=info_by_state_plot)
      info_by_state_plot.set_xlabel("Trait 1")
      info_by_state_plot.set_ylabel("Trait 2")
      info_by_state_plot.set_title(f"Signal {j}'s UW Info Measure")

      # receiver strategy heatmap
      receiver_heatmap_plot = fig.add_subplot(gs[2, j*3:j*3+3])
      sns.heatmap(action_history[i][j], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, 
      fmt=".1f", ax=receiver_heatmap_plot)
      receiver_heatmap_plot.set_xlabel("Trait 1")
      receiver_heatmap_plot.set_ylabel("Trait 2")
      receiver_heatmap_plot.set_title(f"Receiver\'s Signal {j}")

    # info measure plot
    info_plot = fig.add_subplot(gs[3, 0:num_signals])
    # for sig in reversed(range(num_signals)):
    #   ax3.plot(ix, w_infoy[sig], label=f"Signal {sig}")
    #   ax3.fill_between(ix, w_infoy[sig])
    info_plot.plot(ix, infoy[-1], label="Unweighted")
    info_plot.plot(ix, flr_infoy, label="Floor unweighted")
    info_plot.plot(ix, opt_infoy, label="Optimal unweighted")
    info_plot.legend(loc="upper left")
    info_plot.set_xlabel("rollout")
    info_plot.set_ylabel("info measure")
    info_plot.set_title("Unweighted info measure by rollout")

    # weighted info measure plot
    w_info_plot = fig.add_subplot(gs[3, num_signals:num_signals*2])
    # for sig in reversed(range(num_signals)):
    #   ax3.plot(ix, w_infoy[sig], label=f"Signal {sig}")
    #   ax3.fill_between(ix, w_infoy[sig])
    w_info_plot.plot(ix, w_infoy[-1], label="Weighted")
    w_info_plot.plot(ix, w_flr_infoy, label="Floor weighted")
    w_info_plot.plot(ix, w_opt_infoy, label="Optimal weighted")
    w_info_plot.legend(loc="upper left")
    w_info_plot.set_xlabel("rollout")
    w_info_plot.set_ylabel("info measure")
    w_info_plot.set_title("Weighted info measure by rollout")

    # payoff plot
    payoff_plot = fig.add_subplot(gs[3, num_signals*2:num_signals*3])
    # for sig in reversed(range(num_signals)):
    #   ax3.plot(ix, w_infoy[sig], label=f"Signal {sig}")
    #   ax3.fill_between(ix, w_infoy[sig])
    payoff_plot.plot(ix, exp_py, label="Payoff")
    payoff_plot.plot(ix, flr_py, label="Floor payoff")
    payoff_plot.plot(ix, opt_py, label="Optimal payoff")
    payoff_plot.legend(loc="upper left")
    payoff_plot.set_xlabel("rollout")
    payoff_plot.set_ylabel("payoff")
    payoff_plot.set_title("Payoff by rollout")
    

    """ HUGE TEXT BLOCK INCOMING """

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

    text += f"Weighted corner|middle|center: {w_info_corner:.5f} | {w_info_middle:.5f} | {w_info_center:.5f}\n"

    info_by_state_t1 = [np.sum(info_state[:, :, i]) for i in range(game.num_states)]
    info_by_state_t2 = [np.sum(info_state[:, i, :]) for i in range(game.num_states)]

    text += f"Unweighted trait 1's low|medium|high: {info_by_state_t1[0]:.5f} | {info_by_state_t1[1]:.5f} | {info_by_state_t1[2]:.5f}"
    text += f" and trait 2's low|medium|high: {info_by_state_t2[0]:.5f} | {info_by_state_t2[1]:.5f} | {info_by_state_t2[2]:.5f}\n"

    w_info_by_state_t1 = [np.sum(w_info_state[:, :, i]) for i in range(game.num_states)]
    w_info_by_state_t2 = [np.sum(w_info_state[:, i, :]) for i in range(game.num_states)]

    text += f"Weighted trait 1's low|medium|high: {w_info_by_state_t1[0]:.5f} | {w_info_by_state_t1[1]:.5f} | {w_info_by_state_t1[2]:.5f}"
    text += f" and trait 2's low|medium|high: {w_info_by_state_t2[0]:.5f} | {w_info_by_state_t2[1]:.5f} | {w_info_by_state_t2[2]:.5f}"

    """ END OF HUGE TEXT BLOCK """

    # statistics plot
    stats_plot = fig.add_subplot(gs[4, :])
    stats_plot.axis("off")
    stats_plot.annotate(text,
            xy=(0, 0.05), xytext=(0, -40),
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
  if not os.path.exists(f"./simulations/v6/{subfolder}"):
    os.makedirs(f"simulations/v6/{subfolder}/")
  
  version = 1
  final_output_file = f"{output_file}"
  while os.path.isfile(f"{final_output_file}.gif"):
    final_output_file = f"{output_file}#{version}"
    version += 1
  imageio.mimsave(f"{final_output_file}.gif", images, duration=duration)
