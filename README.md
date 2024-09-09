# Main Objective
We aim to analyze the existance of vagueness in natural language using variations of Lewis signaling games. Some interesting questions include:
 - How do we define vagueness in the context of a signaling game?
 - How does vagueness emerge?
 - How does the changing the signaling environment affect the emergence of vagueness?
 - Is vagueness necessary?
 - Does the optimal signaling strategy contain the most information?
 - How would a more complex setup affect the agents and vagueness?

## Signaling Game
A signaling game consists of:
 - A world (a set of states)
 - A sender (a mapping from a world state to a signal)
 - A receiver (a mapping from a signal to an action)
 - A reward function assigning each (state, action) pair a reward

A single round of the game proceeds as follows.
 1. A random state is generated.
 2. The sender observes the state and sends a signal to the receiver.
 3. The receiver, who cannot see the state, takes an action based on the signal from the sender.
 4. A reward is given to the agent based on the (state, action) pair and the reward function.

For this project, we will focus on setups where the state space and the action space is identical, but the signal space is smaller. The reward function will be dependent on the distance between the state and the action such that the smaller the distance, the higher the reward. 

We will use a form of reinforcement learning for both agents. The agents will each have a signal/action "weight" 2D matrix, where the rows correspond to the signals and the columns correspond to the states/actions. The probability that the sender sends signal $i$ on state $w$ will be generated based on the weight in row $i$ column $w$ of the signal weight matrix. The probability that the receiver takes action $j$ on signal $i$ will be generated similarly. Each time the sender receive a reward $r$ for sending signal $i$ on state $w$, $r$ will be added to the weight in row $i$ column $w$ of the signal weight matrix. Similarly for the receiver. Generally, more weight means higher probability.

## Vagueness
A term is vague to the extent that it has borderline cases. In the context of a signaling game, we can think of borderline cases as a region where there are mixed signaling. However, we feel this interpretation does not perfectly capture vagueness in human communication. We believe that vagueness can also be represented as the inability to signal. For this, we introduce an additional "null" signal for the sender to use in borderline cases. In our setups, we study both interpretations.

## Information Content Measure
We use KL divergence as a method to calculate "unweighted" information content, taken from Skyrms' "Signal". The weighted version additionally takes into account the probability of the signal.

## Setups
We will present different variations of the signaling game that we have used in our study so far.

### Basic Setup
This setup aims to see how different parameters of the game affect the agents' ability to form a signaling strategy. 

We also equip the agents with stimulus generalization learning, which is common in human learning and could be a factor that leads to the rise of vagueness. Stimulus generalization refers to when a learning agent conditioned to one stimulus responds the same way to other similar stimuli. In the context of a signaling game, for example, if the current state is $w$, the sender sends signal $i$, and the reward is $r$, then the sender not only adds $r$ to the signal weight in row $i$ column $w$, but also adds a portion of $r$ to the surrounding weights of row $i$ (i.e. row $i$ column $i-1$, row $i$ column $i+1$). We can interpret this as the sender will consider the states surrounding $w$ to be similar to state $w$, similar to how, for example, one would associate the word "blue" with a lot of different shades of blue. The receiver, however, would behave a bit differently with stimulus generalization. If applied to the receiver, stimulus generalization would allow the receiver to react similarly to similar signals when reinforced. Our wish is to consider signals to act as a language, so intuitively, similar words shouldn't always have similar meaning. For the receiver, we will implement stimulus generalization as the receiver will take similar actions for the same signal. This implementation is taken from O'Connor's "Evolution of Vagueness". ([ref](https://cailinoconnor.com/wp-content/uploads/2015/03/The_Evolution_of_Vagueness_official.pdf))

A. Non-null simulations
1. Expectation:
- For the sender's side, we expect the state space to be divided into $k$ "buckets", with $k$ being the number of signals and a "bucket" meaning a contiguous set of states that are mapped to the same signal. In between each bucket would be bordeline regions of mixed signaling.
- For the receiver's side, we expect the receiver to take the median action for each bucket.
- Generally, more favorable reward functions would lead to faster learning, but may get the agents stuck in a non-optimal strategy. Less favorable reward functions would slow down learning and possibly prevent the agents from developing a stable strategy.
2. Results:
- With the right reward parameters, the agents can develop stable strategies that converge close to optimal. However, there are many cases where the borderline regions do not appear in between buckets, but rather at the edges of the state space.

B. Null simulations:
1. Expectations:
- The agents will develop stable strategies that are completely deterministic, and all borderline regions will be replaced by null signal buckets.

2. Results:
- The agents manage to develop stable deterministic strategies that converge close to optimal. However, the borderline regions are pushed to the edges of the state space, similar to the non-null simulations.

C. Additional Note on Information Content
For both type of simulations, information content seems to be correlated to expected payoff. However, the optimal strategy does not necessarily produce the highest information content, and vice versa.

### O'Connor's Setup
This setup aims to compare our implementation with O'Connor's, as shown in O'Connor's "Evolution of Vagueness". ([ref](https://cailinoconnor.com/wp-content/uploads/2015/03/The_Evolution_of_Vagueness_official.pdf))

There are 2 differences between our setups.
a. We use a linear reward function, where the reward parameters are the initial reward and the slope of the function. O'Connor uses a Gaussian reward function, where the reward parameters are the initial reward and the width of the function. Therefore, our implementation has negative rewards, while O'Connor's does not.
b. We use the exponential function in calculating the probability for the signals/actions. O'Connor does not.

* Results:
The results of the two implementations are quite similar with a few differences.
1. Our implementation takes fewer learning steps for the agents to develop a stable strategy, due to the use of negative reward and the exponential function.
2. Our implementation allows the agent to develop strategy that are less vague and contains more information content, due to the use of the exponential function. However, the expected payoff are similar.
3. Setups that uses the exponential function benefits more from wider reinforcement widths, while setups that does not use the exponential function benefits more from more narrow reinforcement widths.

### Multiple Traits Setup
This setup aims to analyze signaling with multi-dimensional state space. 

This is inspired by the fact that a real-world person or object has multiple "traits", i.e. a person could be tall or short, thin or heavy. For now, our focus for this setup has been to look at how information content is distributed across the broader state space. Our expectation is that information content will be lower in the states close to center and higher in the states further from the center. This is because we expect the agents to focus more on the traits that are extreme and distinguishable, instead of states that are average.

For simplicity, we have only been looking at 2-dimensional setups and a few 3-dimensional setups. We also switch to using a Gaussian distribution to generate the states, which is similar to the real world.

* Results:
The agents are able develop strategies close to optimal, with the unweighted information content distributed as we expected (with the outer states containing higher information content). This is true for both 2- and 3-dimensional cases. However, as we will discuss in the next section, the optimal strategies for 3-dimensional cases using Gaussian distribution for the state generation will produce different results.
