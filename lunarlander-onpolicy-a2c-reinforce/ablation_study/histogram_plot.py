import os

import numpy as np
from matplotlib import pyplot as plt

directory = "vectors/actor_critic"
filename = "episode_rewards_plain_AC_rep_3.npy"
file_path = os.path.join(directory, filename)
rewards = np.load(file_path)

plt.hist(rewards, bins=25, alpha=0.75, color='blue')
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.title('Histogram of Rewards Over Training Episodes')
plt.grid(True)

output_directory = "figures"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
output_filename = "rewards_histogram.png"
output_path = os.path.join(output_directory, output_filename)

# Save the plot to the specified file
plt.savefig(output_path)
