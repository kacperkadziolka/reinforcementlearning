import os

import numpy as np
from ablation_study.learning_curve import LearningCurvePlot, smooth_curve

# directory
directory = 'plots'
if not os.path.exists(directory):
    os.makedirs(directory)

# Load the .npy files for REINFORCE
nstep_1 = np.load('plot_vectors/episode_rewards_nstep_1.npy')
nstep_3 = np.load('plot_vectors/episode_rewards_nstep_3.npy')
nstep_5 = np.load('plot_vectors/episode_rewards_nstep_5.npy')
nstep_30 = np.load('plot_vectors/episode_rewards_nstep_30.npy')
nstep_50 = np.load('plot_vectors/episode_rewards_nstep_50.npy')


smoothing_window = 75

Plot = LearningCurvePlot(title='Number of bootstrapping\'s n-step fine-tuning')
Plot.add_curve(smooth_curve(nstep_1, smoothing_window), label="n_step=1")
Plot.add_curve(smooth_curve(nstep_3, smoothing_window), label="n_step=3")
Plot.add_curve(smooth_curve(nstep_5, smoothing_window), label="n_step=5")
Plot.add_curve(smooth_curve(nstep_30, smoothing_window), label="n_step=30")
Plot.add_curve(smooth_curve(nstep_50, smoothing_window), label="n_step=50")
Plot.save('plots/A2C_nstep_tuning.png')
