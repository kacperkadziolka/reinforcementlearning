import numpy as np
from learning_curve import LearningCurvePlot, smooth_curve
import os

def average_over_repetitions(returns_over_repetitions_input):
    smoothing_window = 100

    l_curve = np.mean(np.array(returns_over_repetitions_input), axis=0)
    l_curve = smooth_curve(l_curve, smoothing_window)
    return l_curve

def repetitions_REINFORCE(dir_input):
    returns_rep = []

    for filename in os.listdir(dir_input):
        if filename.endswith('.npy'):
            file_path = os.path.join(dir_input, filename)
            # Store the repetition
            returns_rep.append(np.load(file_path))

    return returns_rep

def repetitions_ActorCritic(dir_input, use_bootstrapping, use_baseline):
    returns_rep = []

    if use_bootstrapping and not use_baseline:
        for filename in os.listdir(dir_input):
            if filename.endswith('.npy') and "BootstrappingOnly" in filename:
                file_path = os.path.join(dir_input, filename)
                # Store the repetition
                returns_rep.append(np.load(file_path))

    if not use_bootstrapping and use_baseline:
        for filename in os.listdir(dir_input):
            if filename.endswith('.npy') and "BaselineOnly" in filename:
                file_path = os.path.join(dir_input, filename)
                # Store the repetition
                returns_rep.append(np.load(file_path))

    if use_bootstrapping and use_baseline:
        for filename in os.listdir(dir_input):
            if filename.endswith('.npy') and "BaselineOnly" not in filename and "BootstrappingOnly" not in filename \
                    and "plain" not in filename:
                file_path = os.path.join(dir_input, filename)
                # Store the repetition
                returns_rep.append(np.load(file_path))

    if not use_bootstrapping and not use_baseline:
        for filename in os.listdir(dir_input):
            if filename.endswith('.npy') and "plain" in filename:
                file_path = os.path.join(dir_input, filename)
                # Store the repetition
                returns_rep.append(np.load(file_path))

    return returns_rep


if __name__ == '__main__':
    directory = 'figures'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # ActorCritic agent variations
    Plot = LearningCurvePlot(title='Various variations of Actor-Critic')
    directory = 'vectors/actor_critic'
    returns_over_repetitions = repetitions_ActorCritic(directory, use_bootstrapping=True, use_baseline=True)
    learning_curve = average_over_repetitions(returns_over_repetitions)
    Plot.add_curve(learning_curve, label="Actor-Critic")

    returns_over_repetitions = repetitions_ActorCritic(directory, use_bootstrapping=True, use_baseline=False)
    learning_curve = average_over_repetitions(returns_over_repetitions)
    Plot.add_curve(learning_curve, label="Actor-Critic with bootstrapping")

    returns_over_repetitions = repetitions_ActorCritic(directory, use_bootstrapping=False, use_baseline=True)
    learning_curve = average_over_repetitions(returns_over_repetitions)
    Plot.add_curve(learning_curve, label="Actor-Critic with baseline")

    returns_over_repetitions = repetitions_ActorCritic(directory, use_bootstrapping=False, use_baseline=False)
    learning_curve = average_over_repetitions(returns_over_repetitions)
    Plot.add_curve(learning_curve, label="Plain Actor-Critic")

    Plot.save('figures/A2C_variations.png')

    # REINFORCE vs AC2 agent
    Plot = LearningCurvePlot(title='REINFORCE vs Actor-Critic')

    directory = 'vectors/reinforce'
    returns_over_repetitions = repetitions_REINFORCE(directory)
    learning_curve = average_over_repetitions(returns_over_repetitions)
    Plot.add_curve(learning_curve, label="REINFORCE")

    directory = 'vectors/actor_critic'
    returns_over_repetitions = repetitions_ActorCritic(directory, use_bootstrapping=False, use_baseline=True)
    learning_curve = average_over_repetitions(returns_over_repetitions)
    Plot.add_curve(learning_curve, label="Actor-Critic with baseline")

    returns_over_repetitions = repetitions_ActorCritic(directory, use_bootstrapping=False, use_baseline=False)
    learning_curve = average_over_repetitions(returns_over_repetitions)
    Plot.add_curve(learning_curve, label="Plain Actor-Critic")

    Plot.save('figures/A2C_vs_REINFORCE.png')
