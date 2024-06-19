# Policy-based RL

## Setup & Requirements
Mandatory dependencies and libraries are listed in `requirements.txt` file.  
Use the command ```pip install -r requirements.txt``` to install all packages.

Please, do not forget to additionally execute required command to install env dependencies:   
`pip install gym[box2d]`

## Running program
We provide the `run.py` file, which serves as an entry point. You can use this file to run the program and initialize the 
desired agent based on the provided command line argument. Note that running this file will not generate visualizations 
of the environment or produce any figures. It will execute a single training run consisting of 10,000 episodes.

- To run REINFORCE agent:
```shell
python run.py --type reinforce
```

- To run ActorCritic agent:
```shell
python run.py --type actorcritic
```

### HP tuning and experiments
To handle extensive computations throughout the project and to be more flexible with running and plotting experiments, 
we devised the following solution:
1. After each training run is completed, we store the average rewards per episode as a NumPy array and save them to disk.
2. Using specially prepared functions, we can now combine multiple independent runs, average them over repetitions, and,
3. if needed, place the same vector in multiple plots.

**HP Tuning**  
To run the hyperparameter tuning, refer to the file `hp_tuning.py`. Note that, we used a single file to tune both agents 
with various parameters, so it needs to be configured for the given hyperparameters. To plot the results, use the file 
`plot_hp_tuning.py`. Similarly, the content of this file should also be adjusted.

**Ablation study**  
The ablation studies were conducted using the set of provided files. First, we executed all the repetitions and saved 
the results vector using the `agent_repetitions.py` file. Later, we reused the obtained vector in various plotting 
combinations with the code from the `plot_agent_repetitions.py` file.

Additionally, the histogram plot was created using the `histogram_plot.py` file.