# DQN in CartPole Environment

## Setup & Requirements
Mandatory dependencies and libraries are listed in delivered `requirements.txt` file.  
Use the command ```pip install -r requirements.txt``` to install all packages.

## Running program
We provide the `main.py` file which acts as an only way to actually see the plotted progress of DQN algorithm 
during its convergence.  

All tuning and experiments done throughout are stored as separate files in this project:
- To run the hyperparameter tuning, see the file: `HpTuning.py`
- To run policy comparison and plot the figure, run: `ExperimentPolicy.py`
- Finally, to investigate the results of ablation study, execute file: `ExperimentAblation.py`

Note that, these files are not presenting any graphical user-friendly plots, instead progress can be tracked with 
console messages.
