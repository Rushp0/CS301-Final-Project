import nni
import time
from nni.experiment import Experiment

search_space = {
    'batch_size': {'_type': 'choice', '_value': [1,2,4,8,16]},
    'epochs': {'_type': 'choice', '_value': [1,2,4,8,16,32,50]}
}

print("START")
# configure experiment
experiment = Experiment('local')
experiment.config.trial_command = 'python model.py'
experiment.config.trial_code_directory = "."

# configure search space
experiment.config.search_space = search_space

# configure tuning alg
experiment.config.tuner.name = 'BOHB'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

# configuremax trials
experiment.config.max_trial_number = 2
experiment.config.trial_concurrency = 1

experiment.run(8080)
# experiment.stop()
