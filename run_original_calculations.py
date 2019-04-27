import argparse
import gin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import influence
import models
import utils

from dqn import DQN
from circle import CircleEnv

parser = argparse.ArgumentParser()
parser.add_argument('split', type=str)
args = parser.parse_args()

gin.parse_config_file('configs/influence/influence.gin')

start_time = time.time()
oracle_init_model = 'data/circle/init_models/oracle_init_model.h5'
oracle_init_target_model = 'data/circle/init_target_models/oracle_init_target_model.h5'
oracle_training_data = 'data/circle/training_data/oracle_training_data.pkl'
test_data = 'data/circle/test_data/oracle_test_data.pkl'

training_data = pd.read_pickle(oracle_training_data)
test_data = pd.read_pickle(test_data)

unique_states = pd.read_pickle('data/circle/experiments/training_data_splits/{}'.format(args.split))

for _, state in unique_states.iterrows():
    state_x_str = str(state.state_x).replace('.', '_')
    state_y_str = str(state.state_y).replace('.', '_')
    file_prefix = "{}_{}".format(state_x_str, state_y_str)
    influence.influence(state, training_data, test_data, oracle_init_model, oracle_init_target_model, file_prefix)

print('TOTAL TIME ELAPSED: {}'.format(time.time() - start_time))
