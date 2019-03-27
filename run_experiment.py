import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg') # in order to save graphs when running on SSH
import matplotlib.pyplot as plt
import h5py
import time

import graphs
import models
import utils

from dqn import DQN

TIMESTR = time.strftime('%Y%m%d-%H%M%S')

# Argument parser setup
# TODO: expand parser
parser = argparse.ArgumentParser()
parser.add_argument('run_identifier', type=str)
parser.add_argument('--n', type=int)
args = parser.parse_args()

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('logs/circle/experiments/{}_{}.log'.format(args.run_identifier, TIMESTR))
file_handler.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

file_formatter = logging.Formatter('[%(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Misc
OBSERVATION_SIZE = 2
ACTION_SIZE = 4
LEARNING_RATE = 0.001
MEMORY_SIZE = 2000

logger.debug('*** Run parameters ***')
logger.debug('Observation size: {}'.format(OBSERVATION_SIZE))
logger.debug('Action size: {}'.format(ACTION_SIZE))
logger.debug('Learning rate: {}'.format(LEARNING_RATE))
logger.debug('Memory size: {}'.format(MEMORY_SIZE))
logger.debug('*'*30)

ORACLE_MODEL_FILE = 'data/circle/models/base_run_20190324-225802_model.h5'
INIT_ORACLE_MODEL_FILE = 'data/circle/init_models/base_run_20190324-225802_init_model.h5'
ORACLE_MODEL_TRAINING_DATA = 'data/circle/training_data/base_run_20190324-225802_training_data.h5'

logger.debug('*** Oracle model details ***')
logger.debug('Model file: {}'.format(ORACLE_MODEL_FILE))
logger.debug('Init model file: {}'.format(INIT_ORACLE_MODEL_FILE))
logger.debug('Model training data file: {}'.format(ORACLE_MODEL_TRAINING_DATA))
logger.debug('*'*30)

# Load data
logger.info('*** Loading data ***')
oracle_model = models.build_basic_network(OBSERVATION_SIZE, ACTION_SIZE, LEARNING_RATE)
oracle_model.load_weights(ORACLE_MODEL_FILE)
oracle_agent = DQN(OBSERVATION_SIZE, ACTION_SIZE, MEMORY_SIZE)
oracle_agent.model = oracle_model

STATES, ORACLE_ACTIONS = utils.generate_agent_actions(oracle_agent, 10000)

training_data = utils.load_data(ORACLE_MODEL_TRAINING_DATA, 'training')
training_states = training_data[:, 0:2]
training_targets = training_data[:, 2:6]

unique_states, unique_counts = np.unique(training_states, axis=0, return_counts=True)
# shuffle order of states so that they appear nicely in plot during training
u_states_counts = np.hstack((unique_states, np.reshape(unique_counts, (len(unique_counts), 1))))
np.random.shuffle(u_states_counts)
unique_states = u_states_counts[:,0:2]
unique_counts = u_states_counts[:,2]

if args.n:
    unique_states = unique_states[:args.n]
    unique_counts = unique_counts[:args.n]

INFLUENCE_DATA_FILE = 'data/circle/experiments/{}_{}_data.h5'.format(args.run_identifier, TIMESTR)
with h5py.File(INFLUENCE_DATA_FILE, 'a') as f:
    f.create_dataset('influence', (0, 4), maxshape=(None, 4))

logger.info('*** Beginning experiment ***')

state_influence = np.empty((len(unique_states), 4))
for i, state_and_count in enumerate(zip(unique_states, unique_counts)):
    state, count = state_and_count
    logger.info('\n** Processing state {}/{}'.format(i+1, len(unique_states)))
    logger.info('Removed state: {}'.format(state))
    logger.debug('Instances of state counted: {}'.format(count))

    new_states, new_targets = utils.remove_state(state, training_states, training_targets)

    logger.info('States removed: {}'.format(len(training_states) - len(new_states)))

    temp_agent_model = models.build_basic_network(OBSERVATION_SIZE, ACTION_SIZE, LEARNING_RATE)
    temp_agent_model.load_weights(INIT_ORACLE_MODEL_FILE)
    temp_agent = DQN(OBSERVATION_SIZE, ACTION_SIZE, MEMORY_SIZE)
    temp_agent.model = temp_agent_model
    temp_agent.train_offline(new_states, new_targets)

    temp_agent_actions = utils.get_agent_actions(temp_agent, STATES)
    influence = utils.agent_accuracy(temp_agent_actions, ORACLE_ACTIONS)
    logger.info('State influence: {}'.format(influence))

    state_influence[i] = np.array((state[0], state[1], influence, count))

    if (i>0 and i%10==0) or i==9:
        logger.info('* Saving plot *')
        plt.clf()
        plt.gca().set_aspect('equal')
        graphs.plot_state_influence(state_influence[:i+1, 0:2], state_influence[:i+1, 2], title='State Influence')
        plt.savefig('data/circle/experiments/figures/{}_{}.png'.format(args.run_identifier, TIMESTR))

        logger.info('* Saving influence data *')
        with h5py.File(INFLUENCE_DATA_FILE, 'a') as f:
            data = f['influence']
            data_len = len(data)
            data.resize((i+1, 4))
            data[:i+1] = state_influence[:i+1]
