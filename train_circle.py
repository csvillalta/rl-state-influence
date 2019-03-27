"""
    File name: train_circle.py
    Author: Christopher Villalta
"""
import argparse
import logging
import os
import time
import numpy as np

import models

from circle import CircleEnv
from dqn import DQN

# Setup argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('run_identifier', type=str)
parser.add_argument('--explore', action='store_true')
parser.add_argument('--render', action='store_true')
parser.add_argument('--no-save', action='store_true')
parser.add_argument('--demo', action='store_true')
args = parser.parse_args()

TIMESTR = time.strftime('%Y%m%d-%H%M%S')

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('logs/circle/training/{}_{}.log'.format(args.run_identifier, TIMESTR))
file_handler.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

file_formatter = logging.Formatter('[%(asctime)s] %(message)s', "%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Parameters
EPISODES = 80
DEMO_EPISODES = 20
EXPLORE_EPISODES = 80
STEPS = 25
LEARNING_RATE = 0.001
MEMORY_SIZE = 2000
TRAINING_DATA_FILE = 'data/circle/training_data/{}_{}_training_data.h5'.format(args.run_identifier, TIMESTR)
MODEL_FILE = 'data/circle/models/{}_{}_model.h5'.format(args.run_identifier, TIMESTR)

# Setup environment
env = CircleEnv(continuous=True)
OBSERVATION_SIZE = env.observation_size
ACTION_SIZE = env.action_size

# Setup agent
model = models.build_basic_network(OBSERVATION_SIZE, ACTION_SIZE, LEARNING_RATE)
target_model = models.build_basic_network(OBSERVATION_SIZE, ACTION_SIZE, LEARNING_RATE)
if args.no_save:
    agent = DQN(OBSERVATION_SIZE, ACTION_SIZE, MEMORY_SIZE)
else:
    agent = DQN(OBSERVATION_SIZE, ACTION_SIZE, MEMORY_SIZE, TRAINING_DATA_FILE)
agent.model = model
agent.target_model = target_model
agent.save_model('data/circle/init_models/{}_{}_init_model.h5'.format(args.run_identifier, TIMESTR))

logger.info(" *** Beginning training ***")
logger.debug('='*30)
logger.debug("Total episodes: {}".format(EPISODES))
logger.debug("Steps per episode: {}".format(STEPS))
logger.debug("Learning rate: {}".format(LEARNING_RATE))
logger.debug("Memory size: {}".format(MEMORY_SIZE))
logger.debug('='*30)

if args.explore:
    logger.info("Beginning exploration episodes.")
    for episode in range(EXPLORE_EPISODES):
        state = env.reset()
        logger.info("Exploration episode {} @ ({:.2f}, {:.2f})".format(episode+1, state[0], state[1]))
        state = np.reshape(state, [1, OBSERVATION_SIZE])
        for step in range(STEPS):
            if args.render: env.render()
            action = agent.explore(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, OBSERVATION_SIZE])
            agent.remember(state, action, reward, next_state, episode, step, done)
            state = next_state

            if done or step == STEPS-1:
                break

logger.info("Beginning training episodes.")
for episode in range(EPISODES):
    state = env.reset()
    logger.debug('Beginning episode {} @ ({:.2f}, {:.2f})'.format(episode+1, env.state[0], env.state[1]))
    state = np.reshape(state, [1, OBSERVATION_SIZE])
    total_reward = 0
    for step in range(STEPS):
        if args.render: env.render()
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        next_state = np.reshape(next_state, [1, OBSERVATION_SIZE])
        agent.remember(state, action, reward, next_state, episode, step, done)
        state = next_state
        agent.replay()

        if done or step == STEPS-1:
            logger.info("Episode: {}/{}, Score: {:.2f}, e: {:.2f}".format(episode+1, EPISODES, total_reward, agent.epsilon*1.0))
            break

        if step % 10 == 0 and step > 0:
            logger.debug("Lowering agent epsilon")
            if agent.epsilon >= agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            logger.debug("Updating target model.")
            agent.update_target_model()


    if episode % 10 == 0 and episode > 0 and not args.no_save:
        logger.info("Saving model and training data.")
        agent.save_training_data()
        agent.save_model(MODEL_FILE)

if not args.no_save:
    logger.info("Saving final model and training data.")
    agent.save_training_data()
    agent.save_model(MODEL_FILE)

if args.demo:
    for episode in range(DEMO_EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, OBSERVATION_SIZE])
        start_state = state
        total_reward = 0
        for step in range(STEPS):
            env.render()
            action = agent.act(state, act_optimal=True)
            next_state, reward, done = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [1, OBSERVATION_SIZE])
            state = next_state

            if done or step == STEPS-1:
                print("Episode: {}/{}, Score: {:.2f}".format(episode+1, DEMO_EPISODES, total_reward))
