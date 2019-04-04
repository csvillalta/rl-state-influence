"""
    File name: train_circle.py
    Author: Christopher Villalta
    Desc: Script for training DQN agents on CircleEnv
"""

import argparse
import logging
import logging.config
import os
import gin
import numpy as np

import models

from circle import CircleEnv
from dqn import DQN

@gin.configurable
def explore(episodes, steps):
    if args.explore:
        for episode in range(episodes):
            state = env.reset()
            for step in range(steps):
                if args.render: env.render()
                action = env.sample()
                next_state, reward, done = env.step(action)
                agent.remember((state, action, reward, next_state, episode, step, done))
                state = next_state

                if done or step == steps-1:
                    break

@gin.configurable()
def train(episodes, steps):
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(steps):
            if args.render: env.render()
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            agent.remember((state, action, reward, next_state, episode, step, done))
            agent.replay()
            state = next_state

            if done or step == steps-1:
                logger.info("Episode: {}/{} | Score: {:.2f} | e: {:.2f}".format(episode+1, episodes, episode_reward, agent.epsilon*1.0))
                break

            if step % 10 == 0 and step > 0:
                if agent.epsilon >= agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay

                agent.update_target_model()

        if episode % 10 == 0 and episode > 0 and not args.no_save:
            agent.save_training_data(TRAINING_DATA_FILE)
            agent.model.save(MODEL_FILE)

    # Save final data and model
    if not args.no_save:
        agent.save_training_data(TRAINING_DATA_FILE)
        agent.model.save(MODEL_FILE)

@gin.configurable
def demo(episodes, steps):
    if args.demo:
        for episode in range(episodes):
            state = env.reset()
            start_state = state
            total_reward = 0
            for step in range(steps):
                env.render()
                action = agent.exploit(state)
                next_state, reward, done = env.step(action)
                total_reward += reward
                state = next_state

                if done or step == steps-1:
                    print("Episode: {}/{} | Score: {:.2f}".format(episode+1, episodes, total_reward))

# Setup argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('run_identifier', type=str)
parser.add_argument('--config', type=str, default='configs/training/basic_training.gin')
parser.add_argument('--explore', action='store_true')
parser.add_argument('--render', action='store_true')
parser.add_argument('--no-save', action='store_true')
parser.add_argument('--demo', action='store_true')
args = parser.parse_args()

# Parse configuration
gin.parse_config_file(args.config)

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('logs/circle/training/{}.log'.format(args.run_identifier))
file_handler.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

file_formatter = logging.Formatter('[%(asctime)s] %(message)s', "%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

TRAINING_DATA_FILE = 'data/circle/training_data/{}_training_data.h5'.format(args.run_identifier)
MODEL_FILE = 'data/circle/models/{}_model.h5'.format(args.run_identifier)

# Setup environment
env = CircleEnv()

# Setup agent
agent = DQN()
agent.model.save('data/circle/init_models/{}_init_model.h5'.format(args.run_identifier))
agent.target_model.save('data/circle/init_target_models/{}_init_target_model.h5'.format(args.run_identifier))

def main():
    if args.explore: explore()   
    train()
    if args.demo: demo()
        

if __name__ == '__main__':
    main()