from dqn import DQN
import logging
import numpy as np
import gym

EPISODES = 6
STEPS = 2
LEARNING_RATE = 0.001
MEMORY_SIZE = 2000
TRAINING_DATA_FILE = 'data/cartpole/cartpole_minimal_example_training_data.h5'
MODEL_FILE = 'data/cartpole/cartpole_minimal_example_model.h5'

env = gym.make('CartPole-v0')
OBSERVATION_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.n

agent = DQN(OBSERVATION_SIZE, ACTION_SIZE, LEARNING_RATE, MEMORY_SIZE, training_dataset_file=TRAINING_DATA_FILE)

for episode in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, OBSERVATION_SIZE])
    total_reward = 0
    for step in range(STEPS):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        next_state = np.reshape(next_state, [1, OBSERVATION_SIZE])
        agent.remember(state, action, reward, next_state, episode, step, done)
        state = next_state

        if done or step == STEPS-1:
            print("Episode: {}/{}, Score: {}, e: {:.2}".format(episode+1, EPISODES, total_reward, agent.epsilon*1.0))
            break

        agent.replay()
        if step % 100 == 0:
            agent.update_target_model()

    if episode % 10 == 0 and episode > 0:
        print("Saving model and training data")
        agent.save_training_data(TRAINING_DATA_FILE)
        agent.save_model(MODEL_FILE)

print("Saving final model and training data")
agent.save_training_data(TRAINING_DATA_FILE)
agent.save_model(MODEL_FILE)
