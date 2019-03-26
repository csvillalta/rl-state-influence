import os
import random

import h5py
import numpy as np

from replay_memory import ReplayMemory

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class DQN(object):
    """Deep Q Network implementation.

    This DQN implementation features the ability to save the minibatch data utilized for training the neural network.

    Attributes:
        observation_size: Size of the observation space.
        action_size: Size of the action space.
        model: The q function approximator.
        target_model: Seperate target function approximator for fixed TD targets.
        memory_size: Capacity of replay memory
        memory: Experience replay memory buffer
        training_data: Buffer of all utilized training data.
        replay_batch_size: Size of minibatches used in training.
        gamma: Discount rate for return.
        epsilon: Random action probability.
        epsilon_min: Minimum random action probability.
        epsilon_decay: Decay rate for random action probability.
        self.training_index: Index used to keep track of where we are in our hdf5.
    """

    def __init__(self, observation_size, action_size, memory_size=1000000, training_data_file=None):
        self.observation_size = observation_size
        self.action_size = action_size
        self.model = None
        self.target_model = None
        self.memory_size = memory_size
        self.memory = ReplayMemory(size=self.memory_size)
        self.training_data = []
        self.training_data_full = []
        self.replay_batch_size = 20
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.training_index = 0
        self.training_data_file = training_data_file
        
        if training_data_file:
            self._init_training_data(training_data_file)

    # TODO: resolve issue with having to delete data files with the same name
    def _init_training_data(self, file):
        with h5py.File(file, 'a') as f:
            dset = f.create_dataset('training', (0, 8),  maxshape=(None, 8))

    def remember(self, state, action, reward, next_state, episode, step, done):
        """Stores experiences in experience replay buffer."""
        self.memory.append((state, action, reward, next_state, episode, step, done))

    def act(self, state, epsilon=None, act_optimal=False):
        """Epsilon greedy policy."""
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.rand() <= epsilon and not act_optimal:
            return random.randrange(self.action_size)
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def explore(self, state):
        """Act randomly to explore state space."""
        return random.randrange(self.action_size)

        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self):
        """Performs a minibatch training procedure using experience replay buffer. Utilizes target network for fixed Q-targets."""
        if len(self.memory) >= self.replay_batch_size:
            minibatch = self.memory.sample(self.replay_batch_size)
            for i, experience in enumerate(minibatch):
                state, action, reward, next_state, episode, step, done = experience
                target = reward
                if not done:
                    target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])

                # TODO: revise training data storage
                # combine the state, target value, episode and step into a single numpy array for storage
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
                training_datum = np.hstack((state, target_f))
                training_datum = np.ravel(training_datum)
                training_datum = np.hstack((training_datum, np.array([episode, step])))
                # currently storing training data as (state_x, state_y, target_a1, target_a2, target_a3, target_a4, episode, step)
                self.training_data.append(training_datum)
                self.training_data_full.append(training_datum)
#             if self.epsilon > self.epsilon_min:
#                 self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_training_data(self):
        """Incrementally saves experiences (states and rewards) used to train our model."""
        with h5py.File(self.training_data_file, 'a') as f:
            dset = f['training']
            # TODO: address magic number 8
            dset.resize((len(dset)+len(self.training_data), 8))
            dset[self.training_index: self.training_index+len(self.training_data)] = np.array(self.training_data)
        self.training_index += len(self.training_data)
        self.training_data = []

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = load_model(model_path)

    def train_offline(self, states, targets):
        for state, target in zip(states, targets):
            state = np.reshape(state, (1, self.observation_size))
            target = np.reshape(target, (1, self.action_size))
            self.model.fit(state, target, epochs=1, verbose=0)
