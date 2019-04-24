import gin
import numpy as np
import pandas as pd

from replay_memory import ReplayMemory
from utils import df_empty

# TODO: Modify agent so it is not tailored specifically for CircleEnv...

@gin.configurable
class DQN(object):
    """Deep Q Network implementation for usage with CircleEnv.

    Attributes:
        observation_size: Size of the observation space.
        action_size: Size of the action space.
        model: The Q function approximator.
        target_model: Seperate target function approximator for fixed TD targets.
        memory_size: Replay memory capacity.
        memory: Experience replay memory buffer.
        batch_size: Size of minibatches used in training.
        gamma: Discount rate for return.
        epsilon: Random action probability.
        epsilon_min: Minimum random action probability.
        epsilon_decay: Decay rate for random action probability.
    """

    def __init__(
        self, 
        observation_size, 
        action_size,
        model,
        target_model,
        memory_size, 
        batch_size, 
        gamma, 
        epsilon, 
        epsilon_min, 
        epsilon_decay):
        
        self.observation_size = observation_size
        self.action_size = action_size
        self.model = model
        self.target_model = target_model
        self.memory_size = memory_size
        self.memory = ReplayMemory(size=self.memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.training_data = []

    def act(self, state):
        """Choose an action based on an epsilon greedy policy."""
        state = np.reshape(state, (1, self.observation_size)) # Must reshape state to feed into Keras model.
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def exploit(self, state):
        state = np.reshape(state, (1, self.observation_size))
        q_values = self.model.predict(state)
        return int(np.argmax(q_values[0]))

    def replay(self):
        if len(self.memory) >= self.batch_size:
            minibatch = self.memory.sample(self.batch_size)
            for experience in minibatch:
                state, action, reward, next_state, episode, step, done = experience
                state = np.reshape(state, (1, self.observation_size))
                next_state = np.reshape(next_state, (1, self.observation_size))
                target = reward
                if not done:
                    target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])

                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
                self.training_data.append((state[0][0], 
                                           state[0][1], 
                                           action, 
                                           reward, 
                                           next_state[0][0], 
                                           next_state[0][1], 
                                           episode, 
                                           step, 
                                           done))

    def remember(self, experience):
        """Stores experiences in experience replay buffer."""
        self.memory.append(experience)
        
    def save_training_data(self, file_name):
        data = pd.DataFrame(self.training_data, columns=['state_x', 'state_y', 'action', 'reward', 'next_state_x', 'next_state_y', 'episode', 'step', 'done'])
        data.to_pickle(file_name)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def replay_offline(self, experience):
        state_x, state_y, action, reward, next_state_x, next_state_y, episode, step, done = experience
        state = np.array([[state_x, state_y]]) # Want this to be of shape (1, 2) to feed into Keras model.
        next_state = np.array([[next_state_x, next_state_y]]) # Want this to be of shape (1, 2) to feed into Keras model.
        action = int(action) # Action is turned into a float when saved to disk, so we must convert to int.
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
        