import numpy as np
import h5py

def agent_accuracy(predicted_actions, optimal_actions):
    return np.average(predicted_actions == optimal_actions)

def demo_agent(agent, env, episodes, steps, render=False):
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, (1, agent.observation_size))
        for step in range(steps):
            if render:
                env.render()
            action = agent.act(state, act_optimal=True)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, (1, agent.observation_size))
            state = next_state

            if done or step == steps-1:
                break

# TODO: generalize this function beyond circle environment
def generate_agent_actions(agent, n):
    """Takes an agent and generates n optimal actions to n randomly generated states."""
    states = np.random.uniform(low=-4, high=4, size=(n, 1, 2)) # each state needs to be 1x2 in order to input into a Keras model
    agent_actions = np.empty((n,))
    for i, state in enumerate(states):
        agent_actions[i] = agent.act(state, act_optimal=True)
    return states, agent_actions

def get_agent_actions(agent, states):
    """Returns optimal actions on a set of states for a given agent."""
    agent_actions = np.empty((len(states,)))
    for i, state in enumerate(states):
        agent_actions[i] = agent.act(state, act_optimal=True)
    return agent_actions
                
def load_data(data_file, data_type):
    data = None
    with h5py.File(data_file, 'r') as f:
        d = f[data_type]
        data = d[:]
    return data

# TODO: check this function
def remove_state(state, states, targets):
    state_indices = np.where((states == state).all(axis=1))
    return np.delete(states, state_indices, axis=0), np.delete(targets, state_indices, axis=0)
                
def train_agent(agent, env, episodes, explore_episodes, steps, model_file, training_data_file, render=False):
    if explore_episodes:
        for episode in range(explore_episodes):
            state = env.reset()
            state = np.reshape(state, [1, env.observation_size])
            for step in range(steps):
                if render: env.render()
                action = agent.explore(state)
                next_state, reward, done = env.step(action)
                next_state = np.reshape(next_state, [1, env.observation_size])
                agent.remember(state, action, reward, next_state, episode, step, done)
                state = next_state

                if done or step == steps-1:
                    break

    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_size])
        total_reward = 0
        for step in range(steps):
            if render: env.render()
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [1, env.observation_size])
            agent.remember(state, action, reward, next_state, episode, step, done)
            state = next_state
            agent.replay()

            if done or step == steps-1:
                print("Episode: {}/{}, Score: {:.2f}, e: {:.2f}".format(episode+1, episodes, total_reward, agent.epsilon*1.0))
                break
            if step % 50 == 0:
                agent.update_target_model()

        if episode % 10 == 0 and episode > 0:
            if training_data_file: agent.save_training_data()
            if model_file: agent.save_model(model_file)

    if training_data_file: agent.save_training_data()
    if model_file: agent.save_model(model_file)
