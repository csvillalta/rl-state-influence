import numpy as np
import pandas as pd

def agent_consistency(predicted_actions, optimal_actions):
    consistency = np.mean(np.array(predicted_actions) == np.array(optimal_actions))
    return consistency

def demo_agent(agent, env, episodes, steps):
    for episode in range(episodes):
        state = env.reset()
        for step in range(steps):
            env.render()
            action = agent.exploit(state)
            next_state, reward, done = env.step(action)
            state = next_state
            
            if done or step==step-1:
                break
    env.close()

# TODO: Generalize this function beyond circle environment.
# TODO: Optimize? It's only used once.
def generate_agent_actions(agent, n):
    """Takes an agent and generates n optimal actions to n randomly generated states."""
    start_coordinate = np.arange(-4, 4.1, 0.25)
    states = np.random.choice(start_coordinate, size=(n, 2))
    data = df_empty(['state_x', 'state_y', 'action'], [np.float16, np.float16, np.int8])
    for state in states:
        action = agent.exploit(state)
        datum = pd.Series({'state_x': state[0], 
                            'state_y': state[1], 
                            'action': action})
        data = data.append(datum, ignore_index=True)
    return data

# TODO: Should look to vectorize this operation.
def get_agent_actions(agent, states):
    """Returns optimal actions on a set of states for a given agent."""
    actions = np.empty(len(states))
    for i, state in enumerate(states):
        actions[i] = agent.exploit(state)
    return actions

def get_q_values(model, states):
    return model.predict(states)

def df_empty(columns, dtypes, index=None):
    """Creates an empty DataFrame with initial column names and types."""
    assert len(columns)==len(dtypes)
    df = pd.DataFrame(index=index)
    for c, d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df

# TODO: Make this function more readable; remove magic numbers etc.
def train_agent_offline(agent, training_data):
    # Mimic minibatch training
    batches = int(np.ceil(len(training_data)/20))
    step = 0 # Start at 18 because training starts at 19 and we increment below.
    # Update target model because it gets updated at step 10.
    for i in range(batches):
        start = i*20
        end = (i+1)*20
        batch = training_data[start: end]
        for j, experience in enumerate(batch):
            agent.replay_offline(experience)
        
        if step%10==0 and step>0:
            agent.update_target_model()
            
        if step<24:
            step += 1
        else:
            step = 0