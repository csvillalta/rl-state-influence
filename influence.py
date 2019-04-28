import gin
import h5py
import numpy as np
import pandas as pd
import time

import models
import utils

from dqn import DQN
from circle import CircleEnv

# Load configuration for DQN and model
gin.parse_config_file('configs/influence/influence.gin')

def influence(state, training_data, test_data, oracle_init_model, oracle_init_target_model, file_prefix):
    """Calculate the influence of a state with respect to the training data. Returns all influences for each state occurence."""
    # We drop duplicates, because a specific state at a specific episode and step can be reused several times
    state_occurences = training_data[(training_data['state_x'] == state['state_x']) & 
                                     (training_data['state_y'] == state['state_y'])].drop_duplicates()

    # Array to hold our state/influence pairs.
    state_influences = np.empty((len(state_occurences), 7))

    count = 0
    for _, state_occurence in state_occurences.iterrows():
        start_time = time.time()
        episode, step = state_occurence['episode'], state_occurence['step']
        
        occurences = len(training_data[(training_data.state_x == state_occurence.state_x) & 
                                  (training_data.state_y == state_occurence.state_y) & 
                                  (training_data.episode == episode) & 
                                  (training_data.step == step)])
        
        # Every state except those that occurs on or after the above step during the above episode.
        # theta_X/E
        full_trace = training_data[(training_data['episode'] != episode) | 
                          (training_data['step'] < step)]
        
        # Every state except those that occur after the above step during the above episode.
        # theta_X/F
        partial_trace = training_data[(training_data['episode'] != episode) | 
                             (training_data['step'] <= step)]

        # Setup our two agents to train on each of the filtered training sets above.
        ft_agent = DQN()
        ft_agent.model.load_weights(oracle_init_model)
        ft_agent.target_model.load_weights(oracle_init_target_model)     
        pt_agent = DQN()
        pt_agent.model.load_weights(oracle_init_model)
        pt_agent.target_model.load_weights(oracle_init_target_model)

        # Train our agents, get their optimal actions on testing data, and get consistencies.
        utils.train_agent_offline(ft_agent, full_trace.to_numpy())
        utils.train_agent_offline(pt_agent, partial_trace.to_numpy())
        ft_q_values = utils.get_q_values(ft_agent.model, training_data[['state_x', 'state_y']].drop_duplicates().to_numpy())
        pt_q_values = utils.get_q_values(pt_agent.model, training_data[['state_x', 'state_y']].drop_duplicates().to_numpy())
        ft_agent_actions = np.argmax(ft_q_values, axis=1)
        pt_agent_actions = np.argmax(pt_q_values, axis=1)
        ft_agent_acc = utils.agent_consistency(ft_agent_actions, test_data['action'].to_numpy())
        pt_agent_acc = utils.agent_consistency(pt_agent_actions, test_data['action'].to_numpy())
        
        # TODO: Carefully consider what we wish to have saved and how we name our save files...
        # Idea: state_x, state_y, episode, step, pt_agent_acc, ft_agent_acc, 
        state_influences[count] = np.array((state_occurence['state_x'], state_occurence['state_y'], episode, step, pt_agent_acc, ft_agent_acc, occurences), dtype=np.float64)
        count += 1
        print("Time elapsed for one loop iteration: {}".format(time.time()-start_time))
    
    data = pd.DataFrame(state_influences, columns=['state_x', 'state_y', 'episode', 'step', 'pt_agent_cons', 'ft_agent_cons', 'occurences'])
    if file_prefix:
        data.to_pickle('data/circle/experiments/influences_v1/infl_'+file_prefix+'.pkl')
    return data


def influence2(state, training_data, test_data, oracle_init_model, oracle_init_target_model, file_prefix):
    """Calculate the influence of a state with respect to the training data. Returns all influences for each state occurence."""
    # We drop duplicates, because a specific state at a specific episode and step can be reused several times
    state_occurences = training_data[(training_data['state_x'] == state['state_x']) & 
                                     (training_data['state_y'] == state['state_y'])].drop_duplicates()

    full_trace = training_data
    partial_trace = training_data
    for _, state_occurence in state_occurences.iterrows():
        start_time = time.time()
        episode, step = state_occurence['episode'], state_occurence['step']
        
        occurences = len(training_data[(training_data.state_x == state_occurence.state_x) & 
                                  (training_data.state_y == state_occurence.state_y) & 
                                  (training_data.episode == episode) & 
                                  (training_data.step == step)])
        
        # Every state except those that occur on or after the above step during the above episode.
        # theta_X/E
        full_trace = full_trace[(full_trace['episode'] != episode) | 
                          (full_trace['step'] < step)]
        
        # Every state except those that occur after the above step during the above episode.
        # theta_X/F
        partial_trace = partial_trace[(partial_trace['episode'] != episode) | 
                             (partial_trace['step'] <= step)]
    print('Traces removed.')

    # Setup our two agents to train on each of the filtered training sets above.
    ft_agent = DQN()
    ft_agent.model.load_weights(oracle_init_model)
    ft_agent.target_model.load_weights(oracle_init_target_model)     
    pt_agent = DQN()
    pt_agent.model.load_weights(oracle_init_model)
    pt_agent.target_model.load_weights(oracle_init_target_model)
    
    # Train our agents and get their q values for all the unique states in the training data
    print('Retraining agents.')
    training_start = time.time()
    utils.train_agent_offline(ft_agent, full_trace.to_numpy())
    print('Trained first agent in {} seconds.'.format(time.time() - training_start))
    training_start = time.time()
    utils.train_agent_offline(pt_agent, partial_trace.to_numpy())
    print('Trained second agent in {} seconds.'.format(time.time() - training_start))
    
    pt_q_values = utils.get_q_values(pt_agent.model, training_data[['state_x', 'state_y']].drop_duplicates().to_numpy())
    ft_q_values = utils.get_q_values(ft_agent.model, training_data[['state_x', 'state_y']].drop_duplicates().to_numpy())
    
    ft_agent.model.save('data/circle/experiments/models/'+file_prefix+'_ft_model.h5')
    pt_agent.model.save('data/circle/experiments/models/'+file_prefix+'_pt_model.h5')
    
    with h5py.File('data/circle/experiments/influences/infl_'+file_prefix+'.hdf5', 'w') as f:
        ptqv = f.create_dataset('pt_q_values', data=pt_q_values)
        ftqv = f.create_dataset('ft_q_values', data=ft_q_values)
        pt = f.create_dataset('pt', data=partial_trace.index.to_numpy())
        ft = f.create_dataset('ft', data=full_trace.index.to_numpy())