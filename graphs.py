import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def discrete_state_density_plot(states, ax=None, title=None):
    unique_states, counts = np.unique(states, axis=0, return_counts=True)
    unique_states_x = unique_states[:, 0]
    unique_states_y = unique_states[:, 1]
    
    goal = matplotlib.patches.Circle((0, 0), 1, fill=False, color='r', linewidth=1, linestyle='--')
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=200)
    ax = ax or plt.gca()
    if title: ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    ax.add_artist(goal)
    ax.scatter(unique_states_x, unique_states_y, c=counts, alpha=0.9, norm=normalize)
    return ax

def plot_states(states, ax=None, title=None):
    ax = ax or plt.gca()
    if title: ax.set_title(title)
    ax.set_aspect('equal')
    goal = matplotlib.patches.Circle((0, 0), 1, fill=False, color='r', linewidth=1, linestyle='--')
    ax.add_artist(goal)
    ax.scatter(states[:,0], states[:,1], c='b', marker='.') 
    return ax

def plot_state_influence(states, influence, fig=None, ax=None, title=None):
    fig = fig or plt.gcf()
    ax = ax or plt.gca()
    
    if title: ax.set_title(title)
    ax.set_aspect('equal')
    goal = matplotlib.patches.Circle((0, 0), 1, fill=False, color='r', linewidth=1, linestyle='--')
    ax.add_artist(goal)
    scatter = ax.scatter(states[:,0], states[:,1], c=influence, marker='.')
    colorbar = fig.colorbar(scatter, ax=ax) 
    return ax
 