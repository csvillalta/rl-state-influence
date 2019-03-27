import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_state_density(states, ax=None, title=None):
    ax = ax or plt.gca()
    ax.set_aspect('equal')
    if title: ax.set_title(title)
    goal = matplotlib.patches.Circle((0,0), 1, fill=False, color='r', linewidth=2, linestyle='--')
    ax.add_artist(goal)
    # hist = ax2.hist2d(states[:,0], states[:,1], bins=100, cmap='Reds')
    ax.scatter(states[:,0], states[:,1], alpha=0.0075, marker='o', c='green')
    return ax
    
def plot_state_frequency(states, ax=None, title=None):
    unique_states, counts = np.unique(states, axis=0, return_counts=True)
    ax = ax or plt.gca()
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if title: ax.set_title(title)
    goal = matplotlib.patches.Circle((0,0), 1, fill=False, color='r', linewidth=2, linestyle='--')
    ax.add_artist(goal)
    scatter = ax.scatter(unique_states[:,0], unique_states[:,1], c=counts, cmap='Reds', marker='.')
    ax.figure.colorbar(scatter, cax=cax, fraction=0.046, pad=0.04)
    return ax
    
def plot_states(states, ax=None, title=None):
    ax = ax or plt.gca()
    ax.set_aspect('equal')
    if title: ax.set_title(title)
    goal = matplotlib.patches.Circle((0, 0), 1, fill=False, color='r', linewidth=2, linestyle='--')
    ax.add_artist(goal)
    ax.scatter(states[:,0], states[:,1], c='b', marker='.') 
    return ax

def plot_state_influence(states, influence, ax=None, title=None):
    ax = ax or plt.gca()
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if title: ax.set_title(title)
    goal = matplotlib.patches.Circle((0, 0), 1, fill=False, color='black', linewidth=2, linestyle='--')
    ax.add_artist(goal)
    scatter = ax.scatter(states[:,0], states[:,1], c=influence, marker='.', cmap='seismic')
    ax.figure.colorbar(scatter, cax=cax, fraction=0.046, pad=0.04) 
    return ax
 