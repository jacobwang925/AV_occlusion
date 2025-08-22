import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter

import pykalman
from pykalman import KalmanFilter

from scipy.interpolate import interp1d


# smoothing methods
def smooth_data(stats, window_size, cutoff_point = 8):
    smoothed_stats = []
    half_window = window_size // 2
    for i in range(cutoff_point):
        smoothed_stats.append(stats[i])

    for i in range(cutoff_point, len(stats)-cutoff_point):
        # Calculate the window range
        start = max(0, i - half_window)
        end = min(len(stats), i + half_window)
        
        # Average the values within the window
        window_average = sum(stats[start:end]) / window_size
        smoothed_stats.append(window_average)
    for i in range(len(stats)-cutoff_point, len(stats)):
        smoothed_stats.append(stats[i])
    
    return smoothed_stats

def convert_distance(position):
    return [pos - 82 for pos in position]

def plot_velocity(folders, labels):
    fig, ax1 = plt.subplots()

    for folder, label in zip(folders, labels):
        with open(folder + '/safe_position.txt', 'r') as file:
            pos = [list(map(float, line.split())) for line in file]
        pos = convert_distance(pos[0])
        
        with open(folder + '/safe_velocity.txt', 'r') as file:
            velocity = [list(map(float, line.split())) for line in file]
        smoothed_velocity = smooth_data(velocity[0], 18)

        # draw curve
        line, = ax1.plot(pos, smoothed_velocity, label=label)

        # use the same color as the curve
        ax1.plot(pos[-1], smoothed_velocity[-1], 
                 marker='^', markersize=8, 
                 color=line.get_color())

    plt.axvline(x=-7, color='k', linestyle='--', linewidth=2, label='Occlussion')

    ax1.set_ylabel('Velocity (m/s)', fontsize=14)
    ax1.tick_params(labelsize=12)
    ax1.set_ylim(-0.1,7.5)
    ax1.tick_params(axis='y')
    plt.yticks(fontsize=12)
    ax1.set_xlabel('Distance (m)', fontsize=14)

    # legend & label
    plt.legend()
    plt.savefig("new_distribution_velocity.pdf", format="pdf", bbox_inches='tight')
    plt.clf()



def plot_control(folders, labels):
    fig, ax1 = plt.subplots()

    for folder, label in zip(folders, labels):
        with open(folder + '/safe_u.txt', 'r') as file:
            control = [list(map(float, line.split())) for line in file]

        
        
        with open(folder + '/safe_position.txt', 'r') as file:
            pos = [list(map(float, line.split())) for line in file]
        pos = convert_distance(pos[0])

        # ==========================
        # without smoothing
        control = control[0]
        # draw curve
        line, = ax1.plot(pos, control, label=label)
        # ==========================

        # ==========================
        # with smoothing
        smoothed_control = smooth_data(stats=control[0], window_size=18, cutoff_point=0)
        # draw curve
        line, = ax1.plot(pos, smoothed_control, label=label)
        # ==========================


    plt.axvline(x=-7, color='k', linestyle='--', linewidth=2, label='Occlussion')

    ax1.set_ylabel('Control', fontsize=14)
    ax1.tick_params(labelsize=12)
    ax1.set_ylim(-1.5,2)
    ax1.tick_params(axis='y')
    plt.yticks(fontsize=12)
    ax1.set_xlabel('Distance (m)', fontsize=14)

    # legend & label
    plt.legend()
    plt.savefig("new_distribution_control.pdf", format="pdf", bbox_inches='tight')
    plt.clf()

    
if __name__ == '__main__':
    folders = ['old_distribution', 'new_distribution', 'occurence_0']
    labels = ['Distribution 1 (Section V-A)', 'Distribution 2', 'Distribution 3']

    plot_velocity(folders, labels)
    plot_control(folders, labels)