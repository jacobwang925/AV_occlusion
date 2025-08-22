import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter

# import pykalman
# from pykalman import KalmanFilter

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

def plot_stats(velocity, position, ax1):
    with open(velocity, 'r') as file:
        v_status = [list(map(float, line.split())) for line in file]
    
    with open(position, 'r') as file:
        pos_status = [list(map(float, line.split())) for line in file]
        pos_status = convert_distance(pos_status[0])
    
    # ax1.plot(pos_status, v_status[0])
    
    smoothed_v = smooth_data(v_status[0], 12)
    smoothed_v = smooth_data(smoothed_v, 12)
    ax1.plot(pos_status, smoothed_v)
    # smooth_pid_v = smooth_data(smooth_pid_v, 12)



if __name__ == '__main__':
    alpha_list = [0.05, 0.1, 0.2, 0.5, 1]
    fig, ax1 = plt.subplots()

    for alpha in alpha_list:
        v_path = 'alpha_' + str(alpha) + '/safe_velocity.txt'
        pos_path = 'alpha_' + str(alpha) + '/safe_position.txt'
        plot_stats(v_path, pos_path, ax1)

    # occlusion position
    plt.axvline(x=-7, color='k', linestyle='--', linewidth=2, label='Occlussion')
    
    # x & y label
    ax1.set_ylabel('Velocity (m/s)', fontsize=14)
    ax1.tick_params(labelsize=12)
    ax1.set_ylim(-0.1,6)
    ax1.tick_params(axis='y')
    plt.yticks(fontsize=12)
    ax1.set_xlabel('Distance (m)', fontsize=14)

    # legend & label
    plt.legend(alpha_list, loc='upper left', fontsize=10)

    # plt.show()
    plt.savefig("alpha.png")
    plt.clf()
