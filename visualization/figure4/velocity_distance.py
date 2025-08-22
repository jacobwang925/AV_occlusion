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

def plot_stats(safe_v, safe_position, risk_v, risk_position, transfuser_v, transfuser_position, pid_v, pid_position): # , transfuser_v, risk_v):
    with open(safe_v, 'r') as file:
        safe_v_stats = [list(map(float, line.split())) for line in file]
    
    with open(safe_position, 'r') as file:
        safe_pos = [list(map(float, line.split())) for line in file]
        safe_pos = convert_distance(safe_pos[0])

    with open(risk_v, 'r') as file:
        risk_v_stats = [list(map(float, line.split())) for line in file]
    
    with open(risk_position, 'r') as file:
        risk_pos = [list(map(float, line.split())) for line in file]
        risk_pos = convert_distance(risk_pos[0])
        
    with open(transfuser_v, 'r') as file:
        transfuser_v_stats = [list(map(float, line.split())) for line in file]
    
    with open(transfuser_position, 'r') as file:
        transfuser_pos = [list(map(float, line.split())) for line in file]
        transfuser_pos = convert_distance(transfuser_pos[0])

    with open(pid_v, 'r') as file:
        pid_v_stats = [list(map(float, line.split())) for line in file]
    
    with open(pid_position, 'r') as file:
        pid_pos = [list(map(float, line.split())) for line in file]
        pid_pos = convert_distance(pid_pos[0])



    fig, ax1 = plt.subplots()

    # PID
    smooth_pid_v = smooth_data(pid_v_stats[0], 12)
    smooth_pid_v = smooth_data(smooth_pid_v, 12)
    collision_point_pid = smooth_pid_v[-1]
    ax1.plot(pid_pos, smooth_pid_v, color='tab:blue', label='PID')
    ax1.plot(transfuser_pos[-1], collision_point_pid, markeredgecolor='tab:blue', markerfacecolor='none', markeredgewidth=2, marker='s', markersize=7)
    # ax1.plot(pid_pos[-1], collision_point_pid, color='tab:blue', marker='s', markersize=7)

    # transfuser
    smooth_transfuser_v = smooth_data(transfuser_v_stats[0], 12)
    smooth_transfuser_v = smooth_data(smooth_transfuser_v, 8, 1)
    collision_point_transfuser = smooth_transfuser_v[-1]
    ax1.plot(transfuser_pos, smooth_transfuser_v, color='tab:green', label='TransFuser')
    ax1.plot(transfuser_pos[-1], collision_point_transfuser, markeredgecolor='tab:green', markerfacecolor='none', markeredgewidth=2, marker='s', markersize=7)
    # ax1.plot(transfuser_pos[-1], collision_point_transfuser, color='tab:green', marker='s', markersize=7)
    
    # risk-based
    smooth_risk_v = smooth_data(risk_v_stats[0], 15)
    smooth_risk_v = smooth_data(smooth_risk_v, 12)
    safe_point_risk = smooth_risk_v[-1]
    ax1.plot(risk_pos, smooth_risk_v, color='tab:orange', label='Worst-Case')
    ax1.plot(risk_pos[-1], safe_point_risk, color='tab:orange', marker='^', markersize=8)
    
    # safe control
    smooth_safe_v = smooth_data(safe_v_stats[0], window_size=10)
    safe_point = smooth_safe_v[-1]
    ax1.plot(safe_pos, smooth_safe_v, color='tab:red', label='Proposed', linewidth='2')
    ax1.plot(safe_pos[-1], safe_point, color='tab:red', marker='^', markersize=8)
    
    # occlusion position
    plt.axvline(x=-7, color='k', linestyle='--', linewidth=2, label='Occlussion')
    
    # x & y label
    ax1.set_ylabel('Velocity (m/s)', fontsize=14)
    ax1.tick_params(labelsize=12)
    ax1.set_ylim(-0.1,7)
    ax1.tick_params(axis='y')
    plt.yticks(fontsize=12)
    ax1.set_xlabel('Distance (m)', fontsize=14)

    # legend & label
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3, 2, 1, 0, 4]
    plt.legend([handles[i] for i in order], [labels[i] for i in order], ncol=2, loc='upper left', fontsize=10)

    # plt.show()
    plt.savefig("velocity_distance.pdf", format="pdf", bbox_inches='tight')
    plt.clf()

    
if __name__ == '__main__':
    safe_v = 'safe_velocity.txt'
    safe_position = 'safe_position.txt'
    risk_v = 'risk_velocity.txt'
    risk_position = 'risk_position.txt'
    transfuser_v = 'transfuser_velocity.txt'
    transfuser_position = 'transfuser_position.txt'
    pid_v = 'pid_velocity.txt'
    pid_pos = 'pid_pos.txt'

    plot_stats(safe_v, safe_position, risk_v, risk_position, transfuser_v, transfuser_position, pid_v, pid_pos)

