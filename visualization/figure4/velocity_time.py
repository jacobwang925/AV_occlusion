import numpy as np
import matplotlib.pyplot as plt

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

def smooth_by_boundary(stats, window_size):
    smoothed_stats = np.zeros_like(stats)
    
    for i in range(len(stats)-window_size):
        window = stats[i:i+window_size]
        window_max = min(window)
        window_mean = np.mean(window)
        if window[0] == window_max:
            smoothed_stats[i] = window_max
        elif i > window_size:
            previous_window = stats[i-window_size:i]
            previous_max = min(previous_window)
            next_window = stats[i+window_size:i+2*window_size]
            next_max = min(next_window)
            if window_max == previous_max or window_max == next_max:
                smoothed_stats[i] = window_max
            if window_max != 0 and previous_max != 0:
                smoothed_stats[i] = (window_max + previous_max) / 2
        else:
            smoothed_stats[i] = window_mean

    
    return smoothed_stats

def smooth_interpolation(stats, kind='linear'):
    x = np.arange(len(stats))  # Create x-coordinates corresponding to each data point
    f = interp1d(x, stats, kind=kind, fill_value="extrapolate")  # Create interpolation function
    
    # Use interpolation to create a smoothed curve over a finer grid
    x_new = np.linspace(0, len(stats)-1, num=len(stats)*20)  # Finer x-coordinates
    smoothed_data = f(x_new)
    
    return x_new, smoothed_data

def plot_stats(safe_v, pid_v, transfuser_v, risk_v):
    with open(safe_v, 'r') as file:
        safe_v_stats = [list(map(float, line.split())) for line in file]

    
    with open(pid_v, 'r') as file:
        pid_v_stats = [list(map(float, line.split())) for line in file]

    with open(transfuser_v, 'r') as file:
        transfuser_v_stats = [list(map(float, line.split())) for line in file]

    with open(risk_v, 'r') as file:
        risk_v_stats = [list(map(float, line.split())) for line in file]
    
    

    # Velocity
    fig, ax1 = plt.subplots()

    # PID
    smooth_pid_v = smooth_data(pid_v_stats[0], 12)
    smooth_pid_v = smooth_data(smooth_pid_v, 12)
    collision_point_pid = smooth_pid_v[-1]

    ax1.plot(smooth_pid_v, color='tab:blue', label='PID')
    ax1.plot(len(smooth_pid_v), collision_point_pid, color='tab:blue', marker='s', markersize=7)
    
    # transfuser
    smooth_transfuser_v = smooth_data(transfuser_v_stats[0], 12)
    smooth_transfuser_v = smooth_data(smooth_transfuser_v, 8, 1)
    collision_point_transfuser = smooth_transfuser_v[-1]

    ax1.plot(smooth_transfuser_v, color='tab:green', label='TransFuser')
    ax1.plot(len(smooth_transfuser_v), collision_point_transfuser, color='tab:green', marker='s', markersize=7)
    
    # risk-based
    smooth_risk_v = smooth_data(risk_v_stats[0], 15)
    smooth_risk_v = smooth_data(smooth_risk_v, 12)
    safe_point_risk = smooth_risk_v[-1]
    ax1.plot(smooth_risk_v, color='tab:orange', label='Risk-Based')
    ax1.plot(len(smooth_risk_v), safe_point_risk, color='tab:orange', marker='^', markersize=8)

    # safe control
    smooth_safe_v = smooth_data(safe_v_stats[0], window_size=10)
    safe_point = smooth_safe_v[-1]
    ax1.plot(smooth_safe_v, color='tab:red', label='Proposed', linewidth='2')
    ax1.plot(len(smooth_safe_v), safe_point, color='tab:red', marker='^', markersize=8)
    


    ax1.set_ylabel('Velocity (m/s)', fontsize=14)
    ax1.tick_params(labelsize=12)
    ax1.set_ylim(-0.1,7.5)
    ax1.tick_params(axis='y')

    # legend & label

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3, 2, 1, 0]
    plt.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper right', fontsize=12)

    # x axis
    delta_t = 1 * 10 / 0.05
    t_label = np.arange(0, len(transfuser_v_stats[0]), delta_t)

    t_tick_label = []
    for t in t_label:
        t_tick_label.append(str(int(t*0.05)))
    ax1.set_xticks(t_label)
    ax1.set_xticklabels(t_tick_label, fontsize=12)
    ax1.set_xlabel('Time (s)', fontsize=14)
    plt.yticks(fontsize=12)
    plt.savefig("velocity_time.pdf", format="pdf", bbox_inches='tight')
    plt.clf()

    
if __name__ == '__main__':
    safe_v = 'safe_velocity.txt'
    pid_v = 'pid_velocity.txt'
    transfuser_v = 'transfuser_velocity.txt'
    risk_v = 'risk_based_velocity.txt'
    plot_stats(safe_v, pid_v, transfuser_v, risk_v)
