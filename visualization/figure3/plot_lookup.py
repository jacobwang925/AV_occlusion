import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def transformation_v(vel):
    if type(vel) == int or type(vel) == float:
        return -vel * 2 + 20 + 0.5
    else:
        for i in range(len(vel)):
            vel[i] = -vel[i] * 2 + 20 + 0.5
        return vel

def transformation_x(pos):
    if type(pos) == int or type(pos) == float:
        return ((pos + 182) / 2 + 0.5)
    else:
        for i in range(len(pos)):
            pos[i] = (pos[i] + 182) / 2 + 0.5
        return pos

def smooth_data(stats, window_size, cutoff_point=8):
    smoothed_stats = []
    half_window = window_size // 2
    for i in range(cutoff_point):
        smoothed_stats.append(stats[i])

    for i in range(cutoff_point, len(stats) - cutoff_point):
        # Calculate the window range
        start = max(0, i - half_window)
        end = min(len(stats), i + half_window)
        
        # Average the values within the window
        window_average = sum(stats[start:end]) / (end-start)
        smoothed_stats.append(window_average)

    for i in range(len(stats)-cutoff_point, len(stats)):
        smoothed_stats.append(stats[i])
    
    return smoothed_stats

def plot_lookup_table(vmin, vmax, vdelta, xmin, xmax, xdelta, filename):
    # read data from csv file
    df = pd.read_csv(filename)
    init_pos = df['init_pos']
    init_speed = df['init_speed']
    safe = df['total_safe']
    unsafe = df['total_unsafe']

    df = df.astype(object)
    # prepare a dataframe for plot
    velocities = np.arange(vmin, vmax, vdelta)[::-1].tolist()
    positions = np.arange(xmin, xmax, xdelta).tolist()
    lookup = pd.DataFrame(1, index=velocities, columns=positions)
    lookup = lookup.astype(float)
    for i in range(len(init_pos)):
        if safe[i] + unsafe[i] == 0:
            F = 0
        else:
            F = safe[i] / (safe[i] + unsafe[i])
        lookup.loc[init_speed[i],init_pos[i]-82] = F


    # smooth out lookup table with Gaussian filter
    smoothed_lookup = gaussian_filter(lookup, sigma=0.7)
    
    # plot heatmap    
    ax = sns.heatmap(smoothed_lookup, vmin=0.4, vmax=1)
    # change the font size of color bar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    # specify ticks on x and y axis
    delta_y = 2
    y_label = np.arange(0, 11, delta_y)
    y_tick_label = ['0', '2', '4', '6', '8', '10']
    ax.set_yticks(transformation_v(y_label.tolist()))
    ax.set_yticklabels(y_tick_label, fontsize=12)

    delta_x = 40
    x_label = -np.arange(0, 183, delta_x)[::-1]
    x_tick_label = ['-160', '-120', '-80', '-40', '0']
    ax.set_xticks(transformation_x(x_label.tolist()))
    ax.set_xticklabels(x_tick_label, fontsize=12)


def plot_trajectory():
    # trajectory paths with initial states (x,v):
    # trajectory1: (-50,0)
    # trajectory2: (-10,9.5)
    paths = {"trajectory1", "trajectory2"}
    starting_points = {(-50.3,4.753877291818753e-10) ,(-8.5,9.76910400390625)}

    for path, point in zip(paths, starting_points):
        # actual
        with open(path + "/stats_velocity.txt", 'r') as file:
            key_v = [list(map(float, line.split())) for line in file]

        with open(path + "/stats_position.txt", 'r') as file:
            key_x = [list(map(float, line.split())) for line in file]

        count = 0
        for i in range(len(key_x[0])):
            if key_x[0][i] > 82:
                count = i
                break

        del key_v[0][count:]
        del key_x[0][count:]

        # transformation
        for i in range(len(key_x[0])):
            key_x[0][i] = key_x[0][i] - 82

        x = transformation_x(key_x[0])
        v = transformation_v(key_v[0])

        # smooth the first trajectory
        if path == "trajectory1":
            # smooth by section
            v[20:190] = smooth_data(stats=v[20:190], window_size=10, cutoff_point=0)
            v[190:255] = smooth_data(stats=v[190:255], window_size=15, cutoff_point=0)
            v[20:190] = smooth_data(stats=v[20:190], window_size=10, cutoff_point=0)
            v[190:255] = smooth_data(stats=v[190:255], window_size=12, cutoff_point=0)
            v[255:len(v)] = smooth_data(stats=v[255:len(v)], window_size=10, cutoff_point=0)
            
        # plot trajectory
        plt.plot(x, v, color='tab:blue', linewidth=1.5)
        # lot starting point
        x_start = transformation_x(point[0]-82)
        v_start = transformation_v(point[1])
        plt.plot(x_start,v_start,color='tab:blue', marker='o')

    # add legend of safe control trajectories
    plt.legend(['State Trajectory']) 


def plot(vmin, vmax, vdelta, xmin, xmax, xdelta, filename):
    plot_lookup_table(vmin, vmax, vdelta, xmin, xmax, xdelta, filename)
    plot_trajectory()

    plt.ylabel('Initial Velocity (m/s)', fontsize=14)
    plt.xlabel('Initial Position (m)', fontsize=14)
    plt.savefig("table_trajectory.pdf", format="pdf", bbox_inches='tight')
    plt.clf()


def main():
    vmin = 0
    vmax = 10.5
    vdelta = 0.5
    xmin = -182
    xmax = 0
    xdelta = 2
    filename = 'plot_lookup_table.csv'

    plot(vmin, vmax, vdelta, xmin, xmax, xdelta, filename)

if __name__ == '__main__':
    main()
