import glob
import os
import sys
import argparse
import csv

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import numpy as np
from tqdm import tqdm

from risk_calc import count_words

def create(xmin, xmax, xdelta, vmin, vmax, vdelta, time_horizon, N, file_id):
    lookup = {}

    velocities = np.arange(vmin, vmax, vdelta)[::-1].tolist()
    positions = np.arange(xmin, xmax, xdelta).tolist()

    total_iteration = N * len(positions) * len(velocities)
    overall_progress = tqdm(total = total_iteration, desc="Overall loop", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]')
    for n in tqdm(range(N), desc="n loop", leave=False):
        for x in tqdm(positions, desc="position loop", leave=False):
            for v in tqdm(velocities, desc="velocity loop", leave=False):
                overall_progress.update(1)
                open("lookup.txt", "w").close()
                cmd = f"python cruise_control.py --time_horizon {time_horizon} --init_pos {x} --init_speed {v} --save_file lookup.txt"
                print(cmd)
                os.system(cmd)
                counts = count_words("lookup.txt", ['safe', 'unsafe'])
                if n == 0:
                    lookup[(x, v)] = [counts['safe'],counts['unsafe']]
                else:
                    lookup[(x, v)][0] += counts['safe']
                    lookup[(x, v)][1] += counts['unsafe']
                
    overall_progress.close()

    with open("risk_lookup_table"+str(file_id)+".csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(['init_pos', 'init_speed', 'total_safe', 'total_unsafe'])
        for key, value in lookup.items():
            writer.writerow([key[0], key[1], value[0],value[1]])


def main():
    parser = argparse.ArgumentParser(description='Range of state')
    parser.add_argument('--xmin', type=float, default=-100)
    parser.add_argument('--xmax', type=float, default=84)
    parser.add_argument('--xdelta', type=float, default=2)
    parser.add_argument('--vmin', type=float, default=0.5)
    parser.add_argument('--vmax', type=float, default=1)
    parser.add_argument('--vdelta', type=float, default=1)
    parser.add_argument('--time_horizon', type=float, default=200)
    parser.add_argument('--N', type=int, default=50)
    parser.add_argument('--file_id', type=str, default=0)

    args = parser.parse_args()
    
    xmin = args.xmin
    xmax = args.xmax
    xdelta = args.xdelta
    vmin = args.vmin
    vmax = args.vmax
    vdelta = args.vdelta
    time_horizon = args.time_horizon
    N = args.N
    file_id = args.file_id

    create(xmin, xmax, xdelta, vmin, vmax, vdelta, time_horizon, N, file_id)

if __name__ == '__main__':
    main()
