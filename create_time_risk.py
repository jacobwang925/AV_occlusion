import glob
import os
import sys
import queue
# import matplotlib.pyplot as plt
# import argparse
import csv

sys.path.append(os.path.abspath('../carla/agents/navigation'))
import controller
  
# reset sys.path
sys.path = sys.path[:-1]

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import time
import numpy as np
import cv2
# import seaborn as sns
from statistics import mean
import pandas as pd
from tqdm import tqdm

import pid_control as carlaPid
from risk_calc import count_words

def average_time(ticks):
    sum = 0
    for i in range(len(ticks)):
        sum = sum + ticks[i][0]
    return (sum/len(ticks))

def create_safe_control(initial_state, tolerance, alpha, N):
    safe_control_table = {}

    total_iteration = N * len(tolerance) * len(initial_state) * len(alpha)
    overall_progress = tqdm(total = total_iteration, desc="Overall loop", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]')

    for state in tqdm(initial_state, desc="position loop", leave=False):
        for epsilon in tqdm(tolerance, desc="position loop", leave=False):
            for a in tqdm(alpha, desc="position loop", leave=False):
                x, v = state[0], state[1]
                open("safe_control.txt", "w").close()
                open("safe_control_time.txt", "w").close()

                for n in tqdm(range(N), desc="n loop", leave=False):
                    overall_progress.update(1)
                    # run safe control N times
                    cmd = f"python safe_control.py --init_pos {x} --init_speed {v} --epsilon {epsilon} --alpha {a} --save_time=True"
                    print(cmd)
                    os.system(cmd)

                # calculate safety probability
                counts = count_words("safe_control.txt", ['safe', 'unsafe'])
                total = sum(counts.values())
                if total == 0:
                    F = counts['safe']/N
                else:
                    F = counts['safe']/total

                with open('safe_control_time.txt', 'r') as file:
                    ticks = [list(map(float, line.split())) for line in file]

                safe_control_table[(x, v, epsilon, a)] = [counts['safe'],counts['unsafe'], F, average_time(ticks)]

    # with open("safe_control_risk.csv", "w") as file:
    with open("safe_control_test_alpha.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(['init_pos', 'init_speed', 'tolerance', 'alpha', 'total_safe', 'total_unsafe', 'safety_probability','average_time_horizon'])
        for key, value in safe_control_table.items():
            writer.writerow([key[0], key[1], key[2], key[3], value[0], value[1], value[2], value[3]])


def create_cruise_control(initial_state, N):
    cruise_control_table = {}

    total_iteration = N * len(initial_state)
    overall_progress = tqdm(total = total_iteration, desc="Overall loop", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]')

    for state in tqdm(initial_state, desc="position loop", leave=False):
        x, v = state[0], state[1]
        open("cruise_control.txt", "w").close()
        open("cruise_control_time.txt", "w").close()
        for n in tqdm(range(N), desc="n loop", leave=False):
            overall_progress.update(1)
            # run cruise control N times
            cmd = f"python pid_control.py --init_pos {x} --init_speed {v} --save_time=True"
            print(cmd)
            os.system(cmd)

        # calculate safety probability
        counts = count_words("cruise_control.txt", ['safe', 'unsafe'])
        total = sum(counts.values())
        if total == 0:
            F = counts['safe']/N
        else:
            F = counts['safe']/total
        # calculate average time horizon
        with open('cruise_control_time.txt', 'r') as file:
            ticks = [list(map(float, line.split())) for line in file]
            
        cruise_control_table[(x, v)] = [counts['safe'],counts['unsafe'], F, average_time(ticks)]

    with open("cruise_control_risk.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(['init_pos', 'init_speed', 'total_safe', 'total_unsafe', 'safety_probability', 'average_time_horizon'])
        for key, value in cruise_control_table.items():
            writer.writerow([key[0], key[1], value[0], value[1], value[2], value[3]])




def create_baseline_control(initial_state, tolerance, N):
    safe_control_table = {}

    total_iteration = N * len(tolerance) * len(initial_state)
    overall_progress = tqdm(total = total_iteration, desc="Overall loop", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]')

    for state in tqdm(initial_state, desc="position loop", leave=False):
        for epsilon in tqdm(tolerance, desc="position loop", leave=False):
            x, v = state[0], state[1]
            open("risk_control.txt", "w").close()
            open("risk_control_time.txt", "w").close()

            for n in tqdm(range(N), desc="n loop", leave=False):
                overall_progress.update(1)
                # run safe control N times
                cmd = f"python risk_based_control.py --init_pos {x} --init_speed {v} --epsilon {epsilon} --save_time=True"
                print(cmd)
                os.system(cmd)

            # calculate safety probability
            counts = count_words("risk_control.txt", ['safe', 'unsafe'])
            total = sum(counts.values())
            if total == 0:
                F = counts['safe']/N
            else:
                F = counts['safe']/total

            with open('risk_control_time.txt', 'r') as file:
                ticks = [list(map(float, line.split())) for line in file]

            safe_control_table[(x, v, epsilon)] = [counts['safe'],counts['unsafe'], F, average_time(ticks)]

    with open("risk_control_risk.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(['init_pos', 'init_speed', 'tolerance', 'total_safe', 'total_unsafe', 'safety_probability','average_time_horizon'])
        for key, value in safe_control_table.items():
            writer.writerow([key[0], key[1], key[2], value[0], value[1], value[2], value[3]])


def create_transfuser_control(initial_state, N):
    transfuser_control_table = {}

    total_iteration = N * len(initial_state)
    overall_progress = tqdm(total = total_iteration, desc="Overall loop", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]')

    for state in tqdm(initial_state, desc="position loop", leave=False):
        x, v = state[0], state[1]
        open("transfuser_control.txt", "w").close()
        open("transfuser_control_time.txt", "w").close()
        for n in tqdm(range(N), desc="n loop", leave=False):
            overall_progress.update(1)
            # run cruise control N times
            cmd = f"python transfuser_control.py --init_pos {x} --init_speed {v} --save_time=True"
            print(cmd)
            os.system(cmd)

        # calculate safety probability
        counts = count_words("transfuser_control.txt", ['safe', 'unsafe'])
        total = sum(counts.values())
        if total == 0:
            F = counts['safe']/N
        else:
            F = counts['safe']/total
        # calculate average time horizon
        with open('transfuser_control_time.txt', 'r') as file:
            ticks = [list(map(float, line.split())) for line in file]
            
        transfuser_control_table[(x, v)] = [counts['safe'],counts['unsafe'], F, average_time(ticks)]

    with open("transfuser_control_risk.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(['init_pos', 'init_speed', 'total_safe', 'total_unsafe', 'safety_probability', 'average_time_horizon'])
        for key, value in transfuser_control_table.items():
            writer.writerow([key[0], key[1], value[0], value[1], value[2], value[3]])


def create_mpc_control(initial_state, N):
    mpc_control_table = {}

    total_iteration = N * len(initial_state)
    overall_progress = tqdm(total = total_iteration, desc="Overall loop", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]')

    for state in tqdm(initial_state, desc="position loop", leave=False):
        x, v = state[0], state[1]
        open("mpc_control.txt", "w").close()
        open("mpc_control_time.txt", "w").close()
        for n in tqdm(range(N), desc="n loop", leave=False):
            overall_progress.update(1)
            # run cruise control N times
            cmd = f"python oa_mpc_control.py --init_pos {x} --init_speed {v} --save_time=True"
            print(cmd)
            os.system(cmd)

        # calculate safety probability
        counts = count_words("mpc_control.txt", ['safe', 'unsafe'])
        total = sum(counts.values())
        if total == 0:
            F = counts['safe']/N
        else:
            F = counts['safe']/total
        # calculate average time horizon
        with open('mpc_control_time.txt', 'r') as file:
            ticks = [list(map(float, line.split())) for line in file]
            
        mpc_control_table[(x, v)] = [counts['safe'],counts['unsafe'], F, average_time(ticks)]

    with open("mpc_control_risk.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(['init_pos', 'init_speed', 'total_safe', 'total_unsafe', 'safety_probability', 'average_time_horizon'])
        for key, value in mpc_control_table.items():
            writer.writerow([key[0], key[1], value[0], value[1], value[2], value[3]])

def create_planning_based_control(initial_state, N):
    planning_control_table = {}

    total_iteration = N * len(initial_state)
    overall_progress = tqdm(total = total_iteration, desc="Overall loop", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]')

    for state in tqdm(initial_state, desc="position loop", leave=False):
        x, v = state[0], state[1]
        open("planning_control.txt", "w").close()
        open("planning_control_time.txt", "w").close()
        for n in tqdm(range(N), desc="n loop", leave=False):
            overall_progress.update(1)
            # run cruise control N times
            cmd = f"python planning_based_control.py --init_pos {x} --init_speed {v} --save_time=True"
            print(cmd)
            os.system(cmd)

        # calculate safety probability
        counts = count_words("planning_control.txt", ['safe', 'unsafe'])
        total = sum(counts.values())
        if total == 0:
            F = counts['safe']/N
        else:
            F = counts['safe']/total
        # calculate average time horizon
        with open('planning_control_time.txt', 'r') as file:
            ticks = [list(map(float, line.split())) for line in file]
            
        planning_control_table[(x, v)] = [counts['safe'],counts['unsafe'], F, average_time(ticks)]

    with open("planning_control_risk.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(['init_pos', 'init_speed', 'total_safe', 'total_unsafe', 'safety_probability', 'average_time_horizon'])
        for key, value in planning_control_table.items():
            writer.writerow([key[0], key[1], value[0], value[1], value[2], value[3]])



def main():
    # initial_state = [(-98,5),(-98,2),(-38,6),(-38,3),(22,2)]

    initial_state = [(-22,2)]
    # tolerance = [0.05, 0.10, 0.15, 0.20]
    # tolerance = [0.05, 0.10]
    # tolerance = [0.05]
    # alpha = [0.05, 0.1, 0.2, 0.5, 1]
    N = 3

    # create_safe_control(initial_state, tolerance, alpha, N)
    # create_cruise_control(initial_state, N)
    # create_baseline_control(initial_state, tolerance, N)
    # create_transfuser_control(initial_state, N)
    # create_mpc_control(initial_state, N)
    create_planning_based_control(initial_state, N)
    

if __name__ == '__main__':
    main()
