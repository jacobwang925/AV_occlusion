import pandas as pd
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import csv

vmin = 0
vmax = 11
vdelta = 0.5
xmin = -100
xmax = 84
xdelta = 2


def lookup_table_processing(folder_path):
    tables = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    lookup = None

    # accumulate tables
    for table in tables:
        file_path = os.path.join(folder_path, table)
        df = pd.read_csv(file_path)
    
        if lookup is None:
            lookup = df.copy()
        else:
            # safe counts
            lookup['total_safe'] += df['total_safe']
            # unsafe counts
            lookup['total_unsafe'] += df['total_unsafe']

    safety_probability = lookup['total_safe'] / (lookup['total_safe'] + lookup['total_unsafe'])
    lookup.insert(2, 'safety_probability', safety_probability)

    # Write the result to a CSV file
    output_file = 'risk_lookup_table_processed.csv'
    lookup.to_csv(output_file, index=False)

    # add safety=1 when initial speed=0
    lookup={}
    df = pd.read_csv(output_file)
    init_pos = df['init_pos']
    init_speed = df['init_speed']
    F = df['safety_probability']
    safe = df['total_safe']

    for i in range(len(init_pos)):
        lookup[(init_speed[i],init_pos[i])] = [F[i], safe[i]]
    for pos in range(-100, 84, 2):
        lookup[(pos,0)] = [1, 1]

    with open(output_file, mode="w", newline='') as file:
        writer = csv.writer(file)
        for key, value in lookup.items():
            writer.writerow([key[0], key[1], value[0]])

    return



def main():
    lookup_table_processing(folder_path = 'tables')

if __name__ == '__main__':
    main()
