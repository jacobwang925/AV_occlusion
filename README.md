# AV_occlusion
Carla implementation of the AV occlusion project

## Safety Probability Estimation
The following experiments are based on Carla 0.9.10 vesion.

Activate Conda environment by running
```sh
 conda activate extreme_driving
```
Launch a terminal in the root Carla directory and execute the simulator on port 2026 by running

```sh
 ./CarlaUE4.sh --world-port=2026
```
Launch a terminal in the root of code and begin a single iteration of the occlusion simulation by running
```sh
 python cruise_control.py
```
which will run the simulation with the default inital speed, and initial position parameters. These can be adjusted using the following flags.
```sh
 python cruise_control.py --init_pos=x --init_speed=v
```
Single runs can save frames to generate videos as well, as seen with the additional flags below:
```sh
 python cruise_control.py --save=True
```

To get a lookup table, set your desired number of iteration and run the following
```sh
 ./lookup.sh
```
which will restart Carla each iteration, and save a lookup heatmap figure.

## Safe Control

