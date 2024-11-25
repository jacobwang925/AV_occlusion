# AV_occlusion
Carla implementation of the AV occlusion project.

The following experiments are based on Carla 0.9.10 vesion.

## Safety Probability Estimation
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
which will run the simulation with the default initial speed, and initial position parameters. These can be adjusted using the following flags.
```sh
 python cruise_control.py --init_pos=x --init_speed=v
```
Single runs can save frames to generate videos as well, as seen with the additional flags below:
```sh
 python cruise_control.py --save=True
```

To get a lookup table, set your desired number of iterations and run the following
```sh
 ./lookup.sh
```
which will restart Carla each iteration, and save a lookup heatmap figure.

## Safe Control
Launch a terminal in the root Carla directory and execute the simulator on port 2026 by running
```sh
 ./CarlaUE4.sh --world-port=2026
```
Launch a terminal in the root of code and implement safe control algorithm by running
```sh
 python safe_control.py
```
which will execute the safe controller. To customize the initial state, risk tolerance, and controller scaling constant, run
```sh
 python safe_control.py --init_pos=x --init_speed=v --epsilon=e --alpha=a
```
To save the plot of velocity and control(brake and throttle), run with argument:
```sh
 python safe_control.py --save_brake=True
```
The trajectory of velocity, control, safety probability, and position can be saved by argument 'save_trajectory':
```sh
 python safe_control.py --save_trajectory=True
```
which saves the above states in files 'stats_velocity.txt', 'stats_u.txt', 'stats_F.txt', 'stats_position.txt' respectively. Each line represents one run.


## Result Visualization

To plot the trajectories with the lookup table, run 
```sh
 python plot_lookup.py
```

To plot the trajectories of control and velocity for both proposed method and PID control, run
 ```sh
 python plot_trajectory.py
```
choose smoothing method if needed.
