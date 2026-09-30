
# AV_occlusion
CARLA implementation of the AV occlusion project.

The following experiments are based on [CARLA 0.9.10](https://carla.readthedocs.io/en/0.9.10/) version.
To set up the conda environment for the following experiments,
```sh
 conda env create -f env.yaml
```

## Safety Probability Estimation
Launch a terminal in the root CARLA directory and execute the simulator on port 2026 by running

```sh
 ./CarlaUE4.sh --world-port=2026
```
Launch another terminal and activate conda environment by running
```sh
 conda activate extreme_driving
```
If Python cannot locate CARLA egg file correctly ("ImportError: No Module named 'carla'"), run
```sh
 export PYTHONPATH=$PYTHONPATH:/home/tongyaoj/Documents/carla9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
```
Go to the root of the repo and run a single iteration of the occlusion simulation with the nominal cruise control via
```sh
 python cruise_control.py
```
which will run the simulation with the default initial speed, and initial position parameters. These can be adjusted using the following flags.
```sh
 python cruise_control.py --init_pos=x --init_speed=v
```
Single runs can save frames to generate videos by including the `--save` flag:
```sh
 python cruise_control.py --save
```

Boolean flags such as `--save`, `--save_time`, and `--save_trajectory` are false when omitted and true when present. Emergency stopping remains enabled by default; pass `--no-emergency` to disable it on controllers that support that option.

To get a lookup table for safety probability, set your desired range of initial position, range of initial speed, discretization steps, number of trials for each state, and run the following
```sh
 python create_risk_lookup.py
```
which will save 'risk_lookup_table.csv' with four columns: initial position, initial speed, total number of safe trials, and total number of unsafe trials. There are arguments to customize the range of initial states and discretization:
* xmin: left margin of inital position
* xmax: right margin of inital position
* xdelta: position discretization step
* vmin: left margin of inital speed
* vmax: right margin of inital speed
* vdelta: speed discretization step
* time_horizon: time horizon of the long-term safety probability
* N: number of trials for each initial state configuration

If CARLA crashes with the error message: "Disabling core dumps. Signal 11 caught", set the argument 'N' in create_risk_lookup.py smaller, adjust the number of iterations in shell file and run 
```sh
 ./lookup.sh
```
which will restart CARLA each iteration, and save a sequence of CSV files.
To visualize the risk lookup table, please refer to the **visualization and evaluation** section.

## Safe Control

The repository includes the processed lookup table as `risk_lookup_table.csv`. To combine raw lookup tables, create a folder named `tables`, place the raw CSV files in it, and run
```sh
 python lookup_postprocessing.py
```

Launch a terminal in the root of the code and implement safe control algorithm by running
```sh
 python safe_control.py
```
which will execute the safe controller. 
Arguments are described as below:
* init_pos: initial position (float between [-100, 82], this range is converted to [-182, 0] in the paper to make the place of occlusion the origin)
* init_speed: initial speed (float >= 0)
* alpha: control constant (float > 0)
* epsilon: risk tolerance (float between (0, 1))
* save: save frames to produce simulation birds' eye view video (True/False)
* save_brake :save velocity and control(brake or throttle command) versus time plot (True/False)
* save_pos: save position versus time plot (True/False)
* save_prob: save safety probability versus time plot (True/False)
* save_time: save simulation time in 'safe_control_time.txt' (True/False)
* save_trajectory: save velocity, position, control, and safety probability in files 'safe_velocity.txt', 'safe_position.txt', 'safe_u.txt', and 'safe_F.txt' respectively. (True/False)



## Visualization and Evaluation

To reproduce the figures in the paper, go to the visualization folder and run the corresponding code.
For **Figure 3**, run the following which will save the plot in 'table_trajectory.pdf'. Here we smoothed out the heatmap with Gaussian filter.
```sh
 python plot_lookup.py
```
For **Figure 4**, first run 6 control methods (PID, risk-based, TransFuser, OA-MPC, planning-based, and proposed) to save their trajectories by
```sh
 python pid_control.py --init_pos=-38 --init_speed=0 --save_trajectory
 python risk_based_control.py --init_pos=-38 --init_speed=0 --epsilon=0.02 --save_trajectory
 python oa_mpc_control.py --init_pos=-38 --init_speed=0 --save_trajectory
 python planning_based_control.py --init_pos=-38 --init_speed=0 --save_trajectory
 python safe_control.py --init_pos=-38 --init_speed=0 --epsilon=0.05 --alpha=0.2 --save_trajectory
```
and launch another terminal and go to the root of 'transfuser_control.py'
```sh
 conda activate tfuse-1
 python transfuser_control.py --init_pos=-38 --init_speed=0 --save_trajectory
```
(Notes: we use initial position as -38m in the simulation where the pedestrians are at 82m. But we transfer the initial position to -120m in the paper, as we set the pedestrian as the origin.)
And then use the trajectory obtained to produce the velocity vs time or velocity vs distance plot by
```sh
 python velocity_time.py
 python velocity_distance.py
```
we also uploaded our experiment trajectory results (text files) in this folder.
**Figure 1** and **Figure 2** source files are 'intersection_3D_figure.pptx' and 'carla_setup.pptx'.

To reproduce **Table II**, set the initial states and number of trials in `evaluate_time_risk.py`, then run
```sh
 python evaluate_time_risk.py
```
which will save 4 files 'safe_control_risk.csv', 'cruise_control_risk.csv', 'risk_control_risk', 'transfuser_control_risk.csv', with columns 'init_pos', 'init_speed', 'tolerance', 'total_safe', 'total_unsafe', 'safety_probability', and 'average_time_horizon'.

The code for plotting Figure 5 can be found at: [Figure_5.ipynb](https://colab.research.google.com/drive/1ZOmGRfEGCCozjD5iQtIjaN3IQuKomPJj#scrollTo=58qCO4V4IPtw)
