import glob
import os
import sys
import queue
import matplotlib.pyplot as plt
import argparse
import csv
import copy
import math
from PIL import Image
from collections import deque
from queue import Queue
from queue import Empty
from threading import Thread
from copy import deepcopy
import torch
import itertools

# export PYTHONPATH=$PYTHONPATH:/home/tongyaoj/Documents/carla9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg


sys.path.append(os.path.abspath('../carla/agents/navigation'))
import controller

# add path for gps target goal
sys.path.append(os.path.abspath('/home/tongyaoj/Documents/carla/PythonAPI/transfuser/leaderboard'))
sys.path.append(os.path.abspath('/home/tongyaoj/Documents/carla9.10/PythonAPI/carla'))
# from agents.navigation.global_route_planner import GlobalRoutePlanner
# from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
# from agents.navigation.local_planner import RoadOption
from leaderboard.utils.route_manipulation import interpolate_trajectory
from agents.navigation.local_planner import RoadOption
  
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

import pid_control as carlaPid
from risk_calc import count_words

from data import lidar_to_histogram_features, draw_target_point, lidar_bev_cam_correspondences

IM_WIDTH = 640
IM_HEIGHT = 960

u_stats = []
# brake = []
velocity_stats = []
velocity_y = []
position_stats = []
safety_probability = []
frames = []
curr_walkers = []
actor_list = []
key_x = []
key_v = []

def process_img(image, world):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    frames.append(i3)
    world.wait_for_tick()
    return i3/255.0

def spawn_walker(world, ego_vehicle, init_pos, walker_id):
    # print(walker_id)
    walker_bp = random.choice(world.get_blueprint_library().filter('walker.pedestrian.*'))
    # create spawn point at intersection walking left to right
    spawn_point = carla.Transform()
    loc = ego_vehicle.get_location()
    if walker_id % 2 == 0:
        loc.x += 82.0 - init_pos + 0.5
    else:
        loc.x += 82.0 - init_pos + 0.1
    loc.y += 12.0 + walker_id / 2
    loc.z += 1.0
    # if walker_id == 1:
    #     print("walker===========",loc.x, loc.y)
    spawn_point.location = loc    
    walker = world.spawn_actor(walker_bp, spawn_point)
    curr_walkers.append(walker)
    return walker

def spawn_occlusion(world, ego_vehicle, init_pos):
    # spawn truck on right side of road
    truck_bp = random.choice(world.get_blueprint_library().filter('vehicle.carlamotors.carlacola'))
    spawn_point = carla.Transform()
    loc = ego_vehicle.get_location()
    loc.x += 75.0 - init_pos
    # loc.x += 30.0 - init_pos

    loc.y += 5.0
    loc.z += 1.0
    spawn_point.location = loc
    truck = world.spawn_actor(truck_bp, spawn_point)
    actor_list.append(truck)
    print('truck spawned')
    return truck

def is_occluded(walker_location, occlusion, occ_dimensions, ego_vehicle):
    walker_pos = walker_location
    occlusion_pos = occlusion.get_location()
    vehicle_pos = ego_vehicle.get_location()
    occ_width = occ_dimensions.x
    occ_height = occ_dimensions.y

    # get vectors
    vector_x = vehicle_pos.x - walker_pos.x
    vector_y = vehicle_pos.y - walker_pos.y

    # get normal vector
    norm_x = -vector_y
    norm_y = vector_x

    # distance from occlusion
    dist = abs((occlusion_pos.x - walker_pos.x) * norm_x + (occlusion_pos.y - walker_pos.y) * norm_y) / (norm_x**2 + norm_y**2)**0.5

    # check if walker is occluded
    if dist <= occ_width/2 or dist <= occ_height/2:
        proj_x = walker_pos.x + (vector_x * norm_x + vector_y * norm_y) / (vector_x**2 + vector_y**2) * vector_x
        proj_y = walker_pos.y + (vector_x * norm_x + vector_y * norm_y) / (vector_x**2 + vector_y**2) * vector_y

        if (proj_x >= occlusion_pos.x - occ_width/2 
            and proj_x <= occlusion_pos.x + occ_width/2 
             and proj_y >= occlusion_pos.y - occ_height/2 
            and proj_y <= occlusion_pos.y + occ_height/2):
            return True

    return False

def is_visible(walker, occlusion, occ_dimensions, ego_vehicle):
    # check if walker is visible to vehicle
    walker_location = walker.get_location()
    # check if walker is within box of sight of vehicle
    if walker_location.y < ego_vehicle.get_location().y + 6.5 and walker_location.y > ego_vehicle.get_location().y - 6.5:
                # check if walker is within 10 meters of the vehicle
                if 0.0 < walker_location.x - ego_vehicle.get_location().x < 10.0:
                    # check if line between line between walker and vehicle intersects with occlusion
                    if not is_occluded(walker_location, occlusion, occ_dimensions, ego_vehicle):
                        return True
    return False

def find_closest_table_entry(x, v, lookup_table):
    # find closest x and v in lookup table
    min_dist = float('inf')
    closest = None
    for key in lookup_table.keys():
        x_key, v_key = key
        dist = np.sqrt((x - x_key)**2 + (v - v_key)**2)
        if dist < min_dist:
            min_dist = dist
            closest = key # [x,v] state
    return closest, lookup_table[closest]





# ============ attatch sensor to vehicle ================
def add_sensor(sensor_type, world, location, rotation, sensor_id, ego_vehicle, sensors):
    # Find the blueprint of the sensor.
    blueprint = world.get_blueprint_library().find(sensor_type)
    if sensor_type == 'sensor.camera.rgb':
        blueprint.set_attribute('image_size_x', '960')
        blueprint.set_attribute('image_size_y', '480')
        blueprint.set_attribute('chromatic_aberration_intensity', str(0.5))
        blueprint.set_attribute('chromatic_aberration_offset', str(0))
        if sensor_id == 'rgb_back':
            blueprint.set_attribute('fov', '100')
        else:
            blueprint.set_attribute('fov', '120')
    # Set the time in seconds between sensor captures
    # carla_frame_rate = 1.0 / 20.0
    if sensor_type == 'sensor.other.gnss':
        blueprint.set_attribute('sensor_tick', '0.01') # ////////////different frame rate?////////////
        blueprint.set_attribute('noise_alt_stddev', str(0.000005))
        blueprint.set_attribute('noise_lat_stddev', str(0.000005))
        blueprint.set_attribute('noise_lon_stddev', str(0.000005))
    else:
        blueprint.set_attribute('sensor_tick', '0.05')

    # if sensor_type.startswith('sensor.lidar'):
    if sensor_type == 'sensor.lidar.ray_cast':
        blueprint.set_attribute('range', str(85.0))
        # if DATAGEN==1:
        #     blueprint.set_attribute('rotation_frequency', str(sensor_spec['rotation_frequency']))
        #     blueprint.set_attribute('points_per_second', str(sensor_spec['points_per_second']))
        # else:
        blueprint.set_attribute('rotation_frequency', str(20))
        blueprint.set_attribute('points_per_second', str(600000))
        
        blueprint.set_attribute('channels', str(64))
        blueprint.set_attribute('upper_fov', str(10)) # default = 10
        blueprint.set_attribute('lower_fov', str(-30)) # default = -30
        blueprint.set_attribute('atmosphere_attenuation_rate', str(0.004))
        blueprint.set_attribute('dropoff_general_rate', str(0.45))
        blueprint.set_attribute('dropoff_intensity_limit', str(0.8))
        blueprint.set_attribute('dropoff_zero_intensity', str(0.4))
        blueprint.set_attribute('sensor_tick', '0.05')
        # sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
        #                                     z=sensor_spec['z'])
        # sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
        #                                     roll=sensor_spec['roll'],
        #                                     yaw=sensor_spec['yaw'])



    blueprint.set_attribute('role_name', sensor_id)
    transform = carla.Transform(location, rotation)
    sensor = world.spawn_actor(blueprint, transform, attach_to=ego_vehicle)
    sensors.append(sensor)
    
    
    return sensor


def speedometer(ego_vehicle):
    velocity= ego_vehicle.get_velocity()
    transform = ego_vehicle.get_transform()
    vel_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    speed = np.dot(vel_np, orientation)
    return speed

# def lidar_data(data):
#     points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
#     points = np.reshape(points,(-1,4))
#     print(data)


global input_data
input_data = {}

# camera_image = {}

# Listen from sensors
# Parsing CARLA physical Sensors
def _parse_image_cb(image, sensor_id): # , input_data, sensor_id):
    # global camera_image
    image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
    image_data = image_data.reshape((image.height, image.width, 4))  # RGBA format
    # camera_image[sensor_id] = image_data[:, :, :3]  # Keep only RGB
    input_data[sensor_id] = (0,image_data)

# def _parse_radar_cb(self, radar_data, tag):
#     # [depth, azimuth, altitute, velocity]
#     points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
#     points = copy.deepcopy(points)
#     points = np.reshape(points, (int(points.shape[0] / 4), 4))
#     points = np.flip(points, 1)
#     self._data_provider.update_sensor(tag, points, radar_data.frame)

def _parse_gnss_cb(gnss_data, sensor_id):
    # global gps_output
    array = np.array([gnss_data.latitude,
                    gnss_data.longitude,
                    gnss_data.altitude], dtype=np.float64)
    # gps_output = (0,array)
    input_data[sensor_id] = (0,array)

def _parse_imu_cb(imu_data, sensor_id):
    # global imu_output
    array = np.array([imu_data.accelerometer.x,
                          imu_data.accelerometer.y,
                          imu_data.accelerometer.z,
                          imu_data.gyroscope.x,
                          imu_data.gyroscope.y,
                          imu_data.gyroscope.z,
                          imu_data.compass,
                         ], dtype=np.float64)
    # imu_output = (0,array)
    input_data[sensor_id] = (0,array)


# def _parse_pseudosensor(self, package, tag):
#     self._data_provider.update_sensor(tag, package.data, package.frame)
# =================================

def _parse_lidar_cb(lidar_data, sensor_id):
    # global lidar_output
    points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
    # print(len(points))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))

    # # x->-y, y->z, z->x
    # x = points[:,0]
    # y = points[:,1]
    # z = points[:,2]
    # intensity = points[0:,3]

    # new_x = z
    # new_y = x
    # new_z = y
    # # print(f"min max x {np.min(new_x):.1f}, {np.max(new_x):.1f},y {np.min(new_y):.1f},{np.max(new_y):.1f},z {np.min(new_z):.1f}, {np.max(new_z):.1f}")

    # transformed_points = np.column_stack((new_x, new_y, new_z, intensity))

    # # lidar_output = (0,points)
    input_data[sensor_id] = (0,points)
    # input_data[sensor_id] = (0,transformed_points)






def safe_controller(ego_vehicle, 
                    world, image_queue, 
                    spawn_points, 
                    init_pos, 
                    init_speed, 
                    alpha, 
                    epsilon, 
                    emergency_activate, 
                    num_walker, 
                    save_time, 
                    sensors, 
                    model):
    # parameters
    # N = 3
    # epsilon = 1e-3

    # spawn standard PID controller
    # spawn a vehicle pid controller
    # args_longitudinal = {
    #     'K_P': 0.05,
    #     'K_D': 0.1,
    #     'K_I': 0.05,
    #     'dt': 0.003
    # }
    # target_speed = 5 #m/s
    target_speed = 5 #m/s

    # =========== PID ============
    # args_longitudinal = {
    #     'K_P': 0.04,
    #     'K_D': 0.03,
    #     'K_I': 0.01,
    #     'dt': 0.001
    # }
    # target_speed = 4.5 # m/s

    # ego_control = controller.PIDLongitudinalController(ego_vehicle, **args_longitudinal)

    # random spawn walker
    # normal distribution
    spawn_ticks = np.zeros(num_walker)
    for i in range(num_walker):
        mean_first_walker = 30
        std_first_walker = 50
        t_max_first_walker = 200

        mean_interval = 120
        std_interval = 50
        t_max_interval = 300
        if i == 0:
            # if in_sight == True:
            #     spawn_ticks[i] = 1
            # else:
            while True:
                spawn_ticks[i] = round(min(np.random.normal(mean_first_walker, std_first_walker), t_max_first_walker))
                if spawn_ticks[i] > 0:
                    break
        else:
            while True:
                interval = round(min(np.random.normal(mean_interval, std_interval), t_max_interval))
                if interval > 0:
                    break
            spawn_ticks[i] = interval + spawn_ticks[i-1]
    # print(spawn_ticks)
	
    # poisson distribution
    # poisson_interval = 130
    # spawn_ticks = np.zeros(num_walker)
    # for i in range(num_walker):
    #     if i == 0:
    #         spawn_ticks[i] = min(np.random.poisson(48, 1), 136)
    #     else:
    #         spawn_ticks[i] = min(np.random.poisson(poisson_interval, 1), poisson_interval*3) + spawn_ticks[i-1]

    # print("start walker at tick:", spawn_tick)
    for walker_id in range(num_walker):
        spawn_walker(world, ego_vehicle, init_pos, walker_id)

    occlusion = spawn_occlusion(world, ego_vehicle, init_pos)
    occlusion_dim = occlusion.bounding_box.extent

    # set initial speed
    ego_vehicle.set_target_velocity(carla.Vector3D(x=init_speed, y=0.0, z=0.0))

    # pull in risk lookup table
    lookup_table = {}
    # with open('lookup_table_processed.csv', 'r', encoding='utf-8-sig') as file:
    with open('risk_lookup_table.csv', 'r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        for row in reader:
            x = float(row[0])
            v = float(row[1])
            F = float(row[2])
            lookup_table[(x, v)] = F
            # if v == 1:
            #     lookup_table[(x, 0)] = 1.0

    tick_count = 0
    spawned = np.zeros(num_walker) # boolean to check if the nth walker is spawned

    time_horizon = 10000

    # pull in risk lookup table
    lookup_table = {}
    # with open('lookup_table_processed.csv', 'r', encoding='utf-8-sig') as file:
    with open('risk_lookup_table.csv', 'r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        for row in reader:
            x = float(row[0])
            v = float(row[1])
            F = float(row[2])
            lookup_table[(x, v)] = F
    

    while tick_count < time_horizon:
        tick_count += 1
        world.tick()
        image = image_queue.get()
        process_img(image, world)

        pos = ego_vehicle.get_location().x - spawn_points[1].location.x + init_pos
        speed = ego_vehicle.get_velocity().x

        key, F = find_closest_table_entry(pos, speed, lookup_table)
        safety_probability.append(F)

        if tick_count == 1:
            print("=======position_x", ego_vehicle.get_location().x, "position_y", ego_vehicle.get_location().y)


        
        # start walker
        for i in range(num_walker):
            if tick_count == spawn_ticks[i] and spawned[i] == 0:
                print("start walker", (i+1)," at tick: ", spawn_ticks[i])
                walk_dir = carla.Vector3D(x=0 , y=-1, z=0)
                curr_walkers[i].apply_control(carla.WalkerControl(direction = walk_dir, speed=1.0))
                spawned[i] = 1

                
        # input_data = {}
        input_data['speed'] = (0,{'speed': speedometer(ego_vehicle)})


        # save lidar data
        lidar_temp = input_data['lidar'][1]
        test_path = '/home/tongyaoj/Documents/carla/PythonAPI/examples/lidar/'
        np.save(test_path+'lidar_'+str(tick_count)+'.npy', lidar_temp)


        # NOTE: Attempt Oct 30 - had input_data
        
        tick_dt = model.tick(input_data, tick_count)
        transfuser_control = model.return_control(tick_dt, tick_count)
        
        transfuser_control.steer = 0.0

        if transfuser_control.brake == 0 and transfuser_control.throttle == 0:
            u_stats.append(0)
        elif transfuser_control.brake != 0:
            u_stats.append(-transfuser_control.brake)
        else:
            u_stats.append(transfuser_control.throttle)

        if pos > 66:
            transfuser_control = carla.VehicleControl(throttle=0.0, brake=1.0)

        ego_vehicle.apply_control(transfuser_control)



    #     # ---------- done get transfuser control -

        # get current state of ego vehicle
        pos = ego_vehicle.get_location().x - spawn_points[1].location.x + init_pos
        speed = ego_vehicle.get_velocity().x


        velocity_stats.append(speed)
        position_stats.append(pos)

        # check for collision
        for walker in curr_walkers:
            if ego_vehicle.get_location().distance(walker.get_location()) < 3.0:
                print("collision detected")
                print("tick: ", tick_count)
                with open('transfuser_control.txt', 'a') as f:
                    f.write('unsafe\n')
                return

        # check if vehicle is at the end of the road
        if ego_vehicle.get_location().x > spawn_points[1].location.x + 100.0 - init_pos:
            print("end of road")
            print("=======safe!!!=======")
            print("tick: ", tick_count)
            if save_time:
                with open('transfuser_control_time.txt', 'a') as f:
                    f.write(str(tick_count) + '\n')
            with open('transfuser_control.txt', 'a') as f:
                f.write('safe\n')
            return




#----------------------------------------------------------------------
# NOTE: add transfuser model
    
path_to_conf_file = '/home/tongyaoj/Documents/carla/PythonAPI/TransFuserAllTownsNoZeroNoSyncZGSeed1'

# Taken from World on Rails
class EgoModel():
    def __init__(self, dt=1./4):
        self.dt = dt
        
        # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
        self.front_wb    = -0.090769015
        self.rear_wb     = 1.4178275

        self.steer_gain  = 0.36848336
        self.brake_accel = -4.952399
        self.throt_accel = 0.5633837

    def forward(self, locs, yaws, spds, acts):
        # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
        steer = acts[..., 0:1].item()
        throt = acts[..., 1:2].item()
        brake = acts[..., 2:3].astype(np.uint8)

        if (brake):
            accel = self.brake_accel
        else:
            accel = self.throt_accel * throt

        wheel = self.steer_gain * steer

        beta = math.atan(self.rear_wb / (self.front_wb + self.rear_wb) * math.tan(wheel))
        yaws = yaws.item()
        spds = spds.item()
        next_locs_0 = locs[0].item() + spds * math.cos(yaws + beta) * self.dt
        next_locs_1 = locs[1].item() + spds * math.sin(yaws + beta) * self.dt
        next_yaws = yaws + spds / self.rear_wb * math.sin(beta) * self.dt
        next_spds = spds + accel * self.dt
        next_spds = next_spds * (next_spds > 0.0)  # Fast ReLU

        next_locs = np.array([next_locs_0, next_locs_1])
        next_yaws = np.array(next_yaws)
        next_spds = np.array(next_spds)

        return next_locs, next_yaws, next_spds



class initialize_model():
    def __init__(self, world):
        import os
        import json
        from copy import deepcopy

        import cv2
        import carla
        from PIL import Image
        from collections import deque

        import torch
        import numpy as np
        import math
        from shapely.geometry import Polygon
        import itertools
        import pathlib
        
        sys.path.append(os.path.abspath('./team_code_transfuser/'))
        from config import GlobalConfig
        from model import LidarCenterNet
        from data import lidar_to_histogram_features, draw_target_point, lidar_bev_cam_correspondences
        args_file = open(os.path.join(path_to_conf_file, 'args.txt'), 'r')
        self.args = json.load(args_file)
        args_file.close()
        # from leaderboard.autoagents import autonomous_agent
        self.config = GlobalConfig(setting='eval')


        # global plan
        # +++++++++++++++++++check++++++++++++++++++++++++++
        start_point = carla.Location(x=140.5166015625 + init_pos, y=4.808422565460205, z=0.27)
        # start_point = carla.Location(x=140.5166015625, y=4.808422565460205, z=0.27)

        end_point = carla.Location(x=140.5166015625+200, y=4.808422565460205, z=0.27)
        config_trajectory = [start_point, end_point]


        print("++++++start+++++++++", start_point)
        gps_route, route = interpolate_trajectory(world, config_trajectory)
        # breakpoint()

        # print(gps_route.location.x, gps_route.location.y, gps_route.location.z)
        _global_plan_coor, self._global_plan = self.set_global_plan(gps_route, route)


        self.control = carla.VehicleControl(throttle=0.0, brake=0.0, steer = 0.0)

        if ('sync_batch_norm' in self.args):
            self.config.sync_batch_norm = bool(self.args['sync_batch_norm'])
        if ('use_point_pillars' in self.args):
            self.config.use_point_pillars = self.args['use_point_pillars']
        if ('n_layer' in self.args):
            self.config.n_layer = self.args['n_layer']
        if ('use_target_point_image' in self.args):
            self.config.use_target_point_image = bool(self.args['use_target_point_image'])
        if ('use_velocity' in self.args):
            use_velocity = bool(self.args['use_velocity'])
        else:
            use_velocity = True

        if ('image_architecture' in self.args):
            image_architecture = self.args['image_architecture']
        else:
            image_architecture = 'resnet34'

        if ('lidar_architecture' in self.args):
            lidar_architecture = self.args['lidar_architecture']
        else:
            lidar_architecture = 'resnet18'

        if ('backbone' in self.args):
            self.backbone = self.args['backbone']  # Options 'geometric_fusion', 'transFuser', 'late_fusion', 'latentTF'
        else:
            self.backbone = 'transFuser'  # Options 'geometric_fusion', 'transFuser', 'late_fusion', 'latentTF'

        self.gps_buffer = deque(maxlen=self.config.gps_buffer_max_len) # Stores the last x updated gps signals.
        self.ego_model = EgoModel(dt=self.config.carla_frame_rate) # Bicycle model used for de-noising the GPS

        self.bb_buffer = deque(maxlen=1)
        self.lidar_pos = self.config.lidar_pos  # x, y, z coordinates of the LiDAR position.
        self.iou_treshold_nms = self.config.iou_treshold_nms # Iou threshold used for Non Maximum suppression on the Bounding Box predictions.



        # Load model files
        self.nets = []
        self.model_count = 0 # Counts how many models are in our ensemble
        for file in os.listdir(path_to_conf_file):
            if file.endswith(".pth"):
                self.model_count += 1
                print(os.path.join(path_to_conf_file, file))
                net = LidarCenterNet(self.config, 'cuda', self.backbone, image_architecture, lidar_architecture, use_velocity)
                if(self.config.sync_batch_norm == True):
                    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net) # Model was trained with Sync. Batch Norm. Need to convert it otherwise parameters will load incorrectly.
                state_dict = torch.load(os.path.join(path_to_conf_file, file), map_location='cuda:0')
                new_state_dict = {}
                for (k, v) in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                print(new_state_dict['_model.image_encoder.features.stem.conv.weight'][0][0][0]) # test
                net.load_state_dict(new_state_dict, strict=False)
                print(net._model.image_encoder.features.stem.conv.weight[0][0][0]) # test
                net.cuda()
                net.eval()
                self.nets.append(net)


        self.stuck_detector = 0
        self.forced_move = 0

        self.use_lidar_safe_check = True
        self.aug_degrees = [0] # Test time data augmentation. Unused we only augment by 0 degree.
        self.steer_damping = self.config.steer_damping
        self.rgb_back = None # For debugging


        # print(self.nets)
        # print(state_dict)

        # self._route_planner = RoutePlanner(self.config.route_planner_min_distance, self.config.route_planner_max_distance)
        
        # prone to error
        # self._global_plan = [({'lat': -6.735948933567215e-05, 'lon': 0.0005879785099296277, 'z': 0.05975331366062164}), ({'lat': -7.420358598153598e-05, 'lon': 0.0010461579591442619, 'z': 0.10669898986816406}), ({'lat': -7.673674569730338e-05, 'lon': 0.0012157408071002908, 'z': 0.05549215152859688}), ({'lat': -8.06277238751818e-05, 'lon': 0.0014762231802236763, 'z': 0.001464062836021185})]
        # self._route_planner.set_route(self._global_plan, True) ##?
        # self.initialized = True
    
    def _init(self):
        self._route_planner = RoutePlanner(7.5, 50)
        # breakpoint()
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    def tick(self, input_data, tick_count):
        """ adjust tick function 
        remaining issues

        - self._route_planner
        - def set_route(self, global_plan,...
        """

        self._init()
        rgb = []
        for pos in ['left', 'front', 'right']:
            rgb_cam = 'rgb_' + pos
            rgb_pos = cv2.cvtColor(input_data[rgb_cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            rgb_pos = self.scale_crop(Image.fromarray(rgb_pos), self.config.scale, self.config.img_width, self.config.img_width, self.config.img_resolution[0], self.config.img_resolution[0])
            rgb.append(rgb_pos)
        rgb = np.concatenate(rgb, axis=1)

        # if(SAVE_PATH != None): #Debug camera for visualizations
        #     # don't need buffer for it always use the latest one
        #     self.rgb_back = input_data["rgb_back"][1][:, :, :3]

        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        if (np.isnan(compass) == True): # CARLA 0.9.10 occasionally sends NaN values in the compass
            compass = 0.0

        result = {
                'rgb': rgb,
                'gps': gps,
                'speed': speed,
                'compass': compass,
                }

        if (self.backbone != 'latentTF'):
            lidar = input_data['lidar'][1][:, :3]
            result['lidar'] = lidar


        pos = self._get_position(result)
        result['gps'] = pos

        self.gps_buffer.append(pos)
        denoised_pos = np.average(self.gps_buffer, axis=0)


        waypoint_route = self._route_planner.run_step(denoised_pos)
        next_wp, next_cmd = waypoint_route[1] if len(waypoint_route) > 1 else waypoint_route[0]
        result['next_command'] = next_cmd.value


        theta = compass + np.pi/2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
            ])

        local_command_point = np.array([next_wp[0]-denoised_pos[0]+3.1, next_wp[1]-denoised_pos[1]])
        # local_command_point = np.array([next_wp[0]-denoised_pos[0], next_wp[1]-denoised_pos[1]])
        local_command_point = R.T.dot(local_command_point)
        
        # quick debug attempt
        result['target_point'] = tuple(local_command_point)

        
        # print('target point', result['target_point'])
        # print('gps', result['gps'])


        # NOTE: alternate fixed
       
        # result['next_command'] = 4
        # result['target_point'] = (-0.12522617195979757+tick_count, -50.82859906855731+tick_count)
        # result['gps'] = np.array([-8.08377439+tick_count, 65.47224859+tick_count]) # fixed

        return result
    
    def return_control(self, tick_data, tick_count):
        """ hand-crafted function to act get input from tick and return"""
        # repeat actions twice to ensure LiDAR data availability
        self.step = tick_count
        if self.step % self.config.action_repeat == 1:
            self.update_gps_buffer(self.control, tick_data['compass'], tick_data['speed'])
            return self.control

        # prepare image input
        image = self.prepare_image(tick_data)

        num_points = None
        if(self.backbone == 'latentTF'): # Image only method
            lidar_bev = torch.zeros((1, 2, self.config.lidar_resolution_width, self.config.lidar_resolution_height)).to('cuda', dtype=torch.float32) #Dummy data
        else:
            # prepare LiDAR input
            if (self.config.use_point_pillars == True):
                lidar_cloud = deepcopy(input_data['lidar'][1])
                lidar_cloud[:, 1] *= -1  # invert
                lidar_bev = [torch.tensor(lidar_cloud).to('cuda', dtype=torch.float32)]
                num_points = [torch.tensor(len(lidar_cloud)).to('cuda', dtype=torch.int32)]
            else:
                lidar_bev = self.prepare_lidar(tick_data)

        
        # prepare goal location input
        target_point_image, target_point = self.prepare_goal_location(tick_data)

        # prepare velocity input
        gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32) # used by controller
        velocity = gt_velocity.reshape(1, 1) # used by transfuser

        # unblock
        is_stuck = False
        # divide by 2 because we process every second frame
        # 1100 = 55 seconds * 20 Frames per second, we move for 1.5 second = 30 frames to unblock
        if(self.stuck_detector > self.config.stuck_threshold and self.forced_move < self.config.creep_duration):
            print("Detected agent being stuck. Move for frame: ", self.forced_move)
            is_stuck = True
            self.forced_move += 1


        # forward pass
        with torch.no_grad():
            pred_wps = []
            bounding_boxes = []
            for i in range(self.model_count):
                rotated_bb = []
                if (self.backbone == 'transFuser'):
                    # put false for 'save_path' for now
                    pred_wp, _ = self.nets[i].forward_ego(image, lidar_bev, target_point, target_point_image, velocity,
                                                          num_points=num_points, save_path=False, stuck_detector=self.stuck_detector,
                                                          forced_move=is_stuck, debug=self.config.debug, rgb_back=self.rgb_back)
                    # NOTE: added
                elif (self.backbone == 'late_fusion'):
                    pred_wp, _ = self.nets[i].forward_ego(image, lidar_bev, target_point, target_point_image, velocity, num_points=num_points)
                elif (self.backbone == 'geometric_fusion'):
                    bev_points = list()
                    cam_points = list()

                    curr_bev_points, curr_cam_points = lidar_bev_cam_correspondences(deepcopy(tick_data['lidar']), lidar_bev, image, self.step, False)
                    bev_points.append(torch.from_numpy(curr_bev_points).unsqueeze(0))
                    cam_points.append(torch.from_numpy(curr_cam_points).unsqueeze(0))

                    bev_points = bev_points[0].long().to('cuda', dtype=torch.int64)
                    cam_points = cam_points[0].long().to('cuda', dtype=torch.int64)
                    pred_wp, _ = self.nets[i].forward_ego(image, lidar_bev, target_point, target_point_image, velocity, bev_points, cam_points, num_points=num_points)
                elif (self.backbone == 'latentTF'):
                    pred_wp, rotated_bb = self.nets[i].forward_ego(image, lidar_bev, target_point, target_point_image, velocity, num_points=num_points)
                else:
                    raise ("The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF")

                pred_wps.append(pred_wp)
                bounding_boxes.append(rotated_bb)

        bbs_vehicle_coordinate_system = self.non_maximum_suppression(bounding_boxes, self.iou_treshold_nms)

        self.bb_buffer.append(bbs_vehicle_coordinate_system)
        self.pred_wp = torch.stack(pred_wps, dim=0).mean(dim=0) #Average the predictions from the ensembles

        # transform to local coordinates
        pred_wp_transformed = []
        for i, degree in enumerate(self.aug_degrees):
            rad = np.deg2rad(degree)
            degree_matrix = np.array([[np.cos(rad), np.sin(rad)],
                                [-np.sin(rad), np.cos(rad)]])
            # inverse
            degree_matrix = degree_matrix.T
            cur_pred_wp = self.pred_wp[i].detach().cpu().numpy()
            transformed_wp = (degree_matrix @ cur_pred_wp.T).T
            pred_wp_transformed.append(transformed_wp)

        self.pred_wp = np.stack(pred_wp_transformed, axis=0)
        self.pred_wp = torch.median(torch.from_numpy(self.pred_wp).to('cuda', dtype=torch.float32), dim=0, keepdims=True)[0]

        if (self.backbone == 'latentTF'):
            safety_box = []
            if(self.bb_detected_in_front_of_vehicle(gt_velocity) == True):
                safety_box.append(True)
        else:
            # safety check
            safety_box = deepcopy(tick_data['lidar'])
            safety_box[:, 1] *= -1  # invert

            # z-axis
            safety_box      = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
            safety_box      = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]

            # y-axis
            safety_box      = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
            safety_box      = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]

            # x-axis
            safety_box      = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
            safety_box      = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]

        steer, throttle, brake = self.nets[0].control_pid(self.pred_wp, gt_velocity, is_stuck)
        
        if is_stuck and self.forced_move==1: # no steer for initial frame when unblocking
            steer = 0.0

        # steer modulation
        if brake or is_stuck:
            steer *= self.steer_damping
        if(gt_velocity < 0.1): # 0.1 is just an arbitrary low number to threshhold when the car is stopped
            self.stuck_detector += 1
        elif(gt_velocity > 0.1 and is_stuck == False):
            self.stuck_detector = 0
            self.forced_move    = 0

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        # Safety controller. Stops the car in case something is directly in front of it.
        if self.use_lidar_safe_check:
            emergency_stop = (len(safety_box) > 0) #Checks if the List is empty
            if ((emergency_stop == True) and (is_stuck == True)):  # We only use the saftey box when unblocking
                print("Detected object directly in front of the vehicle. Stopping. Step:", self.step)
                control.steer = float(steer)
                control.throttle = float(0.0)
                control.brake = float(True)
                # Will overwrite the stuck detector. If we are stuck in traffic we do want to wait it out.

        self.control = control

        self.update_gps_buffer(self.control, tick_data['compass'], tick_data['speed'])
        # NOTE: added
        cv2.waitKey(1)
        return control
    
    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        ds_ids = self.downsample_route(global_plan_world_coord, 50)
        # breakpoint()
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        return self._global_plan_world_coord, self._global_plan

    def downsample_route(self, route, sample_factor):
        """
        Downsample the route by some factor.
        :param route: the trajectory , has to contain the waypoints and the road options
        :param sample_factor: Maximum distance between samples
        :return: returns the ids of the final route that can
        """

        ids_to_sample = []
        prev_option = None
        dist = 0

        for i, point in enumerate(route):
            curr_option = point[1]

            # Lane changing
            if curr_option in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                ids_to_sample.append(i)
                dist = 0

            # When road option changes
            elif prev_option != curr_option and prev_option not in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                ids_to_sample.append(i)
                dist = 0

            # After a certain max distance
            elif dist > sample_factor:
                ids_to_sample.append(i)
                dist = 0

            # At the end
            elif i == len(route) - 1:
                ids_to_sample.append(i)
                dist = 0

            # Compute the distance traveled
            else:
                curr_location = point[0].location
                prev_location = route[i-1][0].location
                dist += curr_location.distance(prev_location)

            prev_option = curr_option

        return ids_to_sample

    def non_maximum_suppression(self, bounding_boxes, iou_treshhold):
        filtered_boxes = []
        bounding_boxes = np.array(list(itertools.chain.from_iterable(bounding_boxes)), dtype=np.object)

        if(bounding_boxes.size == 0): #If no bounding boxes are detected can't do NMS
            return filtered_boxes


        confidences_indices = np.argsort(bounding_boxes[:, 2])
        while (len(confidences_indices) > 0):
            idx = confidences_indices[-1]
            current_bb = bounding_boxes[idx, 0]
            filtered_boxes.append(current_bb)
            confidences_indices = confidences_indices[:-1] #Remove last element from the list

            if(len(confidences_indices) == 0):
                break

            for idx2 in deepcopy(confidences_indices):
                if(self.iou_bbs(current_bb, bounding_boxes[idx2, 0]) > iou_treshhold): # Remove BB from list
                    confidences_indices = confidences_indices[confidences_indices != idx2]

        return filtered_boxes
    
    def prepare_lidar(self, tick_data):
        lidar_transformed = deepcopy(tick_data['lidar']) 
        lidar_transformed[:, 1] *= -1  # invert
        lidar_transformed = torch.from_numpy(lidar_to_histogram_features(lidar_transformed)).unsqueeze(0)
        lidar_transformed_degrees = [lidar_transformed.to('cuda', dtype=torch.float32)]
        lidar_bev = torch.cat(lidar_transformed_degrees[::-1], dim=1)
        return lidar_bev

    def prepare_goal_location(self, tick_data):
        tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
                                            torch.FloatTensor([tick_data['target_point'][1]])]
        # breakpoint()
        target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)

        target_point_image_degrees = []
        target_point_degrees = []
        for degree in self.aug_degrees:
            rad = np.deg2rad(degree)
            degree_matrix = np.array([[np.cos(rad), np.sin(rad)],
                                [-np.sin(rad), np.cos(rad)]])

            current_target_point = (degree_matrix @ target_point[0].cpu().numpy().reshape(2, 1)).T

            target_point_image = draw_target_point(current_target_point[0])
            target_point_image = torch.from_numpy(target_point_image)[None].to('cuda', dtype=torch.float32)
            target_point_image_degrees.append(target_point_image)
            target_point_degrees.append(torch.from_numpy(current_target_point))

        target_point_image = torch.cat(target_point_image_degrees, dim=0)
        target_point = torch.cat(target_point_degrees, dim=0).to('cuda', dtype=torch.float32)

        return target_point_image, target_point

    def prepare_image(self, tick_data):
        image = Image.fromarray(tick_data['rgb'])
        image_degrees = []
        for degree in self.aug_degrees:
            crop_shift = degree / 60 * self.config.img_width
            rgb = torch.from_numpy(self.shift_x_scale_crop(image, scale=self.config.scale, crop=self.config.img_resolution, crop_shift=crop_shift)).unsqueeze(0)
            image_degrees.append(rgb.to('cuda', dtype=torch.float32))
        image = torch.cat(image_degrees, dim=0)
        return image
    
    def shift_x_scale_crop(self, image, scale, crop, crop_shift=0):
        crop_h, crop_w = crop
        (width, height) = (int(image.width // scale), int(image.height // scale))
        im_resized = image.resize((width, height))
        image = np.array(im_resized)
        start_y = height//2 - crop_h//2
        start_x = width//2 - crop_w//2
        
        # only shift in x direction
        start_x += int(crop_shift // scale)
        cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
        cropped_image = np.transpose(cropped_image, (2,0,1))
        return cropped_image

    def update_gps_buffer(self, control, theta, speed):
        yaw = np.array([(theta - np.pi/2.0)])
        speed = np.array([speed])
        action = np.array(np.stack([control.steer, control.throttle, control.brake], axis=-1))

        #Update gps locations
        for i in range(len(self.gps_buffer)):
            loc =self.gps_buffer[i]
            loc_temp = np.array([loc[1], -loc[0]]) #Bicycle model uses a different coordinate system
            next_loc_tmp, _, _ = self.ego_model.forward(loc_temp, yaw, speed, action)
            next_loc = np.array([-next_loc_tmp[1], next_loc_tmp[0]])
            self.gps_buffer[i] = next_loc

        return None
    
    def scale_crop(self, image, scale=1, start_x=0, crop_x=None, start_y=0, crop_y=None):
        (width, height) = (image.width // scale, image.height // scale)
        if scale != 1:
            image = image.resize((width, height))
        if crop_x is None:
            crop_x = width
        if crop_y is None:
            crop_y = height
            
        image = np.asarray(image)
        cropped_image = image[start_y:start_y+crop_y, start_x:start_x+crop_x]
        return cropped_image
    
    # def _get_position(self, tick_data):
    #     gps = tick_data['gps']
    #     gps = (gps - self._route_planner.mean) * self._route_planner.scale
    #     return gps
    
    def _get_position(self, tick_data):
        gps = tick_data['gps']
        mean = np.array([0.0, 0.0]) # for carla 9.10
        scale = np.array([111324.60662786, 111319.490945]) # for carla 9.10
        gps = (gps - mean) * scale
        return gps

    

# Taken from LBC
class RoutePlanner(object):
    def __init__(self, min_distance, max_distance):
        self.saved_route = deque()
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.is_last = False

        self.mean = np.array([0.0, 0.0]) # for carla 9.10
        self.scale = np.array([111324.60662786, 111319.490945]) # for carla 9.10

    def set_route(self, global_plan, gps=False):
        self.route.clear()

        for pos, cmd in global_plan:
            if gps:
                pos = np.array([pos['lat'], pos['lon']])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = np.array([pos.location.x, pos.location.y])
                pos -= self.mean

            self.route.append((pos, cmd))

    def run_step(self, gps):
        if len(self.route) <= 2:
            self.is_last = True
            return self.route

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
            distance = np.linalg.norm(self.route[i][0] - gps)

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        return self.route

    def save(self):
        self.saved_route = deepcopy(self.route)

    def load(self):
        self.route = self.saved_route
        self.is_last = False


class SensorInterface(object):
    def __init__(self):
        self._sensors_objects = {}
        self._data_buffers = {}
        self._new_data_buffers = Queue()
        self._queue_timeout = 10

        # Only sensor that doesn't get the data on tick, needs special treatment
        self._opendrive_tag = None


    def register_sensor(self, tag, sensor_type, sensor):
        if tag in self._sensors_objects:
            raise SensorConfigurationInvalid("Duplicated sensor tag [{}]".format(tag))

        self._sensors_objects[tag] = sensor

        if sensor_type == 'sensor.opendrive_map': 
            self._opendrive_tag = tag

    def update_sensor(self, tag, data, timestamp):
        # print("Updating {} - {}".format(tag, timestamp))
        if tag not in self._sensors_objects:
            raise SensorConfigurationInvalid("The sensor with tag [{}] has not been created!".format(tag))

        self._new_data_buffers.put((tag, timestamp, data))

    def get_data(self):
        try: 
            data_dict = {}
            while len(data_dict.keys()) < len(self._sensors_objects.keys()):

                # Don't wait for the opendrive sensor
                if self._opendrive_tag and self._opendrive_tag not in data_dict.keys() \
                        and len(self._sensors_objects.keys()) == len(data_dict.keys()) + 1:
                    # print("Ignoring opendrive sensor")
                    break

                sensor_data = self._new_data_buffers.get(True, self._queue_timeout)
                # print("Getting {} - {}".format(sensor_data[0],sensor_data[1]))
                data_dict[sensor_data[0]] = ((sensor_data[1], sensor_data[2]))

        except Empty:
            raise SensorReceivedNoData("A sensor took too long to send their data")

        return data_dict
    

class SensorReceivedNoData(Exception):
    """
    Exceptions thrown when the sensors used by the agent take too long to receive data
    """

    def __init__(self, message):
        super(SensorReceivedNoData, self).__init__(message)

class SensorConfigurationInvalid(Exception):
    """
    Exceptions thrown when the sensors used by the agent are not allowed for that specific submissions
    """

    def __init__(self, message):
        super(SensorConfigurationInvalid, self).__init__(message)

# NOTE: end of adding transfuser model
#----------------------------------------------------------------






def main(save, save_pos, save_prob, save_brake, save_trajectory, save_key, save_time, init_pos, init_speed, alpha, epsilon, emergency_activate, num_walker):    
    try:
        client = carla.Client('localhost', 2026)
        client.set_timeout(2.0)

        world = client.get_world()

        global settings
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        spawn_points = world.get_map().get_spawn_points()  
        # print(world.get_map())

        ego_blueprint = blueprint_library.filter('model3')[0]

        # move spawn point to left side of road
        ego_spawn_point = spawn_points[1]
        ego_spawn_point.location.x += 75.0 + init_pos
        ego_spawn_point.location.y -= 3.0

        ego_vehicle = world.try_spawn_actor(ego_blueprint, ego_spawn_point) # 78
        print(ego_vehicle)

        spectator = world.get_spectator()
        # transform = carla.Transform(ego_vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)),ego_vehicle.get_transform().rotation)
        
        transform = carla.Transform(ego_vehicle.get_transform().transform(carla.Location(x=0, y=-0.2, z=0.8)),ego_vehicle.get_transform().rotation)

        # transform = carla.Transform(ego_vehicle.get_transform().transform(carla.Location(x=4,z=7.5)),ego_vehicle.get_transform().rotation)
        spectator.set_transform(transform)

        actor_list.append(ego_vehicle)

        # https://carla.readthedocs.io/en/latest/cameras_and_sensors
        # get the blueprint for this sensor
        blueprint = blueprint_library.find('sensor.camera.rgb')

        # change the dimensions of the image
        blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
        blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
        blueprint.set_attribute('fov', '110')


        # adjust sensor to be above vehicle
        spawn_point = carla.Transform(carla.Location(x=2.5, z=10.0), carla.Rotation(pitch=-90))

        # spawn the sensor and attach to vehicle.
        sensor = world.spawn_actor(blueprint, spawn_point, attach_to=ego_vehicle)

        # add sensor to list of actors    
        image_queue = queue.Queue()
        sensor.listen(image_queue.put)
        actor_list.append(sensor)

        print("start pid")

        # Pharuj attempt
        model = initialize_model(world)




        # =================define sensors amd listen==========================
        sensors = []
        # attach camera
        cam_location = carla.Location(x=1.3, y=0.0, z=2.3)
        rotation_front = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        rotation_left = carla.Rotation(pitch=0.0, yaw=0.0, roll=-60.0)
        rotation_right = carla.Rotation(pitch=0.0, yaw=0.0, roll=60.0)

        rgb_front = add_sensor('sensor.camera.rgb', world, cam_location, rotation_front, 'rgb_front', ego_vehicle, sensors)
        rgb_front.listen(lambda image: _parse_image_cb(image, 'rgb_front')) 
        
        rgb_left = add_sensor('sensor.camera.rgb', world, cam_location, rotation_left, 'rgb_left', ego_vehicle, sensors)
        rgb_left.listen(lambda image: _parse_image_cb(image, 'rgb_left')) 
        
        rgb_right = add_sensor('sensor.camera.rgb', world, cam_location, rotation_right, 'rgb_right', ego_vehicle, sensors)
        rgb_right.listen(lambda image: _parse_image_cb(image, 'rgb_right')) 
        
        depth_front = add_sensor('sensor.camera.depth', world, cam_location, rotation_front, 'depth_front', ego_vehicle, sensors)
        depth_front.listen(lambda image: _parse_image_cb(image, 'depth_front')) 
        
        semantics_front = add_sensor('sensor.camera.semantic_segmentation', world, cam_location, rotation_front, 'semantics_front', ego_vehicle, sensors)
        semantics_front.listen(lambda image: _parse_image_cb(image, 'semantics_front')) 
        
        # attach imu, gps, lidar
        location = carla.Location(x=0.0, y=0.0, z=0.0)
        rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)

        imu = add_sensor('sensor.other.imu', world, location, rotation, 'imu', ego_vehicle, sensors)
        imu.listen(lambda imu_data: _parse_imu_cb(imu_data, 'imu'))

        gps = add_sensor('sensor.other.gnss', world, location, rotation, 'gps', ego_vehicle, sensors)
        gps.listen(lambda gnss_data: _parse_gnss_cb(gnss_data, 'gps'))


        lidar_location = carla.Location(x=1.3, y=0.0, z=2.5)
        # lidar_rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=-90.0) # by configuration
        lidar_rotation = carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0)
        lidar = add_sensor('sensor.lidar.ray_cast', world, lidar_location, lidar_rotation, 'lidar', ego_vehicle, sensors)

        lidar.listen(lambda lidar_data: _parse_lidar_cb(lidar_data, 'lidar'))


        safe_controller(ego_vehicle, world, image_queue, spawn_points, init_pos, init_speed, alpha, epsilon, emergency_activate, num_walker, save_time, sensors, model)


    finally:
        print('destroying sensors')
        for sensor in sensors:
            sensor.stop()
            sensor.destroy()
            
        print('destroying actors')
        for actor in actor_list:
            actor.destroy()

        print("destroying walkers")
        for walker in curr_walkers:
            walker.destroy()
            
        if save_trajectory:
            open("transfuser_velocity.txt", "w").close()
            with open("transfuser_velocity.txt", 'a') as f:
                f.write(" ".join([str(i) for i in velocity_stats]))
                f.write('\n')

            open("transfuser_u.txt", "w").close()
            with open("transfuser_u.txt", 'a') as f:
                f.write(" ".join([str(i) for i in u_stats]))
                f.write('\n')
            
            open("transfuser_F.txt", "w").close()
            with open("transfuser_F.txt", 'a') as f:
                f.write(" ".join([str(i) for i in safety_probability]))
                f.write('\n')
            
            open("transfuser_position.txt", "w").close()
            with open("transfuser_position.txt", 'a') as f:
                f.write(" ".join([str(i) for i in position_stats]))
                f.write('\n')


        if save_pos:
            plt.plot(position_stats)
            plt.ylabel('position')
            plt.xlabel('time')
            plt.savefig('pos_stats_' + str(int(time.time())) + '.png')
            plt.clf()
            print('stats saved')
            
        if save_brake:
            # graph brake stats
            plt.plot(velocity_stats)
            plt.plot(u_stats)
            plt.ylabel('velocity & control')
            plt.xlabel('time')
            plt.savefig('brake_stats_' + str(int(time.time())) + '.png')
            plt.clf()
            print('brake stats saved')
            
        if save_prob:
            plt.plot(safety_probability)
            plt.ylabel('safety probability')
            plt.xlabel('time')
            plt.savefig('safety_prob_' + str(int(time.time())) + '.png')
            plt.clf()
            print('safety probability saved')

        if save:
            # save frames
            # retrieve everything from image queue
            print('retrieving images' + str(image_queue.qsize()))
            while not image_queue.empty():
                image = image_queue.get()
                process_img(image, world)
            for i in range(len(frames)):
                cv2.imwrite(save_path + 'frame' + str(i) + '.png', frames[i])

        print('done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save', type=bool, default=False, help='Save frames')
    parser.add_argument('--save_path', type=str, default='frames/', help='Path to save frames')
    parser.add_argument('--save_pos', type=bool, default=False, help='Save velocity, brake, and position stats')
    parser.add_argument('--save_brake', type=bool, default=False, help='Save brake stats')
    parser.add_argument('--save_prob', type=bool, default=False, help='Save safety probability')
    parser.add_argument('--init_pos', type=float, default=0, help='Initial position of vehicle')
    parser.add_argument('--init_speed', type=float, default=0, help='Initial speed of vehicle')
    parser.add_argument('--alpha', type=float, default=0.2, help='Safe controller parameter')
    parser.add_argument('--epsilon', type=float, default=0.05, help='Safety tolerance')
    parser.add_argument('--emergency_activate', type=bool, default=True, help='Emergency stop controller is activated')
    parser.add_argument('--num_walker', type=int, default=8, help='Number of walker spawned')
    parser.add_argument('--save_file', type=str, default='safe_control.txt', help='File to save safe status')
    parser.add_argument('--save_trajectory', type=bool, default=False, help='Save brake stats')
    parser.add_argument('--save_key', type=bool, default=False, help='Save closet key stats')
    parser.add_argument('--save_time', type=bool, default=False, help='Save time horizon')

    args = parser.parse_args()
    save = args.save
    save_path = args.save_path
    save_pos = args.save_pos
    save_brake = args.save_brake
    save_prob = args.save_prob
    init_pos = args.init_pos
    init_speed = args.init_speed
    alpha = args.alpha
    epsilon = args.epsilon
    emergency_activate = args.emergency_activate
    num_walker = args.num_walker
    save_file = args.save_file
    save_trajectory = args.save_trajectory
    save_key = args.save_key
    save_time = args.save_time

    main(save, save_pos, save_prob, save_brake, save_trajectory, save_key, save_time, init_pos, init_speed, alpha, epsilon, emergency_activate, num_walker)
