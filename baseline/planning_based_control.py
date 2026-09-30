import glob
import os
import sys
import queue
import matplotlib.pyplot as plt
import argparse
import csv

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

import logging

IM_WIDTH = 640
IM_HEIGHT = 960

u_stats = []
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

def process(ego_vehicle, world, image_queue, spawn_points, init_pos, init_speed, num_walker, save_time):
    # random spawn walker with normal distribution
    spawn_ticks = np.zeros(num_walker)
    for i in range(num_walker):
        mean_first_walker = 30
        std_first_walker = 50
        t_max_first_walker = 200

        mean_interval = 120
        std_interval = 50
        t_max_interval = 300
        if i == 0:
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

    for walker_id in range(num_walker):
        spawn_walker(world, ego_vehicle, init_pos, walker_id)

    occlusion = spawn_occlusion(world, ego_vehicle, init_pos)
    occlusion_dim = occlusion.bounding_box.extent

    # set initial speed
    ego_vehicle.set_target_velocity(carla.Vector3D(x=init_speed, y=0.0, z=0.0))

    tick_count = 0
    spawned = np.zeros(num_walker) # boolean to check if the nth walker is spawned

    time_horizon = 10000

    lookup_table = {}
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
        # start walker
        for i in range(num_walker):
            if tick_count == spawn_ticks[i] and spawned[i] == 0:
                print("start walker", (i+1)," at tick: ", spawn_ticks[i])
                walk_dir = carla.Vector3D(x=0 , y=-1, z=0)
                curr_walkers[i].apply_control(carla.WalkerControl(direction = walk_dir, speed=1.0))
                spawned[i] = 1

        # get current state of ego vehicle
        pos = ego_vehicle.get_location().x - spawn_points[1].location.x + init_pos # default = 0.0
        speed = ego_vehicle.get_velocity().x

        velocity_stats.append(speed)
        velocity_y.append(ego_vehicle.get_velocity().y)
        position_stats.append(pos)

        key, F = find_closest_table_entry(pos, speed, lookup_table)
        
        # save stats
        safety_probability.append(F)

        # check visibility
        visible = False
        for curr_walker in curr_walkers:            
            visible = is_visible(curr_walker, occlusion, occlusion_dim, ego_vehicle)
            if visible: # as long as one walker is in sight
                break
        
        # planning based control
        if pos > 75:
            if visible:
                control = carla.VehicleControl(throttle=0.0, brake=0.02)
            else:
                throttle = 0.5 if speed < 5.2 else 0
                control = carla.VehicleControl(throttle=throttle, brake=0.0)
        else:
            target_speed = min(5.2, (78 - pos) / 16)
            if speed < target_speed:
                if speed > 2.5:
                    throttle = max((target_speed - speed) / 4, 0.4)
                else:
                    throttle = max((target_speed - speed) / 5, 0.3)
                control = carla.VehicleControl(throttle=throttle, brake=0.0)
            elif speed > target_speed + 0.2:
                control = carla.VehicleControl(throttle=0.0, brake=0.02)
            else:
                control = carla.VehicleControl(throttle=0.0, brake=0.0)

        if control is not None:
            logging.info(control)
            ego_vehicle.apply_control(control)

        # check for collision
        for walker in curr_walkers:
            if ego_vehicle.get_location().distance(walker.get_location()) < 3.0:
                print("======collision detected======")
                print("tick: ", tick_count)
                with open('planning_control.txt', 'a') as f:
                    f.write('unsafe\n')
                # set ego vehicle to stop
                control = carla.VehicleControl(throttle=0.0, brake=1.0)
                ego_vehicle.apply_control(control)
                return

        # check if vehicle is at the end of the road
        if ego_vehicle.get_location().x > spawn_points[1].location.x + 100.0 - init_pos:
            print("end of road")
            print("=======safe!!!=======")
            print("tick: ", tick_count)
            if save_time:
                with open('planning_control_time.txt', 'a') as f:
                    f.write(str(tick_count) + '\n')
            with open('planning_control.txt', 'a') as f:
                f.write('safe\n')
            return
        

def main(save, save_pos, save_brake, save_trajectory, save_time, init_pos, init_speed, num_walker):
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

        ego_blueprint = blueprint_library.filter('model3')[0]

        # move spawn point to left side of road
        ego_spawn_point = spawn_points[1]
        ego_spawn_point.location.x += 75.0 + init_pos
        ego_spawn_point.location.y -= 3.0

        ego_vehicle = world.try_spawn_actor(ego_blueprint, ego_spawn_point) # 78
        print(ego_vehicle)

        spectator = world.get_spectator()
        transform = carla.Transform(ego_vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),ego_vehicle.get_transform().rotation)
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

        print("start planning-based control")

        process(ego_vehicle, world, image_queue, spawn_points, init_pos, init_speed, num_walker, save_time)

    finally:
        print('destroying actors')
        for actor in actor_list:
            actor.destroy()

        print("destroying walkers")
        for walker in curr_walkers:
            walker.destroy()
            
        if save_trajectory:
            open("planning_velocity.txt", "w").close()
            with open("planning_velocity.txt", 'a') as f:
                f.write(" ".join([str(i) for i in velocity_stats]))
                f.write('\n')
            
            open("planning_u.txt", "w").close()
            with open("planning_u.txt", 'a') as f:
                f.write(" ".join([str(i) for i in u_stats]))
                f.write('\n')
            
            open("planning_F.txt", "w").close()
            with open("planning_F.txt", 'a') as f:
                f.write(" ".join([str(i) for i in safety_probability]))
                f.write('\n')
            
            open("planning_position.txt", "w").close()
            with open("planning_position.txt", 'a') as f:
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
            plt.ylabel('velocity & control')
            plt.xlabel('time')
            plt.savefig('brake_stats_' + str(int(time.time())) + '.png')
            plt.clf()
            print('brake stats saved')

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

    parser.add_argument('--save', action='store_true', help='Save frames')
    parser.add_argument('--save_path', type=str, default='frames/', help='Path to save frames')
    parser.add_argument('--save_pos', action='store_true', help='Save velocity, brake, and position statistics')
    parser.add_argument('--save_brake', action='store_true', help='Save brake statistics')
    parser.add_argument('--init_pos', type=float, default=0, help='Initial position of vehicle')
    parser.add_argument('--init_speed', type=float, default=0, help='Initial speed of vehicle')
    parser.add_argument('--num_walker', type=int, default=8, help='Number of walker spawned')
    parser.add_argument('--save_file', type=str, default='planning_control.txt', help='File to save safe status')
    parser.add_argument('--save_trajectory', action='store_true', help='Save trajectory statistics')
    parser.add_argument('--save_time', action='store_true', help='Save time horizon')

    args = parser.parse_args()

    save = args.save
    save_path = args.save_path
    save_pos = args.save_pos
    save_brake = args.save_brake
    init_pos = args.init_pos
    init_speed = args.init_speed
    num_walker = args.num_walker
    save_file = args.save_file
    save_trajectory = args.save_trajectory
    save_time = args.save_time

    main(save, save_pos, save_brake, save_trajectory, save_time, init_pos, init_speed, num_walker)
