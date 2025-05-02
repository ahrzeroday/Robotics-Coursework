import numpy as np
import kinpy as kp
from scipy.spatial.transform import Rotation as R
import cv2
from util import *
from UR5e import UR5e


"""
MISSION 2: Control of a Robot Manipulator Arm for a Pick and Place Task
Learning outcomes: 1, 2, 3

Scenario:
A manipulator arm is equipped with a camera at its end-effector. 
The existing controller already provides behaviours to move
the robot in joint and/or task space and to open/close the gripper.
There are five cubes randomly distributed in the robot's work space,
and there is a crate, which is always in the same spot.

Task: 
Use the depth camera to detect the cubes. Implement a controller for the 
manipulator arm that clears the objects from the table and drops 
them into the crate.

Hints:
1) INSTALLATION
The project requires the kinpy library, which you should install into
your environment. It is an open source library, so you can directly 
look at the code on github.
Installation with conda & pip:
conda create -n M3 python=3.10
conda activate M3
pip install numpy scipy kinpy opencv-python

2) COORDINATE TRANSFORMS
You can move the robot in workspace by computing the inverse kinematics.
The kinpy.Transform object describes the pose in workspace. 
See more details here: 
https://github.com/neka-nat/kinpy/blob/master/kinpy/transform.py
It consists of a pos (position as [x, y, z]) and rot (rotation). 
The rotation is a quaternion in the [w, x, y, z] format. You can include
other libraries to convert between rotation representations, such as 
rotation matrices, axis-angle, and Euler angles. Just be aware that some
there is an alternative convention for the order of elements in a quaternion,
which is [x, y, z, w] - it is a common source of errors.
scipy.spatial.transform could be a good option, but it uses the [x, y, z, w]
convention, so you would need to convert the quaternions.
(Note: depending on your approach, you may not need this.)

3) GRASPING
Simulating contact-rich tasks is very difficult. You might notice that
the cubes sometimes act unexpectedly when being grasped.
It is generally a good idea to align the gripper as best as possible to the 
parallel surfaces of the cube, and to only move with low velocity when
picking up an object.
"""

# This is a flag to enable/disable debug messages
DEBUG = False

# camera resolution for creating the camera parameters
# (this should match the resolution of the camera used in the simulation)
image_width = 128  
image_height = 68 

# focal length in pixels for the camera to create the camera parameters, it helps to undersrand the camera's field of view and distance between the camera and arm

focal_length_pixels = (image_width / 2) / np.tan(np.deg2rad(60 / 2))  # 60 degrees is the field of view of the camera (hypothetical value)

# camera parameters for the UR5e robot
# These parameters are used to convert between pixel coordinates and real-world coordinates
# in the camera's field of view. The camera is assumed to be mounted at the end-effector of the robot.
# The parameters are defined as follows:
# fx: focal length in x direction (in pixels)
# fy: focal length in y direction (in pixels)
# cx: x-coordinate of the camera center (in pixels)
# cy: y-coordinate of the camera center (in pixels)
# The camera is assumed to be mounted at the end-effector of the robot.
camera_params = {
    'fx': focal_length_pixels,
    'fy': focal_length_pixels,
    'cx': image_width / 2,
    'cy': image_height / 2
}

def main():

    # initialise robot and move to home position
    robot = UR5e(DEBUG)
    # The behaviour coordinate of this robot is state-based, so we need to define the states
    current_state='moving_to_home'
    # cube_loc is the location of the cube in the camera frame ( if found by the camera)
    cube_loc=None

    # A loop that runs until the program is terminated
    while True:
        # print the current state of the robot
        if DEBUG:
            print('----------------------------------')
            print('Current state:', current_state)
        
        # if the robot is in the 'moving_to_home' state, move to the home position
        # and open the gripper
        if current_state=='moving_to_home':
            robot.move_to_joint_pos(robot.home_pos,sync=True)
            robot.open_gripper()
            current_state='searching_cube'

        # if the robot is in the 'searching_cube' state, get the camera depth image
        # and try to find the cube
        # if the cube is found, move to the 'moving_to_cube' state
        # otherwise, move to the 'searching_area' state
        elif current_state=='searching_cube':
            cube_loc=get_cubes(robot.get_camera_depth_image())
            if cube_loc is not None:
                current_state='moving_to_cube'
            else:
                current_state='searching_area'
        
        # if the robot is in the 'searching_area' state, move to a random position
        # and try to find the cube again
        # if the cube is found, move to the 'moving_to_cube' state
        # otherwise, stay in the 'searching_area' state
        elif current_state=='searching_area':
            robot.random_chunk_search()
            cube_loc=get_cubes(robot.get_camera_depth_image())
            if cube_loc is not None:
                current_state='moving_to_cube'
            else:
                current_state='searching_area'

        # if the robot is in the 'moving_to_cube' state, center the cube at the bottom of view because arm's hand is at the bottom
        # and adjust the orientation to the cube
        # move to the position above the cube
        # and move to the 'grasping_cube' state
        # if the cube is not found, move to the 'searching_area' state to find another area, maybe the cubes are in the other area
        elif current_state=='moving_to_cube':
            
            if cube_loc is not None:
                robot.center_cube_at_bottom(cube_loc[0:-1],camera_params)
                robot.adjust_orientation_to_cube(cube_loc[-1])
                best_pos = robot.get_distance_from_box(cube_loc[2])
                joint_pos = robot.inverse_kinematics(best_pos)
                robot.move_to_joint_pos(joint_pos)
                current_state='grasping_cube'
            else:
                current_state='searching_area'

        # if the robot is in the 'grasping_cube' state, close the gripper
        # and move to the position above the cube for 0.3m to avoid collision then move to the 'moving_to_basket' state
        elif current_state=='grasping_cube':
            robot.close_gripper(2)
            best_pos = robot.forward_kinematics()
            best_pos.pos[-1]=0.3
            joint_pos = robot.inverse_kinematics(best_pos)
            robot.move_to_joint_pos(joint_pos, velocity=0.1, timeout=20)
            current_state='moving_to_basket'

        # if the robot is in the 'moving_to_basket' state, move to the basket position
        # and open the gripper to drop the cube into the basket
        # then move to the 'moving_to_home' state and set cube_loc to None because the cube is dropped
        elif current_state=='moving_to_basket':
            robot.move_to_joint_pos(robot.basket_pos, velocity=0.1, timeout=20)
            robot.open_gripper(2)
            current_state='moving_to_home'
            cube_loc=None
        
        else:
            current_state='moving_to_home'
        
    
        
if __name__ == '__main__':
    main()
 