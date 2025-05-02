# Robotics Coursework Repository

This repository contains two coursework projects for my robotics course, implemented using the Webots simulator. The projects focus on robotic control and computer vision, showcasing a manipulator arm for a Pick-and-place with a robot manipulator and Perception and Behaviour Coordination.

## Table of Contents
- [Overview](#overview)
- [Coursework 1: Pick-and-place with a robot manipulator](#coursework-1-pick-and-place-with-a-robot-manipulator)
  - [Description](#description)
  - [Features](#features)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
  - [Usage](#usage)
- [Coursework 2: Perception and Behaviour Coordination](#coursework-2-perception-and-behaviour-coordination)
  - [Description](#description-1)
  - [Features](#features-1)
  - [Dependencies](#dependencies-1)
  - [Installation](#installation-1)
  - [Usage](#usage-1)

## Overview
This repository showcases two robotics coursework projects developed as part of my academic coursework. Both projects are implemented in the Webots simulator:
- **Coursework 1**: Controlling a UR5e robotic arm to pick up cubes and place them in a crate using depth camera data and kinematic control.
- **Coursework 2**: Navigating an autonomous vehicle through a road network, following lanes, detecting traffic lights, and handling crossroads.

The projects demonstrate skills in robot kinematics, computer vision, and state-based control for robotic manipulation and autonomous navigation.

## Coursework 1: Pick-and-place with a robot manipulator

### Description
**Control of a Robot Manipulator Arm for a Pick-and-place with a robot manipulator**

A UR5e robotic arm, equipped with a depth camera at its end-effector, detects and picks up five cubes randomly distributed in its workspace and places them into a fixed crate. The system uses the depth camera to locate cubes, computes inverse kinematics to position the arm, and controls the gripper to grasp and release objects. A state-based controller manages the sequence of actions, including searching, grasping, and placing.

Video:

![](\media\1.mp4)

Stacked cube:

![](\media\1_1.mp4)

**Learning Outcomes:**
1. Implement forward and inverse kinematics for robotic control.
2. Apply computer vision for object detection in a robotic context.
3. Develop a controller for a Pick-and-place with a robot manipulator.

### Features
- **Cube Detection**: Identifies cubes using depth camera images, filtering for square shapes and depth values.
- **State Machine**: Manages states like `moving_to_home`, `searching_cube`, `moving_to_cube`, `grasping_cube`, and `moving_to_basket`.
- **Kinematic Control**: Uses the `kinpy` library for forward and inverse kinematics to position the arm accurately.
- **Gripper Control**: Opens and closes the gripper to pick up and release cubes.
- **Search Strategy**: Employs a grid-based random search to locate cubes when none are detected.
- **Camera Calibration**: Converts pixel coordinates to real-world coordinates using camera parameters.

### Dependencies
- **Webots Simulator**: For running the simulation environment (version R2023b).
- **Python**: Version 3.10 recommended.
- **Libraries**:
  - `numpy`: Numerical computations.
  - `scipy`: Rotation transformations.
  - `kinpy`: Kinematic calculations.
  - `opencv-python`: Image processing and cube detection.
- **URDF File**: `ur5e_2f85_camera.urdf` defines the robot’s kinematic chain (included in `RAS_coursework_1/resources`).

### Installation
1. **Install Webots**:
   - Download and install Webots from [https://cyberbotics.com/](https://cyberbotics.com/).
   - Ensure Webots is configured to use Python 3.10.

2. **Create a Conda Environment**:
   ```bash
   conda create -n robotics python=3.10
   conda activate robotics
   ```

3. **Install Python Dependencies**:
   ```bash
   pip install numpy scipy kinpy opencv-python
   ```

4. **Clone the Repository**:
   ```bash
   git clone https://github.com/ahrzeroday/Robotics-Coursework.git
   cd robotics-coursework
   ```

5. **Set Up Webots Project**:
   - Copy the contents of the `RAS_coursework_1` folder to your Webots project directory.
   - Ensure the `resources/ur5e_2f85_camera.urdf` file is in the correct relative path (`../../resources/`).

### Usage
1. **Launch Webots**:
   - Open Webots and load the world file for the UR5e robot (`../../worlds/`).

2. **Run the Controller**:
   - In Webots, set the controller to use `ras.py` from the `coursework1` folder.
   - Alternatively, run the script directly if Webots is configured for external controllers:
     ```bash
     python RAS_coursework_1/controllers/ras/ras.py
     ```

3. **Simulation**:
   - The robot moves to the home position and opens the gripper.
   - It searches for cubes using the depth camera, centers detected cubes in the view, adjusts orientation, grasps, and places them in the crate.
   - If no cubes are found, it moves to a random grid position to continue searching.
   - The process repeats until all cubes are placed.

4. **Debugging**:
   - Set `DEBUG = True` in `ras.py` to enable debug messages (state transitions, kinematics).

## Coursework 2: Perception and Behaviour Coordination

### Description
**Autonomous Vehicle Lane Following and Traffic Light Navigation**

An autonomous vehicle navigates a road network in Webots, following lanes, detecting traffic lights (red, yellow, green), and handling crossroads. The vehicle uses a front-facing RGB and depth camera to detect the road and traffic lights. A lane controller adjusts steering and speed based on road center detection, while a state machine manages stopping at red lights and turning at crossroads.

Video:

![](\media\2.mp4)

**Learning Outcomes:**
1. Implement computer vision for lane detection and traffic light recognition.
2. Develop a reactive controller for lane following and speed adjustment.
3. Manage complex navigation behaviors using a state-based approach.

### Features
- **Lane Detection**: Identifies the road using HSV color filtering for tarmac and yellow markings.
- **Road Center Tracking**: Computes the road center to guide steering, with memory to smooth transitions.
- **Traffic Light Detection**: Detects red, yellow, and green lights using HSV color segmentation and depth filtering.
- **Crossroads Handling**: Recognizes crossroads and executes timed left or right turns.
- **State Machine**: Manages states like `driving`, `stopping`, `turning_left`, and `turning_right`.
- **Speed Control**: Adjusts speed based on steering angle to prevent tipping during turns.

### Dependencies
- **Webots Simulator**: For running the simulation environment (version R2023b).
- **Python**: Version 3.10 recommended.
- **Libraries**:
  - `numpy`: Numerical computations.
  - `opencv-python`: Image processing for lane and traffic light detection.
- **Webots Vehicle Controller**: Provided by `ras.py` (included in `RAS_coursework_2`).

### Installation
1. **Install Webots**:
   - Download and install Webots from [https://cyberbotics.com/](https://cyberbotics.com/).
   - Ensure Webots is configured to use Python 3.10.

2. **Create a Conda Environment** (if not already done for Coursework 1):
   ```bash
   conda create -n robotics python=3.10
   conda activate robotics
   ```

3. **Install Python Dependencies**:
   ```bash
   pip install numpy opencv-python
   ```

4. **Clone the Repository** (if not already done):
   ```bash
   git clone https://github.com/ahrzeroday/Robotics-Coursework.git
   cd robotics-coursework
   ```

5. **Set Up Webots Project**:
   - Copy the contents of the `RAS_coursework_2` folder to your Webots project directory.

### Usage
1. **Launch Webots**:
   - Open Webots and load the world file for the autonomous vehicle (`../../worlds/`).

2. **Run the Controller**:
   - In Webots, set the controller to use `ras.py` from the `RAS_coursework_2` folder.
   - Alternatively, run the script directly if Webots is configured for external controllers:
     ```bash
     python RAS_coursework_2/controllers/ras/ras.py
     ```

3. **Simulation**:
   - The vehicle starts in `driving` mode, following the road by tracking the center.
   - It detects traffic lights and stops at red lights, resuming when the light turns yellow or green.
   - At crossroads, it executes a timed right turn (configurable to left turn by modifying the code).
   - The vehicle adjusts speed based on steering angle to maintain stability.

4. **Debugging**:
   - Set `debug = True` in `ras.py` to enable visualization and logs.
   - Visualizations include the processed road image (with steering and crossroad rows marked) and bounding boxes around detected traffic lights or signs.
   - Console logs report traffic light status, crossroad detection, and mode changes.
