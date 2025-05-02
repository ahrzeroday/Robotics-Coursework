import numpy as np
import kinpy as kp
from scipy.spatial.transform import Rotation as R
from rasrobot import RASRobot, TIME_STEP


class UR5e(RASRobot):
    def __init__(self,DEBUG=False):
        """
        This is your main robot class. It inherits from RASRobot for access
        to devices such as motors, position sensors, and camera.
        It provides some functions for your conveniene.
        You can change this class as you wish.
        However, you are not allowed to access any private fields of the
        superclass RASRobot.
        """
        super().__init__()
        
        # load the kinematic chain based on the robot's URDF file
        end_link = 'TCP'  # link used for forward and inverse kinematics
        # TCP = Tool Center Point - this is the point between the fingers
        URDF_FN = '../../resources/ur5e_2f85_camera.urdf'
        self.chain = kp.build_serial_chain_from_urdf(open(URDF_FN), end_link)

        self.grid_cols = 5  # Number of columns in grid
        self.grid_rows = 5  # Number of rows in grid
        self.seen_chunks = set()  # Track visited chunks
        self.available_chunks = list(range(self.grid_cols * self.grid_rows)) 
        self.DEBUG = DEBUG  # Enable debug messages
        # print chain on console
        if self.DEBUG:
            print('kinematic chain:')
            print(self.chain)
            print(f'The end link of the chain is <{end_link}>.')
            print('All computations of forward and inverse kinematics apply to this link.')
        
        
    @property
    def home_pos(self):
        """ 
        this is the home configuration of the robot 
        """
        return [1.57, -1.57, 1.57, -1.57, -1.57, 0.0]
    
    @property
    def basket_pos(self):
        """ 
        this is the basket configuration of the robot 
        """
        return [3, -1, 0.9, -1.57, -1.57, 0.0]

    def joint_pos(self):
        """
        :return: (6,) ndarray, the current joint position of the robot
        """
        joint_pos = np.asarray([m.getPositionSensor().getValue() for m in self.motors])
        return joint_pos
        
    def move_to_joint_pos(self, target_joint_pos, timeout=5, velocity=0.8,sync=True):
        """
        synchronised PTP motion
        blocking behaviour, moves the robot to the desired joint position.
        :param target_joint_pos: list/ndarray with joint configuration
        :param timeout: float, timeout in seconds after which this function returns latest
        :param velocity: float, target joint velocity in radians/second
        :return: bool, True if robot reaches the target position
                  else will return False after timeout (in seconds)
        """
        if len(target_joint_pos) != len(self.motors):
            raise ValueError('target joint configuration has unexpected length')
            
        abs_diffs = np.abs(target_joint_pos - self.joint_pos())
        velocity_gains = abs_diffs / np.max(abs_diffs)
        if sync:
            for pos, gain, motor in zip(target_joint_pos, velocity_gains, self.motors):
                motor.setPosition(pos)
                motor.setVelocity(gain * velocity)
        else:
            for pos, motor in zip(target_joint_pos, self.motors):
                motor.setPosition(pos)
                motor.setVelocity(velocity)
            
        # step through simulation until timeout or position reache
        for step in range(int(timeout * 1000) // TIME_STEP):
            self.step()

            # check if the robot is close enough to the target position
            if all(abs(target_joint_pos - self.joint_pos()) < 0.001):
                return True
        if self.DEBUG:
            print('Timeout. Robot has not reached the desired target position.')
        return False
        
    def forward_kinematics(self, joint_pos=None):
        """
        computes the pose of the chain's end link for given joint position.
        :param joint_pos: joint position for which to compute the end-effector pose
                          if None given, will use the robot's current joint position
        :return: kinpy.Transform object with pos and rot
        """
        if joint_pos is None:
            joint_pos = self.joint_pos()
            
        ee_pose = self.chain.forward_kinematics(joint_pos)
        return ee_pose
        
    def inverse_kinematics(self, target_pose):
        """
        Computes a joint configuration to reach the given target pose.
        Note that the resulting joint position might not actually reach the target
        if the target is e.g. too far away.
        :param target_pose: kinpy.Transform, pose of the end link of the chain
        :return: list/ndarray, joint position
        """
        ik_result = self.chain.inverse_kinematics(target_pose, self.joint_pos())
        return ik_result
    
    # this function helps to find best position for the robot to reach the box in z axis (when two boxes are stacked on top of each other)
    # it calculates the distance from the box in z axis and returns the best position for the robot to reach the box
    # the distance is calculated based on the distance from the box in z axis and the distance from the ground
    def get_distance_from_box(self,box_dis):
        """
        :param box_dis: distance from the box in z axis
        :return: best position for the robot to reach the box
        """
        current_pos = self.forward_kinematics()
        # calculate the distance from the box in z axis, it divided by 0.06 because the size of the box is 0.06, So if two boxes are stacked on top of each other,
        # it can calculate the distance correctly
        real_distance = abs(0.5-box_dis)//0.06
        if self.DEBUG:
            print(f"box distance: {box_dis}")
            print(f"current distance: {current_pos.pos[-1]}")
            print(f"real distance: {real_distance}")
            print(f"Calculated distance: {0.01+((real_distance-1)*0.06)}")

        # if the distance is less than 0.01, it will set the distance to 0.01, because the robot's hand need some space to grasp the box
        current_pos.pos[-1] = 0.01+((real_distance-1)*0.06)
        return current_pos
        
    # this function moves the arm to a random position in the grid
    def random_chunk_search(self, chunk_width=128, chunk_height=68, distance_from_ground=0.3):
        """
        Move to a random chunk in the grid.
        :param chunk_width: Width of each chunk in pixels
        :param chunk_height: Height of each chunk in pixels
        :param distance_from_ground: Height of the end effector from the ground
        :return: True if movement was successful, False otherwise
        """
        total_chunks = self.grid_cols * self.grid_rows
        
        # Reset if we've seen all chunks
        if not self.available_chunks:
            if self.DEBUG:
                print(f"All {total_chunks} chunks visited, resetting search cycle")
            self.available_chunks = list(range(total_chunks))
            self.seen_chunks = set()
        
        # Randomly select from remaining chunks
        selected_chunk = np.random.choice(self.available_chunks)
        self.available_chunks.remove(selected_chunk)
        self.seen_chunks.add(selected_chunk)
        if self.DEBUG:
            print(f"Searching in chunk {selected_chunk+1}/{total_chunks} (unvisited: {len(self.available_chunks)})")
        
        # Calculate grid position (n columns × n rows) (varibles are hyperparameters)
        col = selected_chunk % self.grid_cols  
        row = selected_chunk // self.grid_cols 
        
        # Conversion factors (calibrate these)
        pixel_to_meter_x = 0.0015
        pixel_to_meter_y = 0.001
        
        # Calculate offsets from center (home position)
        # Now using 4 columns: positions -1.5, -0.5, +0.5, +1.5 times chunk width
        # And 4 rows: positions +1.5, +0.5, -0.5, -1.5 times chunk height
        chunk_offset_x = (col - 1.5) * chunk_width
        chunk_offset_y = (1.5 - row) * chunk_height
        
        # Convert to robot coordinates
        move_x = chunk_offset_x * pixel_to_meter_x
        move_y = chunk_offset_y * pixel_to_meter_y
        
        # Get home position in Cartesian space
        home_pose = self.forward_kinematics(self.home_pos)
        
        # Create target position (fixed Z)
        target_pos = np.array([
            home_pose.pos[0] + move_x,
            home_pose.pos[1] + move_y,
            distance_from_ground
        ])
        
        # Keep home orientation
        target_pose = kp.Transform(pos=target_pos, rot=home_pose.rot)
        
        try:
            # Move to target
            target_joint_pos = self.inverse_kinematics(target_pose)
            success = self.move_to_joint_pos(target_joint_pos)
            # Check if movement was successful, just a safety check
            if success:
                if self.DEBUG:
                    print(f"Moved to chunk {selected_chunk+1} at X:{target_pos[0]:.3f}, Y:{target_pos[1]:.3f}")
                return True, selected_chunk
            return False, None
        except Exception as e:
            if self.DEBUG:
                print(f"Movement failed: {e}")
            # Return chunk to available list
            self.available_chunks.append(selected_chunk)
            self.seen_chunks.remove(selected_chunk)
            return False, None

    # this function adjusts the orientation of the end effector to face the cube
    # it rotates the end effector around the z axis to face the cube
    # the angle is calculated based on the cube's angle in the image
    def adjust_orientation_to_cube(self, cube_angle):
        """
        Adjust the end effector's orientation to face the cube
        :param cube_angle: Angle of the cube in degrees
        :return: True if movement was successful
        """
        # Get current joint positions
        current_joints = self.joint_pos()
        
        # Create a new joint position
        new_joints = current_joints.copy()
        new_joints[5] += np.deg2rad(cube_angle)  # Adjust the end effector's orientation
        
        # move to the new joint position
        return self.move_to_joint_pos(new_joints)
    
    # this function moves the robot to a position where the cube is at the bottom center of the camera view
    # Why? Because the arm's hand is at the bottom of the camera view
    # it calculates the distance from the cube in the image and moves the robot to the position where the cube is at the bottom center of the camera view
    # the distance is calculated based on the distance from the cube in the image and the distance from the camera
    def center_cube_at_bottom(self, cube_pixel_location, camera_params, y_offset_pixels=25):
        """
        Move the robot to position the cube at the bottom center of the camera view
        :param cube_pixel_location: Tuple (x, y, depth) of the cube in pixel coordinates
        :param camera_params: Dictionary containing camera intrinsic parameters:
                            {'fx': focal_length_x, 'fy': focal_length_y,
                            'cx': principal_point_x, 'cy': principal_point_y}
        :param y_offset_pixels: Distance from bottom of image (in pixels)
        :return: True if movement was successful
        """
        if cube_pixel_location is None:
            if self.DEBUG:
                print("No cube detected in the image")
            return False
        
        # Extract cube information
        cube_x, cube_y, cube_depth = cube_pixel_location
        
        # Get camera parameters
        fx = camera_params['fx']
        fy = camera_params['fy']
        cx = camera_params['cx']
        cy = camera_params['cy']
        image_height = cy * 2  # Assuming cy is the vertical center
        
        # Calculate desired position (center horizontally, near bottom vertically)
        target_x = cx  # Center horizontally
        target_y = image_height - y_offset_pixels  # Fixed position near bottom
        
        # Calculate required movement in image space
        move_x = cube_x - target_x
        move_y = cube_y - target_y
        
        # Convert image movement to robot movement
        robot_move_x = (move_x * cube_depth) / fx
        robot_move_y = (move_y * cube_depth) / fy
        
        # Get current end effector pose
        current_pose = self.forward_kinematics()
        
        # Create a new target pose
        target_pos = current_pose.pos.copy()
        target_pos[0] += robot_move_x  # X movement
        target_pos[1] -= robot_move_y  # Y movement
        
        
        # Create target pose (keep same rotation as current pose)
        target_pose = kp.Transform(pos=target_pos, rot=current_pose.rot)
        
        # Calculate inverse kinematics for the new pose
        try:
            target_joint_pos = self.inverse_kinematics(target_pose)
        except Exception as e:
            print(f"Inverse kinematics failed: {e}")
            return False
        
        # Move to the new joint position
        return self.move_to_joint_pos(target_joint_pos)