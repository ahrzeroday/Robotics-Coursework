#
# PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
# PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
# PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
#
from controller import Supervisor
import numpy as np

from util import display_image, normalize_depth

TIME_STEP = 32


class RASRobot:
    #
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    #
    def __init__(self):
        self.__sup = Supervisor()
        self.__sup.getDevice("depth").enable(TIME_STEP)        
        self.__total_cubes = 5
        
        # set motor velocity, initialise sensors
        self.motors = [
            self.__sup.getDevice("shoulder_pan_joint"),
            self.__sup.getDevice("shoulder_lift_joint"),
            self.__sup.getDevice("elbow_joint"),
            self.__sup.getDevice("wrist_1_joint"),
            self.__sup.getDevice("wrist_2_joint"),
            self.__sup.getDevice("wrist_3_joint"),
        ]
        for m in self.motors:
            m.getPositionSensor().enable(TIME_STEP)
            m.setVelocity(0.8)
        
        # initialise fingers
        self.__fingers = [
            self.__sup.getDevice('ROBOTIQ 2F-85 Gripper::left finger joint')
            # right finger mimics the left finger, so we only need to control one of them
        ]
        for finger in self.__fingers:
            finger.setVelocity(0.8)
            
        # shuffle the cubes
        self.__reset_scene()
         
    #
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    #   
    def __reset_scene(self):
        rng = np.random.default_rng()
        
        for i in range(self.__total_cubes):
            box = self.__sup.getFromDef(f'BOX{i+1}')
            rotation_field = box.getField('rotation')
            quaternion = rng.standard_normal(4)
            quaternion = quaternion / np.linalg.norm(list(quaternion))
            rotation_field.setSFRotation(quaternion.tolist())
     
    #
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    #       
    def close_gripper(self, timeout=1.2):
        """
        blocking behaviour that will close the gripper
        """
        for finger in self.__fingers:
            finger.setTorque(finger.getAvailableTorque()/2)
            
        for step in range(int(timeout * 1000) // TIME_STEP):
            self.step()
    
    #
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    #        
    def open_gripper(self, timeout=1.2):
        """
        blocking behaviour that will open the gripper
        """
        for finger in self.__fingers:
            finger.setPosition(0)
            
        for step in range(int(timeout * 1000) // TIME_STEP):
            self.step()
    
    #
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    #        
    def step(self):
        """
        step function of the simulation
        """
        self.__sup.step()
        depth = self.get_camera_depth_image()
        display_image(normalize_depth(depth), 'depth')
    
    #
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    # PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    #        
    def get_camera_depth_image(self):
        """ 
        This method returns a 2-dimensional array containing the depth of each 
        pixel. RGB camera image and depth image are aligned.
        
        :returns: (64, 128) ndarray
        """
        device = self.__sup.getDevice("depth")
        ret = device.getRangeImage(data_type="buffer")
        ret = np.ctypeslib.as_array(ret, (device.getHeight(), device.getWidth()))
        return ret

