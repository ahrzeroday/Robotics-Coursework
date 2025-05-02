#
# PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
# PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
# PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
#
import numpy as np

from controller import Robot
from vehicle import Driver

import cv2

class RASRobot(object):
    """
    This is the class you will use to interact with the car.
    PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    """
    def __init__(self):
        self.__robot = Driver()
        self.__timestep = int(self.__robot.getBasicTimeStep())

        self.__camera = self.__robot.getDevice("camera")
        self.__camera.enable(self.__timestep)
        
        self.__depth = self.__robot.getDevice("depth")
        self.__depth.enable(self.__timestep)

    def get_camera_image(self):
        """
        This method returns a NumPy array representing the latest image captured by the car's camera.
        It will have 64 rows, 128 columns and 4 channels (red, green, blue, alpha).
        """
        return np.frombuffer(self.__camera.getImage(), np.uint8).reshape((128,256,4))[:,:,:4]

    def get_camera_focal_length(self):
        """ This method returns the focal length of the camera in pixels"""
        return self.__camera.getFocalLength()

    def get_camera_depth_image(self):
        """ This method returns a 2-dimensional array containing the depth of each pixel, as if the sensor were an RGBD device."""
        ret = self.__depth.getRangeImage(data_type="buffer")
        ret = np.ctypeslib.as_array(ret, (self.__depth.getHeight(),self.__depth.getWidth()))
        return ret


    def set_steering_angle(self, angle):
        """
        This is just a proxy for the Webot's API call. It sets the steering angle of the car.
        For more information: https://cyberbotics.com/doc/automobile/driver-library?tab-language=python#wbu_driver_set_steering_angle
        """
        self.__robot.setSteeringAngle(angle)

    def set_speed(self, speed):
        """
        This is just a proxy for the Webot's API call. It sets the speed of the car.
        For more information: https://cyberbotics.com/doc/automobile/driver-library?tab-language=python#wbu_driver_set_cruising_speed
        """
        self.__robot.setCruisingSpeed(speed)
    
    def tick(self):
        """
        You will call this method rather than the typical `step` method used by regular Webots controllers.
        """
        if self.__robot.step() == -1:
            return False

        return True
        
