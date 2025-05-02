import cv2
import numpy as np
from rasrobot import RASRobot
from lane_controller_simple import LaneController

# Function to get the object in the image and classify it as traffic light or sign, Then show the image with bounding boxes (just for visualization)
def get_object(rgb_image,depth_channel):
        global step
        road_size=67
        # Analyze only the first "road_size" rows of the depth channel
        depth_region = depth_channel[:road_size, :]

        # Apply a threshold to isolate objects within a specific depth range (hyperparameter, in this case only objects closer than 30)
        _, thresholded = cv2.threshold(depth_region, 0, 30, cv2.THRESH_BINARY)

        # Perform morphological operations to disconnect objects
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find contours in the processed image
        contours, _ = cv2.findContours(opened.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the full RGB image for drawing bounding boxes
        output_image = rgb_image[:, :, :3].copy()

        # Loop through all detected contours and draw bounding boxes
        for contour in contours:
            traffic_light_colors={'red':0,'yellow':0,'green':0}
            total_black=0
            # Get the bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)
            
            #Extract the region of interest (ROI) within the bounding box
            roi = rgb_image[y:y + h, x:x + w]
            # Preprocess the ROI for better color detection
            roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)  # Reduce noise
            roi_hsv = cv2.cvtColor(roi_blurred, cv2.COLOR_BGR2HSV)

            # Define color ranges in HSV
            # Red (dual ranges because red wraps around 0-180)
            lower_red1 = np.array([0, 30, 30])     # Lower range for red
            upper_red1 = np.array([10, 255, 255])  # Upper range for red
            lower_red2 = np.array([170, 30, 30])   # Lower range for red
            upper_red2 = np.array([180, 255, 255]) # Upper range for red

            # Green
            lower_green = np.array([35, 50, 50])   # Lower range for green
            upper_green = np.array([85, 255, 255]) # Upper range for green

            # Yellow
            lower_yellow = np.array([20, 100, 100])  # Lower range for yellow
            upper_yellow = np.array([35, 255, 255]) # Upper range for yellow

            lower_black = np.array([0, 0, 0])       # Lower range for black
            upper_black = np.array([180, 60, 60])   # Upper range for black

            black_mask = cv2.inRange(roi_hsv, lower_black, upper_black)

            # Create masks for red, green, and yellow
            red_mask1 = cv2.inRange(roi_hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(roi_hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            green_mask = cv2.inRange(roi_hsv, lower_green, upper_green)
            yellow_mask = cv2.inRange(roi_hsv, lower_yellow, upper_yellow)


            # Count the number of red, yellow, and green pixels in the ROI
            traffic_light_colors['red'] += np.sum(red_mask == 255)
            traffic_light_colors['yellow'] += np.sum(yellow_mask == 255)
            traffic_light_colors['green'] += np.sum(green_mask == 255)
            total_black += np.sum(black_mask == 255)

            traffic_light_status = None
            if (traffic_light_colors['red']>10 and traffic_light_colors['red']<30 
                or traffic_light_colors['yellow']>4 and traffic_light_colors['yellow']<10 
                or traffic_light_colors['green']>5 and traffic_light_colors['green']<30) and total_black>5:
                traffic_light_status = max(traffic_light_colors, key=traffic_light_colors.get) 

            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            if traffic_light_status:
                cv2.putText(output_image, f'{traffic_light_status}_TL', (x, y-5), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(output_image, 'Sign', (x, y-5), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
           

        name = 'Full Image with Bounding Boxes'
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, output_image.shape[1]*2, output_image.shape[0]*2)
        cv2.imshow(name, output_image)

        cv2.waitKey(1)


class MyRobot(RASRobot):
    def __init__(self):
        """
        The constructor has no parameters.
        """
        super(MyRobot, self).__init__()

    def run(self):
        """
        This function implements the main loop of the robot.
        """
        debug = True
        # initialise lane controller
        lc = LaneController(debug=debug)  # set debug to True if you want visualisation and logs

        while self.tick():
            # get front camera from car and display it
            image = self.get_camera_image()

            depth = self.get_camera_depth_image()
            depth = np.nan_to_num(depth, posinf=0, neginf=255)
            if debug:
                copy_img=image.copy()
                get_object(copy_img,depth.astype(np.uint8))
    
            # use lane controller to determine steering angle and speed
            steering_angle, speed = lc.get_angle_and_speed(image, depth)
            # set steering angle and speed of car (will be applied in next tick)
            self.set_steering_angle(steering_angle)
            self.set_speed(speed)

if __name__ == '__main__':
    # create a robot and let it run!
    robot = MyRobot()
    robot.run()