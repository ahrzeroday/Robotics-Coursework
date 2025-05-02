import numpy as np
import cv2


class LaneController:
    def __init__(self, debug=False):
        # state of lane controller
        self.mode = 'driving'
        self.turn_counter = 0
        self.stopping_counter = 0 # counter for stopping at red traffic lights
        self.on_crossroads = False  # flag to indicate if we are on a crossroads
        self.crossroad_cool_down = 0  # counter to avoid detecting the same crossroads multiple times
        self.check_traffic_light_counter = 0 # counter to check the traffic light status (if crossroads detected before traffic light detection)
        self.last_traffic_light_status = '' # remember the last traffic light status

        # memory for road center to avoid sudden changes in the steering angle (lazy controller)
        self.road_center_memory = [] # remember the last road center to avoid sudden changes in the steering angle
        self.avg_remmembered_road_center = 20 # average of the last road center to avoid sudden changes in the steering angle (hyperparameter , the higher the value, the smoother the steering angle)

        # some (static) parameters of the controller
        self.steering_row = 94
        self.crossroad_row = 90

        self.max_steering_angle = 0.3
        self.max_speed = 24
        self.min_speed = 18

        # flag for debugging
        self.debug = debug
        self.anti_spam=''   # to avoid spamming the console with traffic light status


    def _display_image(self, image, name, scale=2):
        """
        if debug mode is active, show the image
        """
        if not self.debug:
            return

        image = image.copy()
        image[self.steering_row, :] = 1.0 - image[self.steering_row, :]  # mark steering row
        image[self.crossroad_row, :] = 1.0 - image[self.crossroad_row, :]  # mark crossroad row

        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, image.shape[1]*scale, image.shape[0]*scale)
        cv2.imshow(name, image)
        cv2.waitKey(1)

    def _get_road(self, image):
        """
        This method detects the road. It takes the image, converts it into HSV (Hue, Saturation, Value) and
        filters pixels that are likely to be part of the road.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 0, 0], np.uint8)
        upper = np.array([200, 54, 100], np.uint8)
        mask_tarmac = cv2.inRange(hsv, lower, upper).astype(bool)

        lower = np.array([25, 100, 100], np.uint8)
        upper = np.array([50, 255, 255], np.uint8)
        mask_yellow_road_markings = cv2.inRange(hsv, lower, upper).astype(bool)

        binary_image = np.zeros(image.shape)
        binary_image[mask_tarmac] = 1.0
        binary_image[mask_yellow_road_markings] = 1.0
        binary_image = binary_image[:, :, 0]  # reduce to 1 channel (gray image)
        reduce_noise = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, (3, 3))
        self._display_image(reduce_noise, 'road image')
        return reduce_noise

    def _get_road_center(self, road_image):
        """
        This method uses the binarised road image to determine the center of the road.
        It is returned in the range [-1, 1], where -1 means the center is on the left and +1 on the right.
        If the road is not visible, it returns None.
        """
        road_img_line = road_image[self.steering_row, :]
        if road_img_line.sum() == 0:
            # no road visible
            return None

        idx = road_img_line.nonzero()[0]
        cx = np.mean(idx / len(road_img_line))  # center in [0, 1]
        center_normalised = (cx - 0.5) * 2  # normalised to [-1, 1]
        return center_normalised

    def _at_crossroads(self, road_image):
        """
        This method uses the binarised road image to determine the presence of crossroads.
        It returns True if crossroads are detected, False otherwise.
        """
        steering_sum = road_image[self.steering_row, :].sum()
        crossroad_sum = road_image[self.crossroad_row, :].sum()
        # simple detector: if there is a lot more road visible further away, we assume it's a crossroad
        diff = crossroad_sum - steering_sum
        
        if diff > 80 or (diff > 20 and crossroad_sum > 250): # just hyperparameters to tune
            return True
        return False

    def _at_traffic_light(self, image, depth_channel):
        """
        This method uses the image to detect traffic lights.
        It returns traffic lights' color if traffic lights are detected, False otherwise.
        """
        road_size=67
        # Analyze only the first "road_size" rows of the depth channel
        depth_region = depth_channel[:road_size, :]

        # Apply a threshold to isolate objects within a specific depth range (hyperparameter, in this case only objects closer than 30)
        _, thresholded = cv2.threshold(depth_region, 0, 30, cv2.THRESH_BINARY)

        # Perform morphological operations to disconnect objects
        kernel = np.ones((3,3), np.uint8)
        opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

        # Find contours in the processed image
        contours, _ = cv2.findContours(opened.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through all detected contours and draw bounding boxes
        for contour in contours:
            traffic_light_colors={'red':0,'yellow':0,'green':0}
            total_black=0
            # Get the bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)
            
            #Extract the region of interest (ROI) within the bounding box
            roi = image[y:y + h, x:x + w]
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


            # Count the number of red, yellow, green and Black pixels in the ROI
            traffic_light_colors['red'] += np.sum(red_mask == 255)
            traffic_light_colors['yellow'] += np.sum(yellow_mask == 255)
            traffic_light_colors['green'] += np.sum(green_mask == 255)
            total_black += np.sum(black_mask == 255)

            distance = np.average(depth_channel[y:y + h, x:x + w])  # average distance in the bounding box (depth channel)

            if distance<=25: # if the object is too far away, we ignore it (hyperparameter)
                
                # if we have enough pixels of a color, we assume it's a traffic light (hyperparameters)
                if (traffic_light_colors['red']>10 and traffic_light_colors['red']<30
                    or traffic_light_colors['yellow']>4 and traffic_light_colors['yellow']<10 
                    or traffic_light_colors['green']>5 and traffic_light_colors['green']<30) and total_black>5:

                    # we have a traffic light, return the color with the most pixels 
                    traffic_light_status = max(traffic_light_colors, key=traffic_light_colors.get) 

                    return traffic_light_status

        return False

    def _get_steering_angle(self, road_center):
        """
        this method calculates a steering angle based on the center of the road.
        """
        if road_center is None:
            return self.steering_angle
            
        # try and keep road center on the left
        preferred_road_center_pos = -0.15
        angle = road_center - preferred_road_center_pos
        
        angle = np.clip(angle, -self.max_steering_angle, self.max_steering_angle)
        return angle

    def _get_speed(self, steering_angle):
        """
        this method identifies a speed based on the steering angle.
        if the steering angle is high, we reduce the speed to avoid tipping over.
        """
        # speed factor is 1 if going straight and 0 if turning maximally
        speed_factor = (self.max_steering_angle - abs(steering_angle)) / self.max_steering_angle
        # linearly interpolate between min and max speed
        speed = (self.max_speed - self.min_speed) * speed_factor + self.min_speed
        return speed

    def get_angle_and_speed(self, image,depth_image):
        """
        this is the main function of the lane controller.
        it takes an image and returns a steering angle and a speed.
        """

        steering_angle = 0
        road_image = self._get_road(image)
        road_center = self._get_road_center(road_image)

        if road_center is not None:
            
            if not (abs(road_center)<0.01):   # if the road center is too close to the crossroad algorithm will be confused, so we ignore it (hyperparameter)
                if len(self.road_center_memory)<self.avg_remmembered_road_center: # if the memory is not full, we add the road center to the memory
                    self.road_center_memory=[road_center]*self.avg_remmembered_road_center
                else:
                    self.road_center_memory.pop(0)  # if the memory is full, we remove the oldest road center
                    self.road_center_memory.append(road_center) # add the new road center to the memory
                    road_center = np.mean(self.road_center_memory) # calculate the average of the last road center to avoid sudden changes in the steering angle
            else:
                road_center = np.mean(self.road_center_memory) # if the road center is too close to the crossroad, we use the average of the last road center to avoid sudden changes in the steering angle
        
        # if we are on a crossroads, we don't need to check it again (to avoid problem with detecting same crossroads multiple times)
        if self.crossroad_cool_down>0:
            self.crossroad_cool_down-=1
        else:
            self.crossroad_cool_down=0


        if  not self.on_crossroads and self.crossroad_cool_down==0: # if we are on a crossroads, we don't need to check it again (to avoid problem with the traffic light detection)
            self.on_crossroads = self._at_crossroads(road_image)

        if self.mode == 'turning_left':
            # MODE 'turning_left': make a (blind) turn at a crossroads based on time passed
            self.turn_counter += 1
            if self.turn_counter < 45:
                # keep going straight first to ensure we're on the crossroads
                steering_angle = 0
            elif self.turn_counter < 120:
                # actually make the turn
                steering_angle = -self.max_steering_angle
            else:
                # we finished the turn, go back to normal driving
                self.mode = 'driving'
                self.turn_counter = 0
                if self.debug:
                    print('finished turn, back to normal driving!')
                    
        elif self.mode == 'turning_right':
            # MODE 'turning_right': make a (blind) turn at a crossroads based on time passed
            self.turn_counter += 1
            if self.turn_counter < 70:
                
                # keep going straight first to ensure we're on the crossroads
                steering_angle = 0
            elif self.turn_counter < 150:
                # actually make the turn
                steering_angle = self.max_steering_angle
            else:
                # we finished the turn, go back to normal driving
                self.mode = 'driving'
                self.turn_counter = 0
                if self.debug:
                    print('finished turn, back to normal driving!')

        elif self.mode == 'driving' or self.mode == 'stopping':
            # MODE 'driving': normal driving, reactive controller based on road image
            # MODE 'stopping': stopping at red traffic lights
            

            if road_center is None:
                # oh-oh, we are probably somewhere off-road... hopefully this does not happen
                if self.debug:
                    print('oh-oh, we are probably somewhere off-road... ')
                steering_angle = 0  # maybe we get back to the road?
            else:
                # normal steering
                steering_angle = self._get_steering_angle(road_center)

            # check if we need to stop at a traffic light
            traffic_light_status = self._at_traffic_light(image,depth_image)
            
            if traffic_light_status:
                self.check_traffic_light_counter=50 # if we detect a traffic light, then crossroads can pass (any number >30)

            if traffic_light_status == 'red':
                self.mode = 'stopping'
                self.stopping_counter+=1
                self.last_traffic_light_status = 'red'
            elif traffic_light_status == 'yellow' or traffic_light_status == 'green':
                self.stopping_counter=0 # increase the stopping counter to avoid stopping at yellow or green traffic lights
                self.last_traffic_light_status = traffic_light_status
                self.mode = 'driving'
                
            else:
                # if we are stopping, we need to check if we can start driving again
                if self.stopping_counter < 150 and self.last_traffic_light_status == 'red':
                    
                    self.stopping_counter += 1
                else:
                    
                    self.stopping_counter = 0
                    self.mode = 'driving'
                    self.last_traffic_light_status = ''

            # print traffic light status if it changes, but not too often (anti-spam)
            if self.debug and traffic_light_status and self.anti_spam!=traffic_light_status:
                print(f"{traffic_light_status} Traffic Light Detected !")
                self.anti_spam=traffic_light_status

            
            # check if we need to make a turn
            if self.on_crossroads and self.mode=='driving' and self.check_traffic_light_counter>30:
                self.check_traffic_light_counter=0
                self.on_crossroads = False  # only once per crossroads
                self.crossroad_cool_down = 50 # cool down for a while to avoid detecting the same crossroads multiple times
                # at crossroads -> change to MODE turning
                self.mode = 'turning_right'

                if self.debug:
                    self.anti_spam=''
                    print(f'crossroads detected! {self.mode}')
            elif self.on_crossroads and self.mode=='driving':
                self.check_traffic_light_counter+=1
        else:
            # unknown mode, if this happens we made a programming mistake
            raise ValueError(f'unknown mode: {self.mode}')


        # if we are stopping, we keep the steering angle constant and reduce the speed to a minimum value
        if self.mode=='stopping':
            steering_angle=self._get_steering_angle(road_center)

            # if the steering angle is too high, we keep the steering angle constant and reduce the speed to a minimum value
            if abs(self._get_steering_angle(road_center)) >0.03 and self.stopping_counter<20:

                return steering_angle,self.min_speed
            else:
                return 0,0
        else:
            # normal driving mode -> calculate speed based on steering angle
            speed = self._get_speed(steering_angle)

        return steering_angle, speed
