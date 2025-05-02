from matplotlib import pyplot as plt
import numpy as np
import cv2


def display_image(image, name, scale=2, wait=False):
    """ 
    function to display an image 
    :param image: ndarray, the image to display
    :param name: string, a name for the window
    :param scale: int, optional, scaling factor for the image
    :param wait: bool, optional, if True, will wait for click/button to close window
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, image.shape[1]*scale, image.shape[0]*scale)
    cv2.imshow(name, image)
    cv2.waitKey(0 if wait else 1)
    
# this function is used to display the image in a window that show detected cubes
def normalize_depth(depth_image):
    """
    function to normalize the depth image and find contours of cubes
    :param depth_image: ndarray, the depth image to process
    :return: ndarray, the output image with contours drawn
    """
    # clean the depth image by replacing invalid values with 0
    depth_clean = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
    
    
    # normalize the depth image to the range [0, 255] by using min-max normalization
    depth_normalized = cv2.normalize(
        depth_clean, None, 0, 255, 
        cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    
    # invert the depth image to make closer points brighter (cubes==white, ground==black)
    grayscale = 255 - depth_normalized
    
    # apply dynamic thresholding to create a binary image
    _, output = cv2.threshold(
        grayscale, 0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # apply morphological operations to separate close squares
    kernel = np.ones((1, 1), np.uint8)
    output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)
    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)

    # find contours in the binary image
    contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # create a copy of the original image to draw contours on
    output_image = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    
    for contour in contours:
        # calculate the minimum area rectangle for each contour
        rect = cv2.minAreaRect(contour)
        (x, y), (w, h), angle = rect

        # calculate the aspect ratio of the rectangle, which is the ratio of the longer side to the shorter side
        # this is used to filter out non-square shapes
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        # if the aspect ratio is between 0.95 and 1.2, it is considered a square
        if 0.95 <= aspect_ratio <= 1.2:
            # draw the minimum area rectangle on the output image
            epsilon = 0.03 * cv2.arcLength(contour, True)
            # approximate the contour to a polygon with fewer vertices
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # draw the approximated polygon on the output image
            cv2.drawContours(output_image, [approx], -1, (0, 255, 0), 2)
    
    return output_image

# this function is used to find the cubes in the depth image
def get_cubes(depth_image):
    """
    function to find cubes in the depth image
    :param depth_image: ndarray, the depth image to process
    :return: tuple, the coordinates of the cube center, depth value, and angle"""
    # clean the depth image by replacing invalid values with 0
    depth_clean = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
    
    # normalize the depth image to the range [0, 255] by using min-max normalization
    depth_normalized = cv2.normalize(
        depth_clean, None, 0, 255, 
        cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    
    # invert the depth image to make closer points brighter (cubes==white, ground==black)
    grayscale = 255 - depth_normalized
    
    # apply dynamic thresholding to create a binary image
    _, output = cv2.threshold(
        grayscale, 0, 5, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # apply morphological operations to separate close squares
    kernel = np.ones((1, 1), np.uint8)
    output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)
    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
    # find contours in the binary image
    contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for contour in contours:
        # calculate the minimum area rectangle for each contour
        rect = cv2.minAreaRect(contour)
        (x, y), (w, h), angle = rect

        # calculate the aspect ratio of the rectangle, which is the ratio of the longer side to the shorter side
        # this is used to filter out non-square shapes
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        
        # if the aspect ratio is between 0.95 and 1.2, it is considered a square
        if 0.95 <= aspect_ratio <= 1.2:
            # find height from the center of the square
            center_x = int(x)
            center_y = int(y)
            depth_value = depth_image[center_y, center_x]
            
            # save the square's center coordinates, depth value, and angle to the list
            squares.append((center_x, center_y, depth_value, angle))
    
    # if no squares are found, return None
    if not squares:
        return None
    
    # sort the squares by depth value in descending order
    squares.sort(key=lambda x: x[2], reverse=True)
    # return the first square from the list
    return squares[0]

