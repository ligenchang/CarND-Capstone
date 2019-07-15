from styx_msgs.msg import TrafficLight
import cv2
import numpy as np

class TLCVClassifier(object):

    def __init__(self):

        self.debug = False

        # define color ranges
        self.hsv_red_low_low = (0, 50, 0)
        self.hsv_red_low_high = (10, 255, 255)
        self.hsv_red_high_low = (160, 50, 80)
        self.hsv_red_high_high = (179, 255, 255)
        self.hsv_yellow_low = (25, 100, 100)
        self.hsv_yellow_high = (30, 255, 255)
        self.hsv_green_low = (50, 90, 50)
        self.hsv_green_high = (70, 255, 255)
        self.distance_min = 15
        self.red_canny_low = 10
        self.red_canny_high = 100
        self.yellow_canny_low = 7
        self.yellow_canny_high = 100
        self.green_canny_low = 7
        self.green_canny_high = 50
        self.radious_min = 1
        self.radious_max = 20

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # convert bgr to hsv
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # detect circles of a color
        red_circles = self.detect_traffic_light_circles(hsv, self.hsv_red_low_low, self.hsv_red_low_high, TrafficLight.RED,
            color_range_2_low=self.hsv_red_high_low, color_range_2_high=self.hsv_red_high_high)
        yellow_circles = self.detect_traffic_light_circles(hsv, self.hsv_yellow_low, self.hsv_yellow_high, TrafficLight.YELLOW)
        green_circles = self.detect_traffic_light_circles(hsv, self.hsv_green_low, self.hsv_green_high, TrafficLight.GREEN)

        print('red: {}, yellow: {}, green: {}'.format(
            0 if red_circles is None else len(red_circles), 
            0 if yellow_circles is None else len(yellow_circles),
            0 if green_circles is None else len(green_circles)))

        if self.debug == True:
            self.save_debug_images(red_circles, yellow_circles, green_circles)
        
        if red_circles is not None:
            return TrafficLight.RED
        elif green_circles is not None:
            return TrafficLight.GREEN
        elif yellow_circles is not None:
            return TrafficLight.YELLOW
        else:
            return TrafficLight.UNKNOWN

    def detect_traffic_light_circles(self, hsv_image, color_range_1_low, color_range_1_high, color,
        color_range_2_low=None, color_range_2_high=None):

        if color_range_2_low is not None and color_range_2_high is not None:
            mask_1 = cv2.inRange(hsv_image, 
                np.array(color_range_1_low, dtype = "uint8"),
                np.array(color_range_1_high, dtype = "uint8"))
            mask_2 = cv2.inRange(hsv_image, 
                np.array(color_range_2_low, dtype = "uint8"),
                np.array(color_range_2_high, dtype = "uint8"))
            mask = cv2.addWeighted(mask_1, 1.0, mask_2, 1.0, 0.0)
            mask = cv2.GaussianBlur(mask,(5,5),0)
        else:
            mask = cv2.inRange(hsv_image, color_range_1_low, color_range_1_high)

        if color == TrafficLight.RED:
            canny_low = self.red_canny_low
            canny_high = self.red_canny_high
        elif color == TrafficLight.YELLOW:
            canny_low = self.yellow_canny_low
            canny_high = self.yellow_canny_high
        elif color == TrafficLight.GREEN:
            canny_low = self.green_canny_low
            canny_high = self.green_canny_high
            
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, self.distance_min,                  
                     param1=canny_high, param2=canny_low, minRadius=self.radious_min, maxRadius=self.radious_max)

        return circles

    def save_debug_images(self, image, red_circles, yellow_circles, green_circles):
        debug_image = image.copy()

        if red_circles is not None:
            for circ in range(0,len(red_circles[0,:])):
                cv2.circle(debug_image,
                    (red_circles[0,:][circ][0], red_circles[0,:][circ][1]),
                    int(red_circles[0,:][circ][2]+5), (0,0,155), 2)
        if yellow_circles is not None:
            for circ in range(0,len(yellow_circles[0,:])):
                cv2.circle(debug_image,
                    (yellow_circles[0,:][circ][0], yellow_circles[0,:][circ][1]),
                    int(yellow_circles[0,:][circ][2]+5), (0,0,155), 2)
        if green_circles is not None:
            for circ in range(0,len(green_circles[0,:])):
                cv2.circle(debug_image,
                    (green_circles[0,:][circ][0], green_circles[0,:][circ][1]),
                    int(green_circles[0,:][circ][2]+5), (0,0,155), 2)

        cv2.imwrite("/home/student/output/tl_cv_classified_" + str(rospy.Time.now()) + ".png", debug_image)

