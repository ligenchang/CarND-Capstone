#!/usr/bin/env python
import rospy
import numpy as np

from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from light_classification.tl_cv_classifier import TLCVClassifier
from scipy.spatial import KDTree

import tf
import cv2
import yaml
import PyKDL


STATE_COUNT_THRESHOLD = 3


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.has_image = False
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        # self.light_classifier = TLClassifier()
        self.light_classifier = TLCVClassifier()

        self.listener = tf.TransformListener()
        self.focal_x = self.config['camera_info'].get('focal_length_x', 2300)
        self.focal_y = self.config['camera_info'].get('focal_length_y', 2300)
        self.image_width = self.config['camera_info']['image_width']
        self.image_height = self.config['camera_info']['image_height']

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.tl_closest_wp = None       # Will contain a dict: light id -> closest waypoint idx

        self.loop()

    def loop(self):
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            if self.tl_closest_wp is not None and self.lights and self.pose:
                self.process_image()

            rate.sleep()

    def process_image(self):
        light_wp, state = self.process_traffic_lights()
        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

            # Compute for each stop line, the closest way point
            stop_line_positions = self.config['stop_line_positions']
            self.tl_closest_wp = {i: self.get_closest_waypoint(line[0], line[1])
                                  for (i, line) in enumerate(stop_line_positions)}

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        ###############
        # Check where closest is relative to x and y
        ###############
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closes_coord
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        # Closest point was behind, so take next one
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx

    def get_image_coordinates(self, map_coords):
        """
        Get image coordinates from map coordinates
        """
        # ROS is nice and can provide the transformation
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link", "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link", "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find transformation to image coordinates")
            return -1, -1

        # Apply transformation
        piw = PyKDL.Vector(map_coords.x, map_coords.y, map_coords.z)
        rot = PyKDL.Rotation.Quaternion(*rot)
        trans = PyKDL.Vector(*trans)
        p_car = rot * piw + trans

        x = - p_car[1] / p_car[0] * self.focal_x + self.image_width / 2
        y = - p_car[2] / p_car[0] * self.focal_y + self.image_height / 2 + 340

        return int(x), int(y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.has_image:
            self.last_state = None
            return TrafficLight.UNKNOWN

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get the image coordinates of the light
        light_x, light_y = self.get_image_coordinates(light.pose.pose.position)

        # If outside the frame, then it is unknown
        if light_x < 0 or light_x > self.image_width or light_y < 0 or light_y > self.image_height:
            return TrafficLight.UNKNOWN

        # Crop the image around it
        width, height = 180, 180
        x_min = max(light_x - width // 2, 0)
        x_max = min(light_x + width // 2, self.image_width)
        y_min = max(light_y - height // 2, 0)
        y_max = min(light_y + height // 2, self.image_height)

        cropped_light = cv_image[y_min: y_max, x_min: x_max]
        #cv2.imwrite("/home/student/output/cropped_" + str(rospy.Time.now()) + ".png", cropped_light)

        # Classify the box
        #return light.state
        return self.light_classifier.get_classification(cropped_light)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_index = None

        # Find the nearest light in front of the car
        if self.pose:
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            # Just ignore all traffic lights further than max waypoints (since anyway they won't impact trajectory)
            min_dist = 50   # LOOKAHEAD_WPS
            for i, light in enumerate(self.lights):
                # Get waypoint for the light
                temp_wp_idx = self.tl_closest_wp[i]

                d = temp_wp_idx - car_wp_idx

                # Traffic light is in front of the car and closer that other visited traffic lights
                if d >= 0 and d <= min_dist:
                    min_dist = d
                    closest_light = light
                    line_wp_index = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_index, state

        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
