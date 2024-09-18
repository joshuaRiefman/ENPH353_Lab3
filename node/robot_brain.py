#! /usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
from typing import Union, Tuple
from sensor_msgs.msg import Image


class NoVisibleRoadException(Exception):
    pass


PUBLISH_TOPIC_NAME: str = '/cmd_vel'
IMAGE_PUB_NAME: str = '/robot/vision'
UNKNOWN_POSITION: int = -1

ROBOT_SPEED_LINEAR = 0.66
ROBOT_SPEED_ANGULAR = 5


# pts1 = np.array([[50, 50], [390, 50], [50, 390], [390, 390]], dtype=np.float32)
# pts2 = np.array([[0, 0], [800, 0], [0, 800], [800, 800]], dtype=np.float32)
# M = cv.getPerspectiveTransform(pts1, pts2)


def get_new_position(new_position, prev_position, dt) -> float:
    velocity = abs(new_position - prev_position)
    if velocity > 10:
        velocity = 2 * np.log10(velocity - 9) + 10

    max_change_in_position = velocity * dt
    center_x = np.clip(new_position, prev_position - max_change_in_position, prev_position + max_change_in_position)

    return float(center_x)


def detect_line_center(frame: np.ndarray) -> Tuple[float, int]:
    """
    
    """
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    height: int = gray_img.shape[1]

    slice_row: int = 0

    # Iterate over the rows of the image, starting at the bottom
    # whilst we don't have a line location, and not going above
    # halfway up the screen where the ground goes away due to perspective. 
    while slice_row < height / 2:
        bottom: np.ndarray = gray_img[-(1 + slice_row), :]
        slice_mean: float = np.mean(bottom)

        # Must be 25% darker than the mean to count
        middle: np.ndarray = np.where(bottom < (slice_mean - 0.25 * slice_mean))

        potential_line_location: float = np.mean(middle)

        # If it's NaN, we didn't find a center location
        # so we need to keep looking
        if np.isnan(potential_line_location):
            slice_row += 1
            continue
        
        # Otherwise, we did, so we can safely return it
        else:
            return potential_line_location, slice_row

    raise NoVisibleRoadException


class Brain:
    def __init__(self):
        self._bridge = CvBridge()
        self._rate = rospy.Rate(60)
        
        self._move = Twist()

        self._pub = rospy.Publisher(PUBLISH_TOPIC_NAME, Twist, queue_size=1)
        self._image_pub = rospy.Publisher(IMAGE_PUB_NAME, Image, queue_size=1)
        self._image_sub = rospy.Subscriber("/camera1/image_raw", Image, self._update, queue_size=1)


    def _update(self, data: np.ndarray):
        try:
            cv_image: np.ndarray = self._bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            # cv_image = cv.warpPerspective(raw_cv_image, M, [800, 800])
        except Exception as e:
            rospy.loginfo(e)
            return
        
        found_line = False
        height: int = cv_image.shape[1]

        try:
            center, row = detect_line_center(cv_image)

            found_line = True

        except NoVisibleRoadException:
            # This will cause the robot to just rotate
            center = height / 2
            row = 0 # Not functional, just for cosmetics

        # Compute normalized displacement of line position from center
        normalized_center = (float(center) / float(height)) - 0.5
        
        # Now, scale the value to counteract edge widening bias
        adjusted_normalized_center = np.sign(normalized_center) * np.sqrt(np.abs(normalized_center))
        
        try:   
            vision = self._bridge.cv2_to_imgmsg(cv.circle(cv_image, (int(center), int(height) - int(row) - 40), 20, (0, 0, 255), thickness=-1), encoding="bgr8")
            self._image_pub.publish(vision)
        except Exception as e:
            rospy.loginfo(e)

        self._move.angular.z = adjusted_normalized_center * -ROBOT_SPEED_ANGULAR

        # Only move forwards when we are seeing the line!
        if found_line:
            # Reduce speed when not centered nicely
            reduction_factor = max(0, 1 - (3 * np.abs(adjusted_normalized_center)))
            self._move.linear.x = ROBOT_SPEED_LINEAR * reduction_factor


    def move(self):
        self._pub.publish(self._move)
        self._rate.sleep()


rospy.init_node('topic_publisher')
brain = Brain()

while not rospy.is_shutdown():
    brain.move()
