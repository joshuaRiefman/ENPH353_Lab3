#! /usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
from typing import Union, Tuple
from sensor_msgs.msg import Image


PUBLISH_TOPIC_NAME: str = '/cmd_vel'
UNKNOWN_POSITION: int = -1


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
    line_location = -1

    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    height: int = gray_img.shape[1]

    slice_row: int = 0

    # Iterate over the rows of the image, starting at the bottom
    # whilst we don't have a line location
    while line_location == -1 and slice_row < height:
        bottom: np.ndarray = gray_img[-(1 + slice_row), :]
        middle: np.ndarray = np.where(bottom < np.mean(bottom))  # This will fail if the bottom is all road

        line_location: float = np.mean(middle)

        # If it's NaN, we didn't find a center location
        # so we need to keep looking
        if np.isnan(line_location):
            continue
        
        # Otherwise, we did, so we can safely return it
        else:
            break

    return line_location, height


class Brain:
    def __init__(self):
        self._bridge = CvBridge()
        self._rate = rospy.Rate(2)
        
        self._move = Twist()
        self._move.linear.x = 1.0
        self._i = 0

        self._pub = rospy.Publisher(PUBLISH_TOPIC_NAME, Twist, queue_size=1)
        
        self._image_sub = rospy.Subscriber("/camera1/image_raw", Image, self._update, queue_size=1)


    def _update(self, data: np.ndarray):

        try:
            cv_image: np.ndarray = self._bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        except Exception as e:
            rospy.loginfo(e)
            return
        
        rospy.loginfo(f"Reading... {cv_image.shape}")

        center, height = detect_line_center(cv_image)

        normalized_center = float(center) / float(height)

        rospy.loginfo(f"Center: {normalized_center}")

        if normalized_center > 0.5:
            self._move.linear.x = 5.0
        else:
            self._move.linear.x = -5.0
        

    def move(self):
        rospy.loginfo("Main loop iteration")

        self._pub.publish(self._move)
        self._rate.sleep()



rospy.init_node('topic_publisher')
brain = Brain()

while not rospy.is_shutdown():
    brain.move()
