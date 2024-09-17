#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

PUBLISH_TOPIC_NAME: str = '/cmd_vel'

class Brain:
    def __init__(self):
        self._bridge = CvBridge()
        self._rate = rospy.Rate(2)
        
        self._move = Twist()
        self._pub = rospy.Publisher(PUBLISH_TOPIC_NAME, Twist, queue_size=1)
        
        self._move.linear.x = 1.5
        self._move.angular.z = -0.5

    def move(self):
        self._pub.publish(self._move)
        self._rate.sleep()

# cv_image = bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough')

rospy.init_node('topic_publisher')
brain = Brain()

while not rospy.is_shutdown():
   brain.move()