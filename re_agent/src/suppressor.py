#!/usr/bin/env python

import rospy
from std_msgs.msg import Int16
import numpy as np

rospy.init_node("suppressor")
pub = rospy.Publisher("suppress_act", Int16, queue_size=10)
rate = rospy.Rate(1)
while not rospy.is_shutdown():
    if np.random.rand() < 0.5:
        pub.publish(0)
rate.sleep()
