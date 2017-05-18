#!/usr/bin/env python

import rospy
from std_msgs.msg import Int16
import numpy as np

__version__ = '0.1'
__date__ = '2017/05/11'


rospy.init_node("random_walk")
pub = rospy.Publisher("rand_act", Int16, queue_size=10)
# 3hz
rate = rospy.Rate(3)
while not rospy.is_shutdown():
    act = np.random.randint(5)
    pub.publish(act)
    rate.sleep()
