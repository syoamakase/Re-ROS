#!/usr/bin/env python
import rospy
from std_msgs.msg import Int16
from sensor_msgs.msg import Image

import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

if __name__ == "__main__":

    rospy.init_node("dqn_walk")

    """ Set up DQN """
    #n_stat = SoccerEnv.n_stat
    #n_act = SoccerEnv.n_act
    #model = DQN(n_stat, n_act, L1_rate=None, on_gpu=True)

    """ Set up Publisher """
    pub = rospy.Publisher("dqn_act", Int16, queue_size=10)

    """ Set up Subscriber """
    camera_name = "camera"
    topic_name_cam = camera_name + "/rgb/image_raw"
    bridge = CvBridge()
    def call_back(data):
        d = bridge.imgmsg_to_cv2(data, "bgr8")
        act = 1 #model(d)
        pub.publish(act)
    sub = rospy.Subscriber(topic_name_cam, Image, call_back)

rospy.spin()
