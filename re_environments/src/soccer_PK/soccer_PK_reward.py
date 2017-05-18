#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
import numpy as np
import soccer_PK.utils

# goal 4.5

rospy.init_node("reward")
pub = rospy.Publisher("reward", Float32, queue_size=10)
rate = rospy.Rate(3)
rospy.wait_for_service('/gazebo/get_model_state')
soccer_PK.utils.reset_world()
ball_prev = 3.25
while not rospy.is_shutdown():
    tic = rospy.get_time()
    toc = tic
    prev_reward = None
    while toc - tic < 10:
        done = False
        # pub.publish(reward)
        ball_locationx ,ball_locationy = soccer_PK.utils.get_ball_location()
        if ball_locationx > 4.5:
        	rospy.loginfo("GOAL!!!")
        	reward = 10
        	done = True
        	rospy.set_param("reward_value",[reward, done])
        	tic = rospy.get_time()
        	soccer_PK.utils.reset_world()
        	rospy.sleep(2)
        reward = (ball_prev - ball_locationx) / ball_prev
        if prev_reward != reward:
            rospy.set_param("reward_value",[reward, done])
        prev_reward = reward
        toc = rospy.get_time()
    reward = -10
    done = True
    prev_reward = reward
    # pub.publish(reward)
    rospy.set_param("reward_value",[reward, done])
    soccer_PK.utils.reset_world()
    rospy.sleep(1)
rate.sleep()
