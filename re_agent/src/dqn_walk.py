#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import Int16
from std_msgs.msg import Float32
from sensor_msgs.msg import Image

import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

from chainer import links as L
from chainer import optimizers
from chainerrl import links
from chainerrl import replay_buffer
from chainerrl.action_value import DiscreteActionValue
from dqn import DQN
from network import NatureDQNHead


__date__ = '0.1'

rospy.init_node("dqn_walk")
dir_name = "Sei_Ueno_test.model"

""" Set up DQN """
#n_stat = SoccerEnv.n_stat
n_act = 5#SoccerEnv.n_act

n_frame = 4

# q function
q_func = links.Sequence(
    NatureDQNHead(n_frame),
    L.Linear(512, n_act),
    DiscreteActionValue
)

# optimizer
opt = optimizers.RMSpropGraves(
    lr=2.5e-4, alpha=0.95, momentum=0.95, eps=1e-2)
opt.setup(q_func)

# Replay Buffer
rbuf = replay_buffer.ReplayBuffer(10 ** 5)

# DQN agent
agent = DQN(q_func, opt, rbuf, gpu=None, gamma=0.99,
            explorer=None, replay_start_size=10 ** 4,
            target_update_frequency=10 ** 4,
            update_frequency=n_frame, frame=n_frame)


""" Set up Publisher """
pub = rospy.Publisher("dqn_act", Int16, queue_size=10)

""" Set up Subscriber """
camera_name = "/agent1/camera"
topic_name_cam = camera_name + "/rgb/image_raw"
bridge = CvBridge()
rate = rospy.Rate(3)

'''
ToDo

- decide done variable
- decide how to call by variable
'''

def call_back_cam(data,id):
    if id == 0:
        state = bridge.imgmsg_to_cv2(data, "bgr8")
        action = agent.act(state)
        pub.publish(action)
        time = rospy.get_time()
        rospy.set_param("action_value", [int(action), time])
    else:
        reward, done = rospy.get_param("/reward_value")
        rospy.logwarn(done)
        action_value, time = rospy.get_param("/agent1/action_value")      
        time_now = rospy.get_time()
        next_state = bridge.imgmsg_to_cv2(data, "bgr8")
        agent.add_experience(action_value, reward, next_state, done)
        agent.train()
        agent.stop_episode()
        # save usually in $HOME/.ros/
        agent.save(dir_name)
        agent.load(dir_name)
    # add experience to replay buffer
    # agent.add_experience(action, reward, next_state, done)

    # train agent
    # agent.train()

    # refresh at the end of episode
    # agent.stop_episode()

    # save model
    # agent.save(dir_name)

    # load model
    # agent.load(dir_name)

sub0 = rospy.Subscriber(topic_name_cam, Image, call_back_cam, callback_args=0, queue_size=1)
sub1 = rospy.Subscriber(topic_name_cam, Image, call_back_cam, callback_args=1, queue_size=3)

rospy.spin()
