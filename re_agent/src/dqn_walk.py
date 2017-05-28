#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import Int16
from std_msgs.msg import Float32
from sensor_msgs.msg import Image

import numpy as np
import sys
import cv2
from cv_bridge import CvBridge, CvBridgeError

from chainer import links as L
from chainer import optimizers
from chainerrl import links
from chainerrl import replay_buffer
from chainerrl.action_value import DiscreteActionValue
from dqn import DQN
from network import NatureDQNHead

__version__ =  '0.2'
__date__ = '2017/05/28'

rospy.init_node("dqn_walk")
dir_name = "sample_model"


class agent_DQN():
    def __init__(self, test=False):
        #self.n_stat = SoccerEnv.n_stat
        self.n_act = 5#SoccerEnv.n_act
        self.n_frame = 4
        """ Set up DQN """
        # q function
        self.q_func = links.Sequence(
            NatureDQNHead(self.n_frame),
            L.Linear(512, self.n_act),
            DiscreteActionValue
        )
        # optimizer
        self.opt = optimizers.RMSpropGraves(
            lr=2.5e-4, alpha=0.95, momentum=0.95, eps=1e-2)
        self.opt.setup(self.q_func)

        # Replay Buffer
        self.rbuf = replay_buffer.ReplayBuffer(10 ** 5)

        # DQN agent
        self.agent = DQN(self.q_func, self.opt, self.rbuf, gpu=None, gamma=0.99,
                    explorer=None, replay_start_size=10 ** 4,
                    target_update_frequency=10 ** 4,
                    update_frequency=self.n_frame, frame=self.n_frame)

        """ Set up Publisher """
        self.pub = rospy.Publisher("dqn_act", Int16, queue_size=10)

        """ Set up Subscriber """
        self.camera_name = "/agent1/camera"
        self.topic_name_cam = self.camera_name + "/rgb/image_raw"
        self.bridge = CvBridge()
        self.rate = rospy.Rate(3)
        
        self.agent.load(dir_name)
        self.test = test
        self.N_episode = 0

    # call back from Kinect
    def call_back_cam(self, data,id):
        # to deside action
        if id == 0: 
            state = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # chainer code
            action = self.agent.act(state)
            
            self.pub.publish(action)
            time = rospy.get_time()
            rospy.set_param("action_value", [int(action), time])
        # to train chainer
        else:
            # get data for add_experiment
            reward, done = rospy.get_param("/reward_value") 
            action_value, time = rospy.get_param("/agent1/action_value")      
            time_now = rospy.get_time()
            next_state = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # chainer code
            self.agent.add_experience(action_value, reward, next_state, done)
            if self.test is False:
                self.agent.train()
            
            ### debug
            # rospy.logwarn(done)
            ###
            
            # end one episode
            if done is True and self.prev_done is not True:
                self.prev_done = done
                self.N_episode += 1
                rospy.logwarn(self.N_episode)
                self.agent.stop_episode()
            self.prev_done = done

            # save the chainer variables
            if self.N_episode % 100 == 0:
                # save usually in $HOME/.ros/
                self.agent.save(dir_name)


arg_test = sys.argv[1]
if arg_test in "False":
    test = False
else:
    test = True

ad = agent_DQN(test=test)
sub0 = rospy.Subscriber(ad.topic_name_cam, Image, ad.call_back_cam, callback_args=0, queue_size=1)
sub1 = rospy.Subscriber(ad.topic_name_cam, Image, ad.call_back_cam, callback_args=1, queue_size=1)

rospy.spin()
