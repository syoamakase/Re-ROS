#!/usr/bin/env python
import rospy
from std_msgs.msg import Int16
from sensor_msgs.msg import Image

import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

from chainer import links as L
from chainerrl import links
from chainerrl.action_value import DiscreteActionValue
from dqn import DQN
from network import NatureDQNHead

if __name__ == "__main__":

    rospy.init_node("dqn_walk")

    """ Set up DQN """
    #n_stat = SoccerEnv.n_stat
    n_act = SoccerEnv.n_act

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
    agent = DQN(q_func, opt, rbuf, gpu=True, gamma=0.99,
                explorer=None, replay_start_size=10 ** 4,
                target_update_frequency=10 ** 4,
                update_frequency=n_frame, frame=n_frame)


    """ Set up Publisher """
    pub = rospy.Publisher("dqn_act", Int16, queue_size=10)

    """ Set up Subscriber """
    camera_name = "camera"
    topic_name_cam = camera_name + "/rgb/image_raw"
    bridge = CvBridge()
    def call_back(data):
        d = bridge.imgmsg_to_cv2(data, "bgr8")
        action = agent.act(d)

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

        pub.publish(action)
    sub = rospy.Subscriber(topic_name_cam, Image, call_back)

rospy.spin()
