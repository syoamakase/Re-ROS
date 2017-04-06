#!/usr/bin/env python
import rospy
from std_msgs.msg import Int16

# from susanoh.environments.gazebo_action import GazeboAction
from geometry_msgs.msg import Twist
import numpy as np

topic_name = "accumulator"
rospy.init_node(topic_name)

target_name = "agent1"
topic_name_vel = target_name+"/mobile_base/commands/velocity"

# ga = GazeboAction()

class Accumulator(object):
    def __init__(self):

        self.def_accum_vars = {"rand":[0, 0, 0, 0, 0], "dqn":[0, 0, 0, 0, 0], 
                               "suppress":[0]}
        self.accumulators = {key:self.def_accum_vars[key] for key 
                             in self.def_accum_vars}
        self.lock = {"rand":False, "dqn":False, "suppress":False}
        self.evidence_scale = {"rand":0.5, "dqn":0.1, "suppress":0.1}
        
        self.act_id = 0

    def accum_renew(self, aid, key):
        if self.lock[key]: return
        self.accumulators[key][aid] += self.evidence_scale[key]
        if self.accumulators[key][aid] >= 1.0:
            self.accumulators[key] = self.def_accum_vars[key]
            self.act_id = aid
            return True
        return False

    def call_back_rand_walk(self, message):
        if self.accum_renew(message.data, "rand"):
            pass

    def call_back_dqn(self, message):
        if self.accum_renew(message.data, "dqn"):
            self.lock["rand"] = True

    def call_back_suppress(self, message):
        if self.accum_renew(0, "suppress"):
            self.lock["rand"] = False
            self.lock["dqn"] = False
            
accumulator = Accumulator()
sub_rand_walk = rospy.Subscriber("/agent1/rand_act", Int16, accumulator.call_back_rand_walk)
sub_dqn_walk = rospy.Subscriber("/agent1/dqn_act", Int16, accumulator.call_back_dqn)
sub_suppress = rospy.Subscriber("/agent1/suppress_act", Int16, accumulator.call_back_suppress)
rate = rospy.Rate(3)
while not rospy.is_shutdown():
    # ga.control_action(accumulator.act_id)
    pub = rospy.Publisher('/agent1/mobile_base/commands/velocity', Twist, queue_size=10)
    vel = Twist()
    pub.publish(vel)
rate.sleep()
