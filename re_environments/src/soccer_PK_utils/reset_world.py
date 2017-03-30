# -*- coding: utf-8 -*-

import numpy as np
import rospy
import std_srvs.srv
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnModel, SetModelState, GetModelState
from gazebo_msgs.msg import ModelState

# reset gazebo world(it resets models)
def reset_world(robot_x=3.0,robot_y=0,robot_angle=0,ball_x=3.25,ball_y=0):
    rospy.wait_for_service('gazebo/reset_world')
    # ServiceProxy and call means `rosservice call /gazebo/reset_world`
    srv = rospy.ServiceProxy('gazebo/reset_world', std_srvs.srv.Empty)
    srv.call()

    # set the robot
    model_pose = Pose()
    model_pose.position.x = robot_x
    # model_pose.position.x = 3.0
    model_pose.position.y = robot_y
    model_pose.orientation.z = np.sin(robot_angle/2.0) 
    model_pose.orientation.w = np.cos(robot_angle/2.0) 
    modelstate = ModelState()
    modelstate.model_name = 'mobile_base_player_1'
    modelstate.reference_frame = 'world'
    modelstate.pose = model_pose
    set_model_srv = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
    set_model_srv.call(modelstate)


    # set the ball
    model_pose = Pose()
    model_pose.position.x = ball_x
    # model_pose.position.x = 3.25
    model_pose.position.y = ball_y
    modelstate = ModelState()
    modelstate.model_name = 'soccer_ball'
    modelstate.reference_frame = 'world'
    modelstate.pose = model_pose
    set_model_srv = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
    set_model_srv.call(modelstate)
    # rospy.loginfo("reset world")
