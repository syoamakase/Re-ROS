# Re:ROS

Reinforcement Learning using ROS.

## Usage

1.Copy "re\_ros", "re\_envrionments", "re\_agent", and "re\_rule" in your catkin workspace

2.make(catkin_make) the packages

3.Start an example
	`roslaunch re_ros turtlebot_soccerPK.launch`

## Packages

### re_ros

To choose the environment, agent, rule.

### re_environments

To config envrionment using Gazebo.

### re_agent

To config agent and agent's actions.
In this example, turtlebot_gazebo package is needed.

### re_rule

In this example, re\_rule is used to decide the action.