# re_environments


## launch

Here includes launch file.
In "soccer.launch", launch Gazebo and spawn a soccer ball.


## src

In "soccer_PK",

utils.py : the basic functions that RL uses. ex) reset the objects

soccer\_PK\_reward.py : calcuate reward parameter

## worlds

Here includes world files.
"soccer_field.xml" is the world of Gazebo.
In "models", there are sdf files which define object.

If you want to use your own world, you put your world and change a launch file in re\_ros