# Project Name
Self Driving Car

## Description
The goal is to train an agent to drive by giving it some expert trajectories.

### Install Pygame

Install Pygame's dependencies with:

`sudo apt install mercurial libfreetype6-dev libsdl-dev libsdl-image1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libportmidi-dev libavformat-dev libsdl-mixer1.2-dev libswscale-dev libjpeg-dev`

Then install Pygame itself:

`pip3 install hg+http://bitbucket.org/pygame/pygame`

### Install Pymunk

This is the physics engine used by the simulation. It just went through a pretty significant rewrite (v5) so you need to grab the older v4 version. v4 is written for Python 2 so there are a couple extra steps.

Go back to your home or downloads and get Pymunk 4:

`wget https://github.com/viblo/pymunk/archive/pymunk-4.0.0.tar.gz`

Unpack it:

`tar zxvf pymunk-4.0.0.tar.gz`

Update from Python 2 to 3:

`cd pymunk-pymukn-4.0.0/pymunk`

`2to3 -w *.py`

Install it:

`cd ..`
`python3 setup.py install`

## Visualize Result:
To visualize the current result: `python3 show_result.py 'red'` (options: 'red'/ 'yellow'/ 'brown'/ 'bumping')

## Training:
First, you need to train a model. This will save weights to the `saved-models` folder. *You may need to create this folder before running*. You can train the model by running:

`python3 maxEntIRL.py`

This takes 6 hours in my machine(I am not using GPU). Most of the time is spent on training the Deep Q Network for RL.

## GitHub: https://github.com/mturja-vf-ic-bd/self-driving-car_Robotics-Project/tree/master/ReadMe