apt-get update && apt install mercurial libfreetype6-dev libsdl-dev libsdl-image1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libportmidi-dev libavformat-dev libsdl-mixer1.2-dev libswscale-dev libjpeg-dev
pip3 install hg+http://bitbucket.org/pygame/pygame
cd pymunk-pymunk-4.0.0/pymunk
2to3 -w *.py
cd ..
python3 setup.py install
