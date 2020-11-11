## Using demo.py

The demo.py is split into two parts as below:
* demo_run.py - Does the forward pass and saves the computed results as pickle file and saves them at a defined location in the docker container
* demo_vis.py - Takes in the pickle file and then visualizes it. Can run it on local as well after copying the pickle file to local from docker container.
# Kitti Odometry tracking training label format
label format: 
FrameID, ObjectID, Class, truncated(0-1 Float), occluded(0, 1, 2, 3 int), alpha (observation angle [-pi, pi]), bbox(left, top, right, bottom), 3d-bbox(h, w, l), 3d-bbox-camera-coordinate(x, y, z, rot-y [-pi, pi])
Total 17 columns
eg. 0 0 Car 0 0 -1.113049 254.705453 175.391996 306.541915 203.338418 1.716012 1.628068 3.541835 -21.190459 1.900249 46.495970 -1.538772
