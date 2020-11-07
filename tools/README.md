##Using demo.py

The demo.py is split into two parts as below:
. demo_run.py - Does the forward pass and saves the computed results as pickle file and saves them at a defined location in the docker container
. demo_vis.py - Takes in the pickle file and then visualizes it. Can run it on local as well after copying the pickle file to local from docker container.
