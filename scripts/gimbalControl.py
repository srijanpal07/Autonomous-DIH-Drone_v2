#!/usr/bin/env python3 

import airsim
import math
import time
from airsim_ros_pkgs.msg import GimbalAngleEulerCmd
import rospy

# # connect to the AirSim simulator
# client = airsim.MultirotorClient()
# client.confirmConnection()
# client.enableApiControl(True)
# client.armDisarm(True)

# # MultirotorClient.wait_key('Press any key to takeoff')
# print("Taking off")
# client.takeoffAsync().join()
# print("Ready")

rospy.init_node("gimbalControl")
cmd = GimbalAngleEulerCmd()
cmd.camera_name = "front_center"
cmd.vehicle_name = "PX4"
cmd.yaw = -90
gimbal = rospy.Publisher('/airsim_node/gimbal_angle_euler_cmd', GimbalAngleEulerCmd, queue_size=1)
while not rospy.is_shutdown():
    gimbal.publish(cmd)
    time.sleep(0.5)
    # cmd.yaw *= -1