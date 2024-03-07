#!/usr/local/bin/python3.10
# license removed for brevity

from ast import And
from pickle import TRUE
import rospy
from vision_msgs.msg import BoundingBox2D,Detection2D
from geometry_msgs.msg import Twist, PoseStamped, TwistStamped
from sensor_msgs.msg import TimeReference,NavSatFix
from mavros_msgs.msg import OverrideRCIn, State
from geographic_msgs.msg import GeoPoseStamped
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import CommandBool, CommandBoolRequest, CommandTOL, CommandTOLRequest, SetMode, SetModeRequest
from std_msgs.msg import Float64, String
import math
from math import atan2
import os,re
import sys
import numpy as np
import datetime, time
from pathlib import Path


def gimbal_control(): 
    rospy.init_node("gimbalControl")
    rcmsg = OverrideRCIn()
    rcmsg.channels = np.zeros(18,dtype=np.uint16).tolist()
    rcpub = rospy.Publisher('/mavros/rc/override', OverrideRCIn, queue_size=1)
    publish_rate = time.time()
    rate = rospy.Rate(20) # originally 10hz

    print('Inside Rospy shutdown')
    while not rospy.is_shutdown():
        # rcmsg.channels[7] = int(1000) #90 down send pitch command on channel 8
        # rcmsg.channels[6] = int(1000) #send yaw command on channel 7
        # print('Publishing')
        # rcpub.publish(rcmsg)
        print(f'time.time(): {int(time.time())}', end='\r')
        if (int(time.time())%4 == 0):
            rcmsg.channels[9] = int(1000)
            rcmsg.channels[8] = int(1000)
            rcmsg.channels[7] = int(1000) #90 down send pitch command on channel 8
            rcmsg.channels[6] = int(1000) #send yaw command on channel 7
            rcmsg.channels[5] = int(1000)
            rcpub.publish(rcmsg)
            print('Down', end='\r')
        else:
            rcmsg.channels[9] = int(1900)
            rcmsg.channels[8] = int(1900)
            rcmsg.channels[7] = int(1900) #Front send pitch command on channel 8
            rcmsg.channels[6] = int(1900) #send yaw command on channel 7
            rcmsg.channels[5] = int(1900)
            rcpub.publish(rcmsg)
            print('Up', end='\r')
            
        rate.sleep()

if __name__ == '__main__':
    try:
        gimbal_control()
    except rospy.ROSInterruptException:
        pass
