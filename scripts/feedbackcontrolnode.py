#!/usr/bin/env python3 
# license removed for brevity
from ast import And
import airsim
from airsim_ros_pkgs.msg import GimbalAngleEulerCmd, GPSYaw
import rospy
from vision_msgs.msg import BoundingBox2D,Detection2D
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import TimeReference,NavSatFix
from mavros_msgs.msg import OverrideRCIn, State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, CommandTOL, CommandTOLRequest, SetMode, SetModeRequest
from std_msgs.msg import Float64
import math
from math import atan2
import os,re
import sys
import numpy as np
import datetime, time
from pathlib import Path
# import pandas as pd
global horizontalerror, verticalerror, sizeerror
time_lastbox = None

#------------OPTIONS FOR MODES-------#
OPT_FLOW_MASTER = True # true means optical flow is used
top_down_mode = False  # different operating mode for motion if the drone is far above the feature
hybrid_mode = True # starts horizontal, moves toward object, then once above it, moves into the top-down-mode
yaw_mode = True # whether or not to yaw the entire drone during motion
USE_PITCH_ERROR = True
forward_scan_option = False # this is primarily for smoke generator and grenade experiments, otherwise hybrid mode should work well over a more static plume
fixed_heading_option = True # this mode tells the drone to sit above the smoke for a solid 5 seconds in order to get a heading from the mean flow direction, and then it goes to a fixed height and moves in that direction
controlled_descent_option = True

debugging = False
if debugging:
    print('########### DEBUGGING MODE #############')
#---------------------------------------------------#

# create saving directory
# username = os.getlogin( )
tmp = datetime.datetime.now()
stamp = ("%02d-%02d-%02d" % 
    (tmp.year, tmp.month, tmp.day))
# maindir = Path('/home/%s/1FeedbackControl' % username)
maindir = Path('./SavedData')
runs_today = list(maindir.glob('*%s*_fc-data' % stamp))
if runs_today:
    runs_today = [str(name) for name in runs_today]
    regex = 'run\d\d'
    runs_today=re.findall(regex,''.join(runs_today))
    runs_today = np.array([int(name[-2:]) for name in runs_today])
    new_run_num = max(runs_today)+1
else:
    new_run_num = 1
savedir = maindir.joinpath('%s_run%02d_fc-data' % (stamp,new_run_num))
os.makedirs(savedir)  
fid = open(savedir.joinpath('Feedbackdata.csv'),'w')
fid.write('Timestamp_Jetson,Timestamp_GPS,GPS_x,GPS_y,GPS_z,GPS_lat,GPS_long,GPS_alt_rel,Surveying,Sampling,MoveUp,AboveObject,Pitch,Size_error,OptFlow_On,Flow_x,Flow_y,Vspeed,Fspeed,Hspeed\n')

print_pitch = False
print_size_error = False
print_mode = False
print_vspeed = False
print_yawrate = False
print_flow=True
print_alt = False
print_time = False



# print_
# bounding box options
setpoint_size = 0.9 #fraction of frame that should be filled by target. Largest axis (height or width) used.
setpoint_size_approach = 1.5 # only relevant for hybrid mode, for getting close to target
# deadzone_size = 0.0 #deadzone on controlling size
# deadzone_position = 0.0 #deadzone on controlling position in frame

# optical flow parameters
alt_flow = 3 # altitude at which to stop descent and keep constant for optical flow
alt_sampling = 6 # 1.5 # altitude setpoint at which to do a controlled sampling based on mean flow direction
alt_min = 0.5 # minimum allowable altitude


# gain values
size_gain = 1
yaw_gain = 1
gimbal_pitch_gain = -40 # previously -100, this needs to be adjusted depending on teh frame rate for detection (20fps=-50gain, while saving video; 30fps=-25gain with no saving)
gimbal_yaw_gain = 12 # previously 30, adjusting now for faster yolo

traverse_gain = 2.5
# traverse_gain = 3
flow_gain = 10 # previously 0.25 when it worked ok but was a bit fast, wind speed < 5mph
# flow_survey_gain = 300 # made flow gain much larger from simple calculation (10 pixel movement measured when drone should respond about 1.5 m/s ish)
flow_survey_gain = 60 # gain per meter of altitude
# vertical_gain = 3 # half of the size_gain value
vertical_gain = 2 # half of the size_gain value
# new gain for error term for pitch
pitcherror_gain_min = 0.75 # sets a floor on how much teh drone can be slowed down by pitch feedback
                           # larger value (max 1) means the forward speed will be attenuated less (i.e., not as slowed down by teh pitch feedback) 
# limit parameters
yaw_center = 1500 # 1500 is straight ahead
# 1000 is straight ahead
pitch_up=1000 # this value seems to drift sometimes
alt_delta=0 # how high to move after crossing munumum alt
limit_speed = 2
limit_speed_v = 1 # different speed limit for changing altitude
limit_yawrate = .4
limit_pitchchange = 100
limit_yawchange = 100
limit_max_yaw = yaw_center+500 # 2000 is 45 deg right
limit_min_yaw = yaw_center-500 # 2000 is 45 deg left
move_up_speed=0.5
alt_set_appr_speed = -0.5
fscan_speed = 1
sizeerror_flow_thresh = 0.1 # this is the fraction of the setpoint size that the error needs to be within in order to initiate optical flow

# initialize
guided_mode = False
horizontalerror = 0 
verticalerror=0 
sizeerror=0 
vspeed = 0 # positive is upwards
hspeed = 0 # positive is to the right
fspeed = 0 # positive is forwards
flow_x = flow_y = flow_t = 0
yaw = 0
yawrate = 0
move_up = False # initialized value
moving_to_set_alt = False # only used after optical flow survey
alt = 10 # initialized value outside the safegaurd
gps_x = gps_y = gps_t = 0
gps_lat = gps_long = gps_alt = gps_alt_rel = 0
above_object = False
sampling = False
forward_scan = True # this should be on to start
surveying = False
OPT_FLOW=False
OPT_COMPUTE_FLAG = False
MOVE_ABOVE = False
pitch_down = pitch_up+900 #1900 is 90 deg down
pitch_thresh = pitch_down-200
pitch_45 = pitch_up + (pitch_down - pitch_up)//2
if forward_scan_option or top_down_mode:
    pitch_init = pitch_down 
else:
    pitch_init = pitch_45
pitchcommand = pitch_init
yawcommand = yaw_center
#ADDED for airsim
airsim_yaw = 0
publish_rate = 0

# def changePitch(p):
#     global gimbal
#     pitch = (1000 - p) / 10
#     cmd = GimbalAngleEulerCmd()
#     cmd.camera_name = "front_center"
#     cmd.vehicle_name = "PX4"
#     cmd.pitch = pitch
#     gimbal.publish(cmd)

# def changeYaw(yawcommand):
#     global gimbal
#     yaw = ((yawcommand - yaw_center) * 45) / 500
#     cmd = GimbalAngleEulerCmd()
#     cmd.camera_name = "front_center"
#     cmd.vehicle_name = "PX4"
#     cmd.yaw = yaw
#     gimbal.publish(cmd)

def moveAirsimGimbal(pitchcommand, yawcommand):
    '''
    Converts gimbal's pwm commands to angles for running is simulation
    pitchcommand - Pitch PWM. 1000 is straight ahead (0 deg) and 1900 is straight down (-90 deg) 
    yawcommand - Yaw PWM. 1000 is -45 deg and 2000 is 45 deg

    '''
    global gimbal, airsim_yaw, yaw
    airsim_yaw = math.degrees(yaw)
    # print("Drone Yaw RAW =", airsim_yaw, "AFTER =", airsim_yaw+360 if airsim_yaw<0 else airsim_yaw)
    if airsim_yaw<0:
        airsim_yaw += 360
    gimbal_pitch = (1000 - pitchcommand) / 10 
    gimbal_yaw = ((yaw_center - yawcommand) * 45) / 500
    cmd = GimbalAngleEulerCmd()
    cmd.camera_name = "front_center"
    cmd.vehicle_name = "PX4"
    cmd.pitch = gimbal_pitch
    # cmd.pitch = -10
    # cmd.yaw = -airsim_yaw + 90
    cmd.yaw = gimbal_yaw - airsim_yaw + 90
    if cmd.yaw>=360:
        cmd.yaw = cmd.yaw % 360
    
    # print("Cam Angle= ", cmd.yaw, ", Gimbal Pitch = ", cmd.pitch, "AirsimYaw =", airsim_yaw, "Gimbal Yaw =", gimbal_yaw)
    gimbal.publish(cmd)

def offboard():
    # Set to OFFBOARD mode so that it uses mavros commands for navigation
    mode = SetModeRequest()
    mode.base_mode = 0
    mode.custom_mode = "OFFBOARD"
    print("Setting up...", end ="->")
    setm = rospy.ServiceProxy('/mavros/set_mode', SetMode)
    resp = setm(mode)
    print(resp)
    # time.sleep(1) 

# def getAirsimYaw(state):
#     global airsim_yaw
#     airsim_yaw = state.yaw
#     print("Airsim yaw = ", airsim_yaw)
    

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def pose_callback(pose):
    global yaw,alt, gps_x, gps_y
    q = pose.pose.orientation
    # yaw = atan2(2.0*(q.y*q.z + q.w*q.x), q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z)
    r,p,y = euler_from_quaternion(q.x,q.y,q.z,q.w)
    yaw = y
    alt = pose.pose.position.z
    # if print_alt:
    # print(f"Altitude: {alt} m")
    gps_x = pose.pose.position.x
    gps_y = pose.pose.position.y
    # moveAirsimGimbal(pitchcommand, yawcommand)

    

def state_callback(state):
    """
    check if drone FCU is in loiter or guided mode
    """
    global guided_mode
    # if state.mode == 'GUIDED':
    # print("STATE:", state.mode)
    if state.mode == 'OFFBOARD':
        guided_mode = True
    else:
        print("!!!!!!!!!!!!!!!! NOT OFFBOARD")
        guided_mode = False
    # print(f'Guided mode: {guided_mode}')
    # print(state.mode)
def time_callback(gpstime):
    global gps_t
    gps_t = float(gpstime.time_ref.to_sec())
    # gps_t = gps_t
    if print_time: print(f"Time: {gps_t}")
    # print(gps_t)
    # print(rospy.get_time())
    # print(time.time())
def gps_callback(gpsglobal):
    global gps_lat,gps_long, gps_alt
    gps_lat = gpsglobal.latitude
    gps_long = gpsglobal.longitude
    gps_alt = gpsglobal.altitude
    # print(f'gps_alt {gps_alt}')
    
def rel_alt_callback(altrel):
    global gps_alt_rel
    gps_alt_rel = altrel.data # relative altitude just from GPS data
    # print(f'gps_alt_rel {gps_alt_rel}')


def boundingbox_callback(box):
    global horizontalerror, verticalerror, sizeerror
    global time_lastbox, pitchcommand, yawcommand
    global MOVE_ABOVE, OPT_FLOW
    # positive errors give right, up
    if box.bbox.center.x != -1:
        time_lastbox = rospy.Time.now()
        # bboxsize = max(box.bbox.size_x, box.bbox.size_y)
        bboxsize = (box.bbox.size_x + box.bbox.size_y)/2 # take the average so that a very thin, long box will still be seen as small
        
        if not above_object: # different bbox size desired for approach and above stages for hybrid mode
            sizeerror = setpoint_size_approach - bboxsize # if box is smaller than setpoit, error is positive
            # if USE_PITCH_ERROR:
            #     # this should keep the drone moving forward even when the fov is filled with the object, but needs to have a low gain
            #     sizeerror += (pitch_down - pitchcommand)/(pitch_down - pitch_up)
        else:
            sizeerror = setpoint_size - bboxsize # if box is smaller than setpoit, error is positive
        
    

        if not OPT_FLOW:
            horizontalerror = .5-box.bbox.center.x # if box center is on LHS of image, error is positive
            verticalerror = .5-box.bbox.center.y # if box center is on upper half of image, error is positive 
            # print('Horzontal error: %f' % horizontalerror)
            # print('Vertical error: %f' % verticalerror)
            pitchdelta = verticalerror * gimbal_pitch_gain
            pitchdelta = min(max(pitchdelta,-limit_pitchchange),limit_pitchchange)
            # print(f"Pitch command,delta: {pitchcommand},{pitchdelta}")
            pitchcommand += pitchdelta
            pitchcommand = min(max(pitchcommand,1000),2000)
            yawdelta = horizontalerror * gimbal_yaw_gain
            yawdelta = min(max(yawdelta,-limit_yawchange),limit_yawchange)
            # print(f"Yaw command, delta: {yawcommand},{yawdelta}")
            yawcommand += yawdelta
            yawcommand = min(max(yawcommand,1000),2000)
        if print_size_error:
            print('Setpoint - bbox = %f' % sizeerror)

        if pitchcommand < pitch_thresh and bboxsize > 0.75: # if close and gimbal pitched upward, move to get above the object
            MOVE_ABOVE = True
            # pass
            # print('Moving above command')

        
    return

def flow_callback(flow):
    global horizontalerror, verticalerror,time_lastbox
    global pitchcommand, yawcommand
    global flow_x,flow_y,flow_t
    global OPT_FLOW,OPT_COMPUTE_FLAG
    
    # typical values might be around 10 pixels, depending on frame rate
    flow_x = flow.size_x # movement to teh right in the image is positive
    flow_y = -flow.size_y # this is made negative so that positive flow_y means the object was moving toward the top of the image (RAFT returns negative y for this (i.e., toward smaller y coordinates))
    # now movement to the top is positive flow_y
    flow_t = float(gps_t)
    # adjust the feedback error using the optical flow
    if OPT_FLOW:
        print('doing optical flow feedback')

        # if not OPT_COMPUTE_FLAG:
        #     horizontalerror = verticalerror = 0
        OPT_COMPUTE_FLAG = True # this signals to later in the code that the first usable optical flow data can be pplied (instead of inheriting errors from bbox callback)
        horizontalerror = -flow_x # to be consistent with teh bounding box error
        verticalerror = flow_y
        if not above_object: # this should never end up being called normally, just for debugging optical flow in side-view
            # pitch
            pitchdelta = verticalerror * gimbal_pitch_gain
            pitchdelta = min(max(pitchdelta,-limit_pitchchange),limit_pitchchange)
            pitchcommand += pitchdelta
            pitchcommand = min(max(pitchcommand,1000),2000)
            yawdelta = horizontalerror * gimbal_yaw_gain
            yawdelta = min(max(yawdelta,-limit_yawchange),limit_yawchange)
            yawcommand += yawdelta
            yawcommand = min(max(yawcommand,1000),2000)
        if print_flow:
            print('Flow x,y = %f,%f' % (flow_x,flow_y))
        
        time_lastbox = rospy.Time.now()
    return

def dofeedbackcontrol():
    global pitchcommand, yawcommand
    global above_object, forward_scan
    global yaw_mode,OPT_FLOW,OPT_COMPUTE_FLAG,MOVE_ABOVE
    global move_up, USE_PITCH_ERROR
    global moving_to_set_alt
    global hspeed,vspeed,fspeed
    global yawrate
    global horizontalerror,verticalerror
    global twistpub, twistmsg,rcmsg,rcpub, gimbal
    global publish_rate
    publish_rate = time.time()

    #Initialize publishers/subscribers/node
    # print("Initializing feedback node...")
    # rospy.init_node('feedbacknode', anonymous=False)
    rospy.Subscriber('/bounding_box', Detection2D, boundingbox_callback)
    rospy.Subscriber('/mavros/local_position/pose', PoseStamped, pose_callback)
    rospy.Subscriber('/mavros/time_reference',TimeReference,time_callback)
    rospy.Subscriber('/mavros/global_position/global',NavSatFix,gps_callback)
    rospy.Subscriber('/mavros/global_position/rel_alt',Float64,rel_alt_callback)
    rospy.Subscriber('/flow',BoundingBox2D,flow_callback)
    rospy.Subscriber('/mavros/state',State,state_callback)
    # rospy.Subscriber('/airsim_node/origin_geo_point',GPSYaw,getAirsimYaw)

    twistpub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=1)
    rcpub = rospy.Publisher('/mavros/rc/override', OverrideRCIn, queue_size=1)
    gimbal = rospy.Publisher('/airsim_node/gimbal_angle_euler_cmd', GimbalAngleEulerCmd, queue_size=1)

    # print(rospy.get_published_topics())

    # control loop
    twistmsg = Twist()
    rcmsg = OverrideRCIn()
    rcmsg.channels = np.zeros(18,dtype=np.uint16).tolist()
    rate = rospy.Rate(20) # 20hz - originally 20hz


    print("Feedback node initialized, starting control")
    while not rospy.is_shutdown():
        # offboard()
        # t1 = time.time()
            # move_up = True # will move up until desired altitude, to reset
        # elif alt > alt_min+alt_delta:
            # move_up = False     # desired altitude reached
 
        if forward_scan_option and forward_scan:
            # in this mode, the drone will just start moving forward until it sees the smoke below it
            fspeed = fscan_speed
            vspeed = 0
            hspeed = 0
            print('Scanning for object')
        

        #feedback control algorithm
        #don't publish if message is old
        # if (time_lastbox != None and rospy.Time.now() - time_lastbox < rospy.Duration(.5)) or debugging:
        if not guided_mode:
            print("NOT GUIDED")
            pitchcommand = pitch_init 
            yawcommand = yaw_center
            fspeed = hspeed = vspeed = 0
            yaw_mode = True # turn yaw back on
            above_object = False
            moving_to_set_alt = False
            OPT_FLOW = False # turn off teh optical flow mode
            OPT_COMPUTE_FLAG = False
            if forward_scan_option:
                # turn this initial mode back on
                forward_scan = True
        elif (time_lastbox != None and rospy.Time.now() - time_lastbox < rospy.Duration(.5)):

            # t1 = time.time()
            
            # safeguard for vertical motion
            if alt < alt_min:
                rise_up(dz = 2,vz=0.5) #ORIGINAL
                # rise_up(dz = 1,vz=2)
            # end the forward scan phase once in teh air looking down and recognizing a particle
            # only do this if this option is turned on
            if forward_scan_option and forward_scan:
                if above_object and alt > alt_flow and pitchcommand > pitch_thresh:
                    forward_scan = False # setting this to false ensures that it wont be triggered again at the next pass
                    fspeed = 0
            
            # to try and get the drone to move up if the smoke plume is only in front of it
            if MOVE_ABOVE:
                rise_up(dz=3,vz = 1) #ORIGINAL
                # rise_up(dz=1.5,vz = 2)
                MOVE_ABOVE = False
                continue

            if top_down_mode: # largely deprecated, combined with hybrid mode
                if alt < alt_flow and above_object:
                    if OPT_FLOW_MASTER:
                        OPT_FLOW = True # switch to optical flow feedback control mode
                        if not OPT_COMPUTE_FLAG:
                            horizontalerror = verticalerror = 0
                # lateral movement (hspeed > 0 moves right)
                hspeed = -horizontalerror * traverse_gain
                # forward movement   (fspeed > 0 move backward)
                fspeed = verticalerror * traverse_gain

                # vertical movement depending on the minimum altitude safeguard
                # vspeed > 0 moves upward
                if move_up:
                    vspeed = move_up_speed
                else:
                    # vspeed=-1
                    vspeed = -sizeerror * vertical_gain # size error is negative because you want to move down (negative velocity) to get closer
                
                yawrate = 0 # don't rotate/yaw teh drone if in top-down mode 
                yawcommand = yaw_center   # stay centered
                pitchcommand = pitch_down # stare down



            elif hybrid_mode:

                

                # determine if above object based on pitch
                # if in a forward scan, then this will be triggered immediately when object is recognized, since pitch is already down
                if not above_object: # once entered above mode, cannot go back
                                    #  if this isn't done, then the transition between modes is really rough
                    if print_mode:
                        print('Hybrid mode: Approach phase')
                    if pitchcommand > pitch_thresh and alt > alt_min:
                        above_object=True
                        print('#----------ABOVE OBJECT-----------#')
                        # USE_PITCH_ERROR = False # turning this off once moving downward
                            
                            
                    else:

                        above_object=False

                else: 
                    if print_mode:
                        print('Hybrid mode: Above object')
                        
                #------FOR DEBUGGING------#
                if debugging:
                    above_object=True
                #-------------------------#
                if above_object and fixed_heading_option and not moving_to_set_alt: # only do this if not already moving to setpoint
                    
                    fspeed_surv,hspeed_surv = survey_flow()
                    moving_to_set_alt = True # reporting end of survey
                    fspeed = hspeed = vspeed = 0
                    continue

                # REMOVED SINCE NOT USING OPTICAL FLOW FOR ACTIVE FEEDBACK CONTROL ANYMORE
                # # initialize optical flow if above object and if we have lowered to desired height
                # # desired height can either be determined by the altitude choice or the proximity accuracy
                # if above_object and (alt < alt_flow or sizeerror < sizeerror_flow_thresh*setpoint_size):
                #     if OPT_FLOW_MASTER:
                #         OPT_FLOW = True # switch to optical flow feedback control mode
                #         if not OPT_COMPUTE_FLAG: # if optical flow data hasn't been received yet, errors should be kept at zero
                #             # horizontalerror = verticalerror = 0
                #             print('Reinitializing hspeed and fspeed to zero when beginning opt flow')
                #             horizontalerror = verticalerror = 0
                            
                # # lateral movement (hspeed > 0 moves right)
                # if OPT_FLOW and OPT_COMPUTE_FLAG:
                #     hspeed += -horizontalerror * flow_gain
                # else:
                #     hspeed = -horizontalerror * traverse_gain
                hspeed = -horizontalerror * traverse_gain
                # forward movement   (fspeed > 0 move forward)
                if above_object: # top-view, determining forward movement based on 
                    # if OPT_FLOW and OPT_COMPUTE_FLAG:
                    #     fspeed += verticalerror * flow_gain
                       
                    # else:
                    #     fspeed = verticalerror * traverse_gain
                    fspeed = verticalerror * traverse_gain
                        
                else: # side-view, determining approach speed based on size of bounding box
                    fspeed = sizeerror * size_gain

                    if USE_PITCH_ERROR:
                        # slow down the forward speed when the gimbal starts pitching down
                        fspeed_adjust = (pitch_down - pitchcommand)/(pitch_down - pitch_up)
                        fspeed_adjust = min(max(fspeed_adjust,1),pitcherror_gain_min)
                        fspeed *= fspeed_adjust
                
                
                # vertical movement depending on the minimum altitude safeguard
                if above_object:
                   
                    # if OPT_FLOW: # keep at steady height during optical flow
                    #     vspeed=0
                    # else:
                    #     if forward_scan_option: # in this mode, vertical motion will just be steady, and the focus is on horizontal
                    #         vspeed = -alt_set_appr_speed # this does better with very elongated plumes?
                    #     else:
                    #         vspeed = -sizeerror * vertical_gain # size error is negative 

                    if not fixed_heading_option:
                        vspeed = -sizeerror * vertical_gain # size error is negative because you want to move down (negative velocity) to get closer
                        
                else:
                    vspeed=0

                # assigning gimbal pitch and yaw depending on mode
                if print_vspeed:
                    print('Vertical speed: %f' % vspeed)
                if above_object:
                    yawrate = 0 # don't rotate/yaw teh drone if in top-down mode 
                    yaw_mode = False
                    pitchcommand = pitch_down
                    yawcommand = yaw_center
                else:
                    yawrate = ((yawcommand - yaw_center)/1000)*yaw_gain

                # if print_mode:
                #     print('Above object:')
                #     print(above_object)

            else: # nik's old version
                # set vertical motion to zero
                vspeed = 0

                #calculate raw commands
                if pitchcommand < 1800: 
                    fspeed = sizeerror * size_gain
                else: # if the gimbal is pitching down to see (i.e., pitch servo > 1800), stop going forward
                    fspeed = 0
                # yawrate = horizontalerror * yaw_gain #commented out when gimbal yaw is active
                yawrate = ((yawcommand - yaw_center)/1000)*yaw_gain
                hspeed = -horizontalerror * traverse_gain # this only gets used if yaw mode is off
            #---------------------------------#

        elif time_lastbox != None and (rospy.Time.now() - time_lastbox > rospy.Duration(0.5)) and moving_to_set_alt:
            # if its been more than half a second without detection during descent, stop lateral movement
            hspeed = 0
            fspeed = 0
        
        elif time_lastbox != None and (rospy.Time.now() - time_lastbox > rospy.Duration(5)) and not moving_to_set_alt: # added condition here so that even if smoke isn't seen, descent continues after survey
            # if nothing detected for 5 seconds, reset gimbal position, and if more than 10 seconds, go back to manual control from RC
            # also reinitializes other settings
            # print('#--------------RESETTING....------------#')
            pitchcommand = pitch_init 
            yawcommand = yaw_center
            fspeed = hspeed = vspeed = 0
            yaw_mode = True # turn yaw back on
            above_object = False
            OPT_FLOW = False # turn off teh optical flow mode
            OPT_COMPUTE_FLAG = False
            if forward_scan_option:
                # turn this initial mode back on
                forward_scan = True
            # if alt < 7.5:
            #     rise_up(dz=5) # stops movement laterally, and drone rises 5 meters
            # print("Duration check = ", rospy.Time.now() - time_lastbox)
            if (rospy.Time.now() - time_lastbox < rospy.Duration(10)):
                # print("Pitch Command -> ", pitchcommand)
                # print("Yaw Command -> ", yawcommand)
                rcmsg.channels[7] = int(pitchcommand) #send pitch command on channel 8
                rcmsg.channels[6] = int(yawcommand) #send yaw command on channel 7

        
        # out of loop, send commands
        
        # check if altitude setpoint reached
        if fixed_heading_option and moving_to_set_alt:
            alt_diff = alt - alt_sampling

            

            if debugging: alt_diff = 0
            
            if abs(alt_diff) < 0.5:
                print(f'Reached setpoint alt at {alt} m')
                vspeed = 0 # desired alttitude reached
                moving_to_set_alt = False
                sample_along_heading(fspeed_surv,hspeed_surv) # start sampling route
                rise_up(dz=5) # get out of smoke
            elif alt_diff < 0: # too low
                vspeed = abs(alt_set_appr_speed) # force to be positive
                print(f'Moving to setpoint alt at {vspeed} m/s')
            elif alt_diff > 0: # too high
                vspeed = -abs(alt_set_appr_speed) # force to be negative (move down)
                print(f'Moving to setpoint alt at {vspeed} m/s')
            
        #bound controls to ranges
        fspeed = min(max(fspeed,-limit_speed),limit_speed) #lower bound first, upper bound second
        hspeed = min(max(hspeed,-limit_speed),limit_speed)
        vspeed = min(max(vspeed,-limit_speed_v),limit_speed_v) # vertical speed
        yawrate = min(max(yawrate,-limit_yawrate),limit_yawrate)
        yawcommand = min(max(yawcommand,1000),2000)
        pitchcommand = min(max(pitchcommand,1000),2000)
        
        # print('fpeed:',fspeed)
        # print('dt',time.time()-t1)

        # horizontal motion
        if yaw_mode:
            twistmsg.linear.x = math.cos(yaw)*fspeed
            twistmsg.linear.y = math.sin(yaw)*fspeed
            # if MOVE_ABOVE:
            #     twistmsg.linear.z = 1
            # else:
            #     twistmsg.linear.z = 0
            twistmsg.angular.z = yawrate
        else:
            twistmsg.linear.x = math.cos(yaw)*fspeed + math.sin(yaw)*hspeed
            twistmsg.linear.y = math.sin(yaw)*fspeed - math.cos(yaw)*hspeed
            twistmsg.angular.z = 0
        
        # publishing
        twistmsg.linear.z = vspeed  # vertical motion
        rcmsg.channels[7] = int(pitchcommand) #send pitch command on channel 8
        rcmsg.channels[6] = int(yawcommand) #send yaw command on channel 7
        # print("Pitch Command -> ", pitchcommand)
        # print("Yaw Command -> ", yawcommand)
        # print("Publishing control cmd after", time.time() - publish_rate, "seconds")
        twistpub.publish(twistmsg)
        # rcpub.publish(rcmsg)
        moveAirsimGimbal(pitchcommand, yawcommand)
        publish_rate = time.time()
        if print_pitch:
            print('Pitch command: %f' % (pitchcommand))
        if print_yawrate:
            print('Yaw rate: %f' % yawrate)
        if print_alt:
            print(f"Altitude: {alt} m")
        
        # writing control states and data to csv
        save_log()
        
        rate.sleep()

def rise_up(dz = 5,vz=1):
    # simple loop to go up or down, usually to get above object
    print(f'Rising {dz}m at {vz}m/s...')
    global twistpub, twistmsg
    twistmsg.linear.x = 0
    twistmsg.linear.y = 0
    twistmsg.linear.z = vz
    
    for i in range(int((dz/vz)/0.2)):
        twistpub.publish(twistmsg)
        time.sleep(0.2)
    
    twistmsg.linear.z = 0
    twistpub.publish(twistmsg)
    return

def survey_flow():
    """
    separated loop that keeps the drone fixed while observing the flow in the bounding box below
    """
    global surveying
    global twistpub, twistmsg,rcmsg,rcpub
    # function for starting a survey of the flow from above, and then calling a heading to travel towards
    global fspeed,hspeed,vspeed
    survey_samples = 10
    survey_duration = 10
    # hold position for _ seconds
    twistmsg.linear.x = 0
    twistmsg.linear.y = 0
    twistmsg.linear.z = 0
    twistmsg.angular.z = 0 # stop any turning
    twistpub.publish(twistmsg)
    rcmsg.channels[7] = pitch_down #send pitch command on channel 8
    rcmsg.channels[6] = yaw_center #send yaw command on channel 7
    rcpub.publish(rcmsg)
    
    t1 = gps_t
    t0 = time.time()
    vx = []
    vy = []
    #-----DEBUGGING-------#
    if debugging:
        for iii in range(survey_samples):
            vx.append(1)
            vy.append(3)
    #---------------------#
    t_log = time.time()
    flow_prev = flow_x
    print('#-------------------Beginning survey---------------------#')
    surveying = True
    while True: # collect samples until a certain number reached, and for at least 5 seconds
        if not guided_mode:
            return 0,0
        if len(vx) >= survey_samples and (time.time() - t0 > 5): # do this at least 5 seconds duration
            break
        if flow_t > t1 and flow_x != flow_prev: # only use flow values after this sequence starts
                if ~np.isnan(flow_x):   # only use if not nan
                    vx.append(flow_x*flow_survey_gain*(alt-3))
                    vy.append(flow_y*flow_survey_gain*(alt-3)) # gain is attenuated or amplified by altitude
                    
                flow_prev = flow_x
                

                    
        if time.time()-t0 > 30: # only do this for 15 seconds at most
            print('#-------------------Failed to determine heading--------------#')
            # heading_obtained = False
            return 0,0
        if time.time()-t_log > 0.1: # do this at 10Hz at most
            save_log() # makes sure data keeps being saved with GPS
            t_log = time.time()
            rcmsg.channels[7] = pitch_down #send pitch command on channel 8
            rcmsg.channels[6] = yaw_center #send yaw command on channel 7
            rcpub.publish(rcmsg)
            rcpub.publish(rcmsg)
            twistpub.publish(twistmsg)
            # print('latest flow time',flow_t)

    
    fspeed_surv = np.nanmean(vy) # vertical velocity
    hspeed_surv = np.nanmean(vx)  # horizontal
    print('#-----------------------Got heading-------------#')
    print('Samples collected: vx')
    print(vx)
    print('Samples collected: vy')
    print(vy)
    print('Mean vx: %f, Mean vy: %f' % (hspeed_surv,fspeed_surv))
    
    # check speeds, lower below limit but keep direction
    # shouldn't need to worry about dividing by zero, since the limits are above this
    if abs(fspeed_surv) > limit_speed:
        hspeed_surv *= abs(limit_speed/fspeed_surv)
        fspeed_surv = limit_speed * fspeed_surv/abs(fspeed_surv) # keep the sign
        
    if abs(hspeed_surv) > limit_speed:
        fspeed_surv *= abs(limit_speed/hspeed_surv)
        hspeed_surv = limit_speed * hspeed_surv/abs(hspeed_surv)
        
    print('Heading after clipping: %f, %f' % (hspeed_surv,fspeed_surv))
  
    # should be below limits now, not needing these next lines
    # fspeed = min(max(fspeed,-limit_speed),limit_speed) #lower bound first, upper bound second
    # hspeed = min(max(hspeed,-limit_speed),limit_speed)
    

    
    # heading_obtained = True
    # set_heading(fspeed_surv,hspeed_surv)
    print('#----------------------Survey complete------------------------#')
    surveying = False
    return fspeed_surv,hspeed_surv

        
def sample_along_heading(fspeed_surv,hspeed_surv):
    """
    keeps fixed altitude and moves along a prescribed direction obtain from flow survey prior
    """
    global twistpub, twistmsg,rcmsg,rcpub
    # function for setting the flow direction obtained after surveying
    global sampling
    global fspeed,hspeed,vspeed
    # alt_diff = alt_sampling - alt
    # 
    # t_log = time.time()
    # print('Current altitude: %f' % alt)
    # 
    # print('#------------Moving to setpoint altitude (%f m)--------#' % alt_sampling)
    # # get to setpoint altitude
    # # vspeed_tmp = 0
    # while abs(alt_diff) > 0.5:
    #     if alt_diff < 0:    # if already above the setpoint
    #         vspeed_tmp = -0.75 # move downward
    #     elif alt_diff > 0:
    #         vspeed_tmp = 0.75 # move upward
    #     # print('Vertical speed command:',vspeed)
    #     vspeed_tmp = min(max(vspeed_tmp,-limit_speed_v),limit_speed_v) # vertical speed
        
    #     # print('Vertical speed command:',vspeed)
    #     twistmsg.linear.z = vspeed_tmp
    #     twistpub.publish(twistmsg)
    #     rcmsg.channels[7] = pitch_down #send pitch command on channel 8
    #     rcmsg.channels[6] = yaw_center #send yaw command on channel 7
    #     rcpub.publish(rcmsg)
    #     # rcpub.publish(rcmsg)
    #     alt_diff = alt_sampling - alt # update based on new altitude
    #     # print('latest alt diff',alt_diff)
    #     # print(gps_t)
    
    #     if time.time()-t_log > 0.1: # do this at 10Hz at most
    #         save_log() # makes sure data keeps being saved with GPS
    #         t_log = time.time()
    #         print('Current alt: %f m; Setpoint: %f m' % (alt,alt_sampling))
    #         print('Vertical speed command:',vspeed_tmp)
    #     # start lateral movements

    sampling = True
    
    # stop motion in z-direction
    twistmsg.linear.x = math.cos(yaw)*fspeed_surv + math.sin(yaw)*hspeed_surv
    twistmsg.linear.y = math.sin(yaw)*fspeed_surv - math.cos(yaw)*hspeed_surv
    twistmsg.linear.z = 0 # stay at that height
    # keeping moving laterally for 20 seconds after reaching set height
    t_log = time.time()
    t0 = time.time()
    moveAirsimGimbal(1000, 1500)
    while time.time()-t0 < 20:
        twistpub.publish(twistmsg)
        rcmsg.channels[7] = pitch_down #send pitch command on channel 8
        rcmsg.channels[6] = yaw_center #send yaw command on channel 7
        rcpub.publish(rcmsg)
        rcpub.publish(rcmsg)
        if time.time()-t_log > 0.1: # do this at 10Hz at most
            save_log() # makes sure data keeps being saved with GPS
            t_log = time.time()
            print('#------------------Sampling...---------------------------#')
        
        
    print('#---------------Sampling...DONE-----------------------#')
    sampling = False
    return


def save_log():
    """
    writing data to csv
    """
    fid.write('%f,%f,%f,%f,%f,%f,%f,%f,%s,%s,%s,%s,%f,%f,%s,%f,%f,%f,%f,%f\n' % 
        (time.time(),gps_t,gps_x,gps_y,alt,gps_lat,gps_long,gps_alt_rel,str(surveying),str(sampling),str(move_up),str(above_object),pitchcommand,sizeerror,str(OPT_FLOW),flow_x,flow_y,vspeed,fspeed,hspeed))

if __name__ == '__main__':
    print("Initializing feedback node...")
    rospy.init_node('feedbacknode', anonymous=False)
    # rospy.init_node("droneControl")
    twist_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=1)
    # arm_pub = rospy.Publisher('/mavros/cmd/arming', CommandBool, queue_size=1)
    # takeoff_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel_unstamped', CommandTOL, queue_size=1)

    # TAKEOFF
    takeoff = CommandTOLRequest()
    takeoff.min_pitch = 0
    takeoff.yaw = 0
    takeoff.latitude = 47.641468
    takeoff.longitude = -122.140165
    takeoff.altitude = -10
    print("Taking off", end ="->")
    fly = rospy.ServiceProxy('/mavros/cmd/takeoff', CommandTOL)
    resp = fly(takeoff)
    print(resp)
    time.sleep(2) 

    # ARM the drone
    arm = CommandBoolRequest()
    arm.value = True
    print("Arming - 1st attempt", end ="->")
    # arm_pub.pub(arm)
    arming = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
    resp = arming(arm)
    print(resp)
    time.sleep(5)
    print("Arming - 2nd attempt", end ="->")
    resp = arming(arm)
    print(resp)
    time.sleep(5)
    # print("Armed (?)") 

    # Move to set posiiton
    print("Moving...")
    go = PoseStamped()
    go.pose.position.x = 0
    go.pose.position.y = 0
    go.pose.position.z = 20
    go.pose.orientation.z = -0.8509035
    go.pose.orientation.w = 0.525322
    twist_pub.publish(go)
    time.sleep(0.2)

    offboard()
    
    for i in range(30):
        #Start poition (X=-66495.023860,Y=49467.376329,Z=868.248719)
        # Move to set posiiton
        print("Moving...")
        go = PoseStamped()
        go.pose.position.x = -10
        go.pose.position.y = 10
        go.pose.position.z = 20
        go.pose.orientation.z = 1
        # go.pose.orientation.w = 0.525322
        twist_pub.publish(go)
        time.sleep(0.2)

        # #Start poition (X=-69645.023860,Y=46533.376329,Z=898.248719)
        # # Move to set posiiton
        # print("Moving...")
        # go = PoseStamped()
        # go.pose.position.x = 0
        # go.pose.position.y = 10
        # go.pose.position.z = 20
        # go.pose.orientation.z =  0.7071068
        # go.pose.orientation.w =  0.7071068
        # twist_pub.publish(go)
        # time.sleep(0.2)

    print("GOING AUTONOMOUS")
    time.sleep(5)
    try:
        dofeedbackcontrol()
    except rospy.ROSInterruptException:
        pass

