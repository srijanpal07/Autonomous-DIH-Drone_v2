#!/usr/local/bin/python3.10 
# license removed for brevity
from ast import And
from pickle import TRUE
import airsim
from airsim_ros_pkgs.msg import GimbalAngleEulerCmd, GPSYaw
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
from filterpy.kalman import KalmanFilter


global horizontalerror, verticalerror, sizeerror
time_lastbox = None
global horizontalerror_smoketrack, verticalerror_smoketrack, sizeerror_smoketrack, horizontalerror_smoketrack_list
horizontalerror_smoketrack_list =[]
# global body_vel_pub
time_lastbox_smoketrack = None
global head, slope_deg
head, slope_deg = 0.0, 0.0
global source_gps
source_gps = [0.0, 0.0, 0.0] # lattitude, longitude and altitude
global setpoint_global_pub


#------------OPTIONS FOR MODES-------#
OPT_FLOW_MASTER = True # true means optical flow is used
top_down_mode = False  # different operating mode for motion if the drone is far above the feature
hybrid_mode = True # starts horizontal, moves toward object, then once above it, moves into the top-down-mode
yaw_mode = True # Previously True # whether or not to yaw the entire drone during motion
USE_PITCH_ERROR = False # Previuosly True
forward_scan_option = True # Previously False changes to True this is primarily for smoke generator and grenade experiments, otherwise hybrid mode should work well over a more static plume
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
print_flow=False
print_alt = False

#-----------------------srijan----------------------------#
print_stat = False
print_stat_test = True
print_flags = False
print_state = True
print_speeds = False

sampling_time = 250 # previously 20, changed to 10 to cope up with smoke direction change
smoke_dir = ''
first_look_around = True
sample_along_heading = False
global sampling_t0
sampling_t0 = None
global track_sampling_time
track_sampling_time = True
global fspeed_head, hspeed_head
global print_iter
print_iter = 0
#------------------------srijan---------------------------#


# bounding box options
setpoint_size = 0.9 #fraction of frame that should be filled by target. Largest axis (height or width) used.
setpoint_size_approach = 1.5 # only relevant for hybrid mode, for getting close to target

# optical flow parameters
alt_flow = 3 # altitude at which to stop descent and keep constant for optical flow
alt_sampling = 5 # 1.5 # altitude setpoint at which to do a controlled sampling based on mean flow direction
alt_min = 1 # minimum allowable altitude

# gain values
size_gain = 1
yaw_gain = 1
gimbal_pitch_gain = -40 # previously -100, this needs to be adjusted depending on teh frame rate for detection (20fps=-50gain, while saving video; 30fps=-25gain with no saving)
gimbal_yaw_gain = 12 # previously 30, adjusting now for faster yolo

traverse_gain = 2.5
# traverse_gain = 3simSetCameraOrientation
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
limit_speed_v = 2 # originally 1 # different speed limit for changing altitude
limit_yawrate = 20
limit_pitchchange = 100
limit_yawchange = 100
limit_max_yaw = yaw_center+500 # 2000 is 45 deg right
limit_min_yaw = yaw_center-500 # 2000 is 45 deg left
move_up_speed=3
alt_set_appr_speed = -3 # previously 1.5
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
yawrate = 5 # previously 0 (not sue if this was the original value)
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



# --------------------- Kalman Filter Initialization --------------------- #
# Create a Kalman filter for yaw angle prediction
global kf, previous_yaw_measurements
previous_yaw_measurements = []
kf = KalmanFilter(dim_x=2, dim_z=1)

# Define the state transition matrix (constant velocity model)
kf.F = np.array([[1.0, 1.0],
                [0.0, 1.0]])

# Define the measurement function (identity matrix for simplicity)
kf.H = np.array([[1.0, 0.0]])

# Define the process noise covariance matrix
kf.Q = np.array([[0.01, 0.0],
                [0.0, 0.01]])

# Define the measurement noise covariance matrix
kf.R = np.array([[0.1]])

# Initialize the state estimate and covariance matrix
initial_yaw = 0.0  # Initial yaw angle
initial_yaw_rate = 0.0  # Initial yaw rate
kf.x = np.array([[initial_yaw],
                [initial_yaw_rate]])  # Initial state estimate
kf.P *= 100.0  # Initial covariance matrix

# --------------------- Kalman Filter Initialization --------------------- #



def moveAirsimGimbal(pitchcommand, yawcommand):
    '''
    Converts gimbal's pwm commands to angles for running is simulation
    pitchcommand - Pitch PWM. 1000 is straight ahead (0 deg) and 1900 is straight down (-90 deg) 
    yawcommand - Yaw PWM. 1000 is -45 deg and 2000 is 45 deg

    '''
    if print_stat: print(f'-------------------Inside moveAirsimGimbal()-------------------\npithchcommand: {pitchcommand}, yawcommand: {yawcommand}')
    global gimbal, airsim_yaw, yaw
    airsim_yaw = math.degrees(yaw)
    # print("Drone Yaw RAW =", airsim_yaw, "AFTER =", airsim_yaw+360 if airsim_yaw<0 else airsim_yaw)
    if airsim_yaw<0:
        airsim_yaw += 360
    gimbal_pitch = (1000 - pitchcommand) / 10 # previously: gimbal_pitch = (1000 - pitchcommand) / 10 , (280 - pitchcommand) / 18 , 
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



def euler_to_quaternion(roll, pitch, yaw):
    # Convert Euler angles to quaternion
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # Calculate quaternion components
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return x, y, z, w



def pose_callback(pose):
    global yaw,alt, gps_x, gps_y
    q = pose.pose.orientation
    # yaw = atan2(2.0*(q.y*q.z + q.w*q.x), q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z)
    r,p,y = euler_from_quaternion(q.x,q.y,q.z,q.w)
    yaw = y
    alt = pose.pose.position.z
    gps_x = pose.pose.position.x
    gps_y = pose.pose.position.y
    if print_stat: print(f"----------------Inside pose_callback():----------------\nalt: {alt}, gps_x:{gps_x}, gps_y:{gps_y}")
    if print_stat: print(f'yaw: {yaw}, alt: {alt}, gps_x:{gps_x}, gps_y:{gps_y}')



def compass_hdg_callback(heading):
    global head
    global source_gps
    global sample_along_heading
    global slope_deg

    head = heading.data
    if print_stat: print(f'Heading: {head}')
    dlat = math.radians(gps_lat-source_gps[0])
    dlon = math.radians(gps_long-source_gps[1])
    slope_rad = math.atan2(dlon, dlat)
    slope_deg = math.degrees(slope_rad)
    if print_stat_test and not sample_along_heading: print(f'Heading: {head}, Slope(deg): {slope_deg}, Diff: {head - slope_deg}, Slope(rad): {slope_rad} ', end='\r')
    # if sample_along_heading: 
        # smoketrack_yaw()


def smoketrack_yaw():
    global slope_deg
    global setpoint_global_pub
    global head
    global gps_lat, gps_long, gps_alt

    yaw = head - slope_deg
    setpoint = _build_global_setpoint2(gps_lat, gps_long, gps_alt, yaw) # yaw = 0 is heading 90 in airsim
    setpoint_global_pub.publish(setpoint)
    if print_stat_test: print(f'smoketrack_yaw activated! Heading: {head}, Slope(deg): {slope_deg}, Diff: {head - slope_deg}')




def state_callback(state):
    """
    check if drone FCU is in loiter or guided mode
    """
    global guided_mode
    global print_state
    if print_stat: print(f"----------------Inside state_callback():----------------\nstate.mode={state.mode}")
    if state.mode == 'OFFBOARD':
        guided_mode = True
        if print_stat: print('OFFBOARD')
        if print_flags and print_state: 
            print("state.mode == 'OFFBOARD' -> guided_mode = True")
            print_state = False
    else:
        print("!!!!!!!!!!!!!!!! NOT OFFBOARD")
        guided_mode = False
        if print_flags: print("state.mode == 'Not OFFBOARD' -> guided_mode = False")



def time_callback(gpstime):
    global gps_t
    gps_t = float(gpstime.time_ref.to_sec())
    # gps_t = gps_t
    if print_stat: print(f"----------------Inside time_callback:----------------\nTime: {gps_t}")



def gps_callback(gpsglobal):
    global gps_lat, gps_long, gps_alt

    gps_lat = gpsglobal.latitude
    gps_long = gpsglobal.longitude
    gps_alt = gpsglobal.altitude
    if print_stat: print(f"----------------Inside gps_callback():----------------\ngps_alt: {gps_alt}, gps_long: {gps_long}, gps_alt: {gps_alt}")
    if print_stat: print(f'gps_alt: {gps_alt}, gps_lat:{gps_lat}, gps_long:{gps_long}')
    


def rel_alt_callback(altrel):
    global gps_alt_rel
    gps_alt_rel = altrel.data # relative altitude just from GPS data
    if print_stat: print(f"----------------Inside rel_alt_callback():---------------\ngps_alt_rel: {gps_alt_rel}")



def boundingbox_callback(box):
    global horizontalerror, verticalerror, sizeerror
    global time_lastbox, pitchcommand, yawcommand
    global MOVE_ABOVE, OPT_FLOW
    # positive errors give right, up
    if box.bbox.center.x != -1:
        if print_stat: print("----------------Inside boundingbox_callback():----------------\nBounding box found : box.bbox.center.x != -1")
        time_lastbox = rospy.Time.now()
        bboxsize = (box.bbox.size_x + box.bbox.size_y)/2 # take the average so that a very thin, long box will still be seen as small
        
        if not above_object: # different bbox size desired for approach and above stages for hybrid mode
            if print_stat: print("Not above object ... ")
            sizeerror = setpoint_size_approach - bboxsize # if box is smaller than setpoit, error is positive
        else:
            if print_stat: print("above_object was set to true previously")
            sizeerror = setpoint_size - bboxsize # if box is smaller than setpoit, error is positive

        if print_size_error: print('Setpoint - bbox (sizeerror) = %f' % sizeerror)

        if not OPT_FLOW:
            if print_stat: print("OPT_FLOW is set to False: Optical flow not running")   
            horizontalerror = .5-box.bbox.center.x # if box center is on LHS of image, error is positive
            verticalerror = .5-box.bbox.center.y # if box center is on upper half of image, error is positive 
            if print_stat: print(f'Horzontal error: {round(horizontalerror, 5)}, Vertical error: {round(verticalerror, 5)}')

            '''
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
            '''

            if print_stat: print(f"Changed pitchcommand,delta: {round(pitchcommand, 5)},{round(pitchdelta, 5)}  changed yawcommand, delta: {round(yawcommand, 5)},{round(yawdelta, 5)}")

        if pitchcommand < pitch_thresh and bboxsize > 0.75: # if close and gimbal pitched upward, move to get above the object
            MOVE_ABOVE = True
            if print_stat: print('Gimbal pitched upward moving above to get above object ... : MOVE_ABOVE is set to True')
    else:
        if print_stat: print("----------------Inside boundingbox_callback():----------------\nBounding box not found : box.bbox.center.x = -1")
   
    return



def segmentation_callback(box):
    global horizontalerror_smoketrack, verticalerror_smoketrack, sizeerror_smoketrack, horizontalerror_smoketrack_list
    global time_lastbox_smoketrack, pitchcommand, yawcommand
    global MOVE_ABOVE, OPT_FLOW
    global kf, previous_yaw_measurements
    global sampling
    
    # positive errors give right, up
    if box.bbox.center.x != -1 and box.bbox.size_x > 7000:
        if print_stat_test: 
            if sampling: print("Sampling ... Yawing using Segmentation")
        time_lastbox_smoketrack = rospy.Time.now()
        bboxsize = (box.bbox.size_x + box.bbox.size_y)/2 # take the average so that a very thin, long box will still be seen as small
        
        if not above_object: # different bbox size desired for approach and above stages for hybrid mode
            if print_stat: print("Not above object ... ")
            sizeerror_smoketrack = 0 # setpoint_size_approach - bboxsize # if box is smaller than setpoit, error is positive
        else:
            if print_stat: print("above_object was set to true previously")
            sizeerror_smoketrack = 0 # setpoint_size - bboxsize # if box is smaller than setpoit, error is positive

        if print_size_error: print('Setpoint - bbox (sizeerror) = %f' % sizeerror_smoketrack)

        if not OPT_FLOW:
            if print_stat: print("OPT_FLOW is set to False: Optical flow not running")   
            horizontalerror_smoketrack = .5-box.bbox.center.x # if box center is on LHS of image, error is positive
            verticalerror_smoketrack = .5-box.bbox.center.y # if box center is on upper half of image, error is positive 
            
            # Update the Kalman filter with the new measurement
            update_kalman_filter(kf, horizontalerror_smoketrack)
            previous_yaw_measurements.append(horizontalerror_smoketrack)
            
            if print_stat: print(f'Horzontal error: {round(horizontalerror_smoketrack, 5)}, Vertical error: {round(verticalerror, 5)}')
            # if print_stat: print(f"Changed pitchcommand,delta: {round(pitchcommand, 5)},{round(pitchdelta, 5)}  changed yawcommand, delta: {round(yawcommand, 5)},{round(yawdelta, 5)}")

        if pitchcommand < pitch_thresh and bboxsize > 0.75: # if close and gimbal pitched upward, move to get above the object
            MOVE_ABOVE = True
            if print_stat: print('Gimbal pitched upward moving above to get above object ... : MOVE_ABOVE is set to True')

    else:
        if print_stat_test: 
            if sampling: print("Sampling ... Yawing using Kalman Filter")
        horizontalerror_smoketrack = kf.x[0, 0] 

    return



def flow_callback(flow):
    global horizontalerror, verticalerror,time_lastbox
    global pitchcommand, yawcommand
    global flow_x,flow_y,flow_t
    global OPT_FLOW,OPT_COMPUTE_FLAG
    
    # typical values might be around 10 pixels, depending on frame rate
    flow_x = flow.size_x # movement to the right in the image is positive
    flow_y = -flow.size_y # this is made negative so that positive flow_y means the object was moving toward the top of the image (RAFT returns negative y for this (i.e., toward smaller y coordinates))
    # now movement to the top is positive flow_y
    flow_t = float(gps_t)
    # adjust the feedback error using the optical flow
    if OPT_FLOW:
        if print_stat: print('----------------Inside flow_callback():----------------\nDoing optical flow feedback ...')
        OPT_COMPUTE_FLAG = True # this signals to later in the code that the first usable optical flow data can be pplied (instead of inheriting errors from bbox callback)
        horizontalerror = -flow_x # to be consistent with the bounding box error
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
            if print_stat: print('Flow x,y = %f,%f' % (flow_x,flow_y))
            if print_stat: print(f'horizontal error: {horizontalerror}, verticalerror: {verticalerror}')
        
        time_lastbox = rospy.Time.now()
    else:
        if print_stat: print("----------------Inside flow_callback():----------------\nOPT_FLOW was turned off: OPT_FLOW = False")
    return



def _build_global_setpoint2(latitude, longitude, altitude, yaw=0.0):
    """Builds a message for the /mavros/setpoint_position/global topic
    """
    geo_pose_setpoint = GeoPoseStamped()
    geo_pose_setpoint.header.stamp = rospy.Time.now()
    geo_pose_setpoint.pose.position.latitude = latitude
    geo_pose_setpoint.pose.position.longitude = longitude
    geo_pose_setpoint.pose.position.altitude = altitude

    
    roll = 0.0
    pitch = 0.0
    q_x, q_y, q_z, q_w= euler_to_quaternion(roll, pitch, yaw)
    geo_pose_setpoint.pose.orientation.x = q_x
    geo_pose_setpoint.pose.orientation.y = q_y
    geo_pose_setpoint.pose.orientation.z = q_z
    geo_pose_setpoint.pose.orientation.w = q_w
    

    return geo_pose_setpoint



def build_local_setpoint(x, y, z, yaw):
    """
    Builds a message for the /mavros/setpoint_position/local topic
    """
    local_pose_setpoint = PoseStamped()
    local_pose_setpoint.header.stamp = rospy.Time.now()
    local_pose_setpoint.pose.position.x = x
    local_pose_setpoint.pose.position.y = y
    local_pose_setpoint.pose.position.z = z
    
    roll = 0.0
    pitch = 0.0
    q_x, q_y, q_z, q_w= euler_to_quaternion(roll, pitch, yaw)
    local_pose_setpoint.pose.orientation.x = q_x
    local_pose_setpoint.pose.orientation.y = q_y
    local_pose_setpoint.pose.orientation.z = q_z
    local_pose_setpoint.pose.orientation.w = q_w

    return local_pose_setpoint



def build_setpoint_attitude_cmd_vel(x_vel, y_vel, z_vel, x_ang, y_ang, z_ang):
    """
    Builds a message for the /mavros/setpoint_attitude/cmd_vel topic
    not getting
    """
    attitude_setpoint = TwistStamped()
    attitude_setpoint.header.stamp = rospy.Time.now()
    attitude_setpoint.twist.linear.x = float(x_vel)
    attitude_setpoint.twist.linear.y = float(y_vel)
    attitude_setpoint.twist.linear.z = float(z_vel)
    attitude_setpoint.twist.angular.x = float(x_ang)
    attitude_setpoint.twist.angular.y = float(y_ang)
    attitude_setpoint.twist.angular.z = float(z_ang)

    return attitude_setpoint



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
    global smoketrack_pub, sample_along_heading
    global sampling_t0, track_sampling_time
    global fspeed_head, hspeed_head
    global kf, previous_yaw_measurements
    global head, source_gps
    global setpoint_global_pub

    publish_rate = time.time()

    

    #Initialize publishers/subscribers/node
    rospy.Subscriber('/bounding_box', Detection2D, boundingbox_callback)
    rospy.Subscriber('/segmentation_box', Detection2D, segmentation_callback)
    rospy.Subscriber('/mavros/state',State,state_callback)
    rospy.Subscriber('/mavros/local_position/pose', PoseStamped, pose_callback)
    rospy.Subscriber('/mavros/time_reference',TimeReference,time_callback)
    rospy.Subscriber('/mavros/global_position/global',NavSatFix,gps_callback)
    rospy.Subscriber('/mavros/global_position/rel_alt',Float64,rel_alt_callback)
    rospy.Subscriber('/flow',BoundingBox2D,flow_callback)
    rospy.Subscriber('/mavros/state',State,state_callback)
    rospy.Subscriber('/mavros/global_position/compass_hdg',Float64,compass_hdg_callback)
    twistpub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=1)
    
    # setpoint_local_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=1)
    setpoint_global_pub = rospy.Publisher('/mavros/setpoint_position/global', GeoPoseStamped, queue_size=1) # used for yaw
    # setpoint_attitude_pub = rospy.Publisher('/mavros/setpoint_attitude/cmd_vel', TwistStamped, queue_size=1)

    rcpub = rospy.Publisher('/mavros/rc/override', OverrideRCIn, queue_size=1)
    gimbal = rospy.Publisher('/airsim_node/gimbal_angle_euler_cmd', GimbalAngleEulerCmd, queue_size=1)
    smoketrack_pub = rospy.Publisher('/smoketrack', String, queue_size=1)

    # control loop
    twistmsg = Twist()
    rcmsg = OverrideRCIn()
    rcmsg.channels = np.zeros(18,dtype=np.uint16).tolist()

    rate = rospy.Rate(20) # 20hz - originally 20hz

    
    
    while not rospy.is_shutdown():

        if forward_scan_option and forward_scan:
            # in this mode, the drone will just start moving forward until it sees the smoke below it
            fspeed = fscan_speed
            vspeed = 0
            hspeed = 0
            if print_stat: print('Forward_scan is on and Scanning for object ...')
            if print_flags: print('forward_scan_option = True | forward_scan = True')
        else:
            if print_stat: print("Forward scan turned off as scanning for object was completed")
            if print_stat: print('Either forward_scan_option = False or forward_scan = False')
        
        #feedback control algorithm
        #don't publish if message is old
        if not guided_mode:
            if print_stat: print("NOT GUIDED ...")
            pitchcommand = pitch_init 
            yawcommand = yaw_center
            hspeed = vspeed = 0
            yaw_mode = True # Previously True # turn yaw back on
            above_object = False
            moving_to_set_alt = False
            OPT_FLOW = False # turn off the optical flow mode
            if print_flags: print('yaw_mode = True | above_object = False | moving_to_set_alt = False')
            OPT_COMPUTE_FLAG = False
            if forward_scan_option:
                # turn this initial mode back on
                forward_scan = True
                if print_flags: print('forward_scan = True')
        elif (time_lastbox != None and (rospy.Time.now() - time_lastbox < rospy.Duration(.5)) and not moving_to_set_alt and not sample_along_heading):
            if print_stat: print("Inside : rospy.Time.now() - time_lastbox < rospy.Duration(.5) and not moving_to_set_alt ...")
            
            # safeguard for vertical motion
            if alt < alt_min:
                rise_up(dz = 2,vz=0.5) #ORIGINAL

            # end the forward scan phase once in air looking down and recognizing a particle
            # only do this if this option is turned on
            if forward_scan_option and forward_scan:
                if above_object and alt > alt_flow and pitchcommand > pitch_thresh:
                    forward_scan = False # setting this to false ensures that it wont be triggered again at the next pass
                    fspeed = 0
                    if print_stat: print("forward_scan turned off and fspeed set to zero")
                    if print_flags: print('forward_scan = False')
                    
            # to try and get the drone to move up if the smoke plume is only in front of it
            if MOVE_ABOVE:
                rise_up(dz=30,vz = 3) #previously : rise_up(dz=3,vz = 1)
                MOVE_ABOVE = False
                continue
    
            if hybrid_mode:
                # determine if above object based on pitch
                # if in a forward scan, then this will be triggered immediately when object is recognized, since pitch is already down
                if not above_object: # once entered above mode, cannot go back -  if this isn't done, then the transition between modes is really rough
                    if print_stat: print('Hybrid mode: Approach phase')
                    if pitchcommand > pitch_thresh and alt > alt_min:
                        above_object=True
                        if print_stat: print('#------------------- ABOVE OBJECT -------------------#')
                        if print_flags: print('above_object = True | USE_PITCH_ERROR = False')
                        USE_PITCH_ERROR = False # turning this off once moving downward     
                    else:
                        above_object=False
                        if print_stat: print('Not above object')
                        if print_flags: print('above_object = False')
                else: 
                    if print_stat: print('Hybrid mode: Above object: Continue ...')
                        
                #------FOR DEBUGGING------#
                if debugging:
                    above_object=True
                #-------------------------#

                if above_object and fixed_heading_option and not moving_to_set_alt: # only do this if not already moving to setpoint                    
                    if print_flags: print("above_object = True | fixed_heading_option = True | moving_to_set_altitude = False")
                    fspeed_surv,hspeed_surv = survey_flow()
                    fspeed_head = fspeed_surv
                    hspeed_head = hspeed_surv
                    if print_flags: print("moving_to_set_alt = True") 
                    if print_stat: print("End of survey")
                    moving_to_set_alt = True # reporting end of survey
                    fspeed = hspeed = vspeed = 0
                    continue

                hspeed = -horizontalerror * traverse_gain
                # forward movement   (fspeed > 0 move forward)
                if above_object: # top-view, determining forward movement based on
                    if print_stat: print('As above_object - Inside fspeed correction ...') 
                    fspeed = verticalerror * traverse_gain      
                    if print_stat: print(f'fspeed: {fspeed}, traverse_gain: {traverse_gain}, verticalerror: {verticalerror}')       
                else: # side-view, determining approach speed based on size of bounding box
                    fspeed = sizeerror * size_gain
                    if print_stat: print('As not above_object - Inside fspeed correction ...') 
                    if print_stat: print(f'fspeed: {fspeed}, size_gain: {size_gain}, sizeerror: {sizeerror}')
                    if USE_PITCH_ERROR:
                        # slow down the forward speed when the gimbal starts pitching down
                        fspeed_adjust = (pitch_down - pitchcommand)/(pitch_down - pitch_up)
                        fspeed_adjust = min(max(fspeed_adjust,1),pitcherror_gain_min)
                        fspeed *= fspeed_adjust
                
                # vertical movement depending on the minimum altitude safeguard
                if above_object: 
                    if not fixed_heading_option: vspeed = -sizeerror * vertical_gain # size error is negative because you want to move down (negative velocity) to get closer  
                else: vspeed=0

                # assigning gimbal pitch and yaw depending on mode
                if print_vspeed: print('Vertical speed: %f' % vspeed)
                if above_object:
                    yawrate = 0 # don't rotate/yaw the drone if in top-down mode 
                    yaw_mode = False # previously false originally
                    if print_flags: print('yaw_mode = False')
                    if print_stat: print(f'above_object = True and yawrate: {yawrate}')                    
                    pitchcommand = pitch_down
                    yawcommand = yaw_center
                else:
                    yawrate = ((yawcommand - yaw_center)/1000)*yaw_gain
                    if print_stat: print(f'above_object = False and yawrate: {yawrate}')
            else:
                pass
        elif moving_to_set_alt and not sample_along_heading:
            # if its been more than half a second without detection during descent, stop lateral movement
            if print_stat: print("Inside : moving_to_set_alt and not sample_along_heading ...")
            if print_stat: print("Its been more than half a second without detection during descent, stopping lateral movement")
            hspeed = -(horizontalerror) * traverse_gain * 2
            fspeed = (verticalerror-0.15) * traverse_gain  # (verticalerror-0.10) to make the drone come down close to smoke source
            if print_stat: print(f'fspeed: {fspeed}, traverse_gain: {traverse_gain}, verticalerror: {verticalerror}') 
        elif time_lastbox != None and (rospy.Time.now() - time_lastbox > rospy.Duration(5)) and not moving_to_set_alt and not sample_along_heading: # added condition here so that even if smoke isn't seen, descent continues after survey
            # if nothing detected for 5 seconds, reset gimbal position, and if more than 10 seconds, go back to manual control from RC
            # also reinitializes other settings
            print('#--------------RESETTING....------------#')
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
            # print("Duration check = ", rospy.Time.now() - time_lastbox)
            if (rospy.Time.now() - time_lastbox < rospy.Duration(10)):
                rcmsg.channels[7] = int(pitchcommand) #send pitch command on channel 8
                rcmsg.channels[6] = int(yawcommand) #send yaw command on channel 7


        # out of loop, send commands       
        # check if altitude setpoint reached
        if fixed_heading_option and moving_to_set_alt:
            alt_diff = alt - alt_sampling
            if debugging: alt_diff = 0
            if print_flags: print(f'fixed_heading_option = True | moving_to_set_altitude = True')

            if abs(alt_diff) < 0.7:
                if print_stat: print(f'Reached setpoint alt at {alt} m')
                vspeed = 0 # desired alttitude reached
                moving_to_set_alt = False
                sample_along_heading = True
                source_gps[0], source_gps[1], source_gps[2] = gps_lat, gps_long, gps_alt
                print(f'source_gps: {source_gps}')
                if print_flags : print('moving_to_set_alt = False | sample_along_heading = True')
            elif alt_diff < 0: # too low
                vspeed = abs(alt_set_appr_speed) # force to be positive
                if print_stat: print(f'Too Low ... Moving to setpoint alt at {vspeed} m/s')
            elif alt_diff > 0: # too high
                vspeed = -abs(alt_set_appr_speed) # force to be negative (move down)
                smoketrack_pub.publish('Smoke Tracking On')
                if print_stat: print(f'Too high ... Moving to setpoint alt at {vspeed} m/s')


        if sample_along_heading: #and not moving_to_set_alt: 
            if print_stat: print('Sample along heading test')
            forward_scan = False
            above_object = False
            if print_flags: print('forward_scan = False | above_object = False')
            if track_sampling_time: sampling_t0 = time.time()
            sample_heading_test(-fspeed_head,-hspeed_head)


        #bound controls to ranges
        if print_stat: print(f'fspeed: {fspeed}, vspeed: {vspeed}, hspeed:{hspeed}')
        fspeed = min(max(fspeed,-limit_speed),limit_speed) #lower bound first, upper bound second
        hspeed = min(max(hspeed,-limit_speed),limit_speed)
        vspeed = min(max(vspeed,-limit_speed_v),limit_speed_v) # vertical speed
        yawrate = min(max(yawrate,-limit_yawrate),limit_yawrate)
        yawcommand = min(max(yawcommand,1000),2000)
        pitchcommand = min(max(pitchcommand,1000),2000)
        if print_stat: print(f'After boundings: fspeed: {fspeed}, vspeed: {vspeed}, hspeed:{hspeed}')
        if print_stat: print(f'fspeed, hspeed, vspeed: {fspeed}, {hspeed}, {vspeed}\nyawrate, yawcommand, pitchcommand: {yawrate}, {yawcommand}, {pitchcommand}')

        # horizontal motion
        if yaw_mode:
            if print_stat: print(f'yaw_mode was set to True: yaw:{yaw}')
            twistmsg.linear.x = math.cos(yaw)*fspeed + math.sin(yaw)*hspeed
            twistmsg.linear.y = math.sin(yaw)*fspeed - math.cos(yaw)*hspeed
            twistmsg.angular.z = 0 # previously twistmsg.angular.z = yawrate
        else:
            if print_stat: print(f'yaw_mode was set to False: yaw:{yaw}')
            twistmsg.linear.x = math.cos(yaw)*fspeed + math.sin(yaw)*hspeed
            twistmsg.linear.y = math.sin(yaw)*fspeed - math.cos(yaw)*hspeed
            twistmsg.angular.z = 0
        
        # publishing
        if not sample_along_heading:
            twistmsg.linear.z = vspeed  # vertical motion
            rcmsg.channels[7] = int(pitchcommand) #send pitch command on channel 8
            rcmsg.channels[6] = int(yawcommand) #send yaw command on channel 7
            if print_stat: print(f'----------------Publishing Commands----------------')
            if print_stat: print(f'Pitch Command -> {pitchcommand}, Yaw Command -> {yawcommand}')
            if print_stat: print("Publishing control cmd after", time.time() - publish_rate, "seconds")
            if print_stat: print(f'x_speed: {twistmsg.linear.x}, y_speed: {twistmsg.linear.y}, z_speed: {twistmsg.linear.z}')
            twistpub.publish(twistmsg)
            moveAirsimGimbal(pitchcommand, yawcommand)
            publish_rate = time.time()

        if print_pitch: print('Pitch command: %f' % (pitchcommand))
        if print_yawrate: print('Yaw rate: %f' % yawrate)
        if print_alt: print(f"Altitude: {alt} m")
        if print_speeds: print(f'fspeed: {round(fspeed, 5)}, hspeed: {round(hspeed, 5)}, vspeed: {vspeed}')
        
        # writing control states and data to csv
        save_log()
        
        rate.sleep()

    '''
    yaw = 100
    while not rospy.is_shutdown():
        setpoint = _build_global_setpoint2(gps_lat, gps_long, gps_alt+5, yaw) # yaw = 0 is heading 90 in airsim
        setpoint_global_pub.publish(setpoint)
        #print('Inside Lopp')
        #setpoint_raw = build_local_setpoint(50, 0, 10, yaw)
        #setpoint_local_pub.publish(setpoint_raw)
        #print(f'Publishing Command', end='\r')
        #setpoint_raw2 = build_local_setpoint(-50, 0, 0, 0)
        #setpoint_local_pub.publish(setpoint_raw2)
        #setpoint_attitude_cmd_vel = build_setpoint_attitude_cmd_vel(10, 0, 0, 0, 0, 0)
        #setpoint_attitude_pub.publish(setpoint_attitude_cmd_vel)
    '''
    
    
    

def rise_up(dz = 5,vz=3):
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
    global yaw
    print('#-------------------Beginning survey---------------------#')

    survey_samples = 10 # originally 10
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
    surveying = True
    while True: # collect samples until a certain number reached, and for at least 5 seconds
        if not guided_mode:
            return 0,0

        if len(vx) >= survey_samples and (time.time() - t0 > 10): # do this at least 5 seconds duration
            break

        if flow_t > t1 and flow_x != flow_prev: # only use flow values after this sequence starts
                if ~np.isnan(flow_x):   # only use if not nan
                    vx.append(flow_x*flow_survey_gain*(alt-3))
                    vy.append(flow_y*flow_survey_gain*(alt-3)) # gain is attenuated or amplified by altitude                   
                flow_prev = flow_x
                    
        if time.time()-t0 > 30: # only do this for 15 seconds at most
            print('#-------------------Failed to determine heading-------------------#')
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
    print('#-----------------------Got heading-----------------------#')
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
    print('#----------------------Survey complete------------------------#')
    surveying = False

    #------------------------------srijan--------------------------------#
    print(f'yaw: {yaw}')
    i = 0
    twistmsg.linear.x = 0
    twistmsg.linear.y = 0
    twistmsg.linear.z = 0
    yaw_speed = 0.2
    precision = 0.009 # previously 0.005
    low_yawspeed = False
    if hspeed_surv > 0:
        yaw_speed = (- 0.2)

    dir_surv = math.atan2(fspeed_surv, hspeed_surv) + (3.141592653589793/2)
    print(f'dir_surv: {dir_surv}')
    twistmsg.angular.z = yaw_speed

    while True:
        i = i + 1
        formated_yawspeed = "{:.5f}".format(yaw_speed)
        if i%50000==0 and print_stat: print(f'Yaw: {yaw}, dirSurv: {dir_surv}: yawspeed: {formated_yawspeed}')
        twistpub.publish(twistmsg)
        if low_yawspeed and (yaw <= dir_surv+precision and yaw >= dir_surv-precision):
            break
        elif not low_yawspeed and (yaw <= dir_surv+0.5 and yaw > dir_surv+precision):
            yaw_speed = yaw_speed/5
            low_yawspeed = True
        elif not low_yawspeed and (yaw < dir_surv-precision and yaw >= dir_surv-0.5):
            yaw_speed = yaw_speed/5
            low_yawspeed = True
        else:
            yaw_speed = yaw_speed
        twistmsg.angular.z = yaw_speed

    print(f'Yaw Complete: yaw - {yaw}')

    twistmsg.linear.x = 0
    twistmsg.linear.y = 0
    twistmsg.linear.z = 0
    twistmsg.angular.z = 0
    twistpub.publish(twistmsg)
    #------------------------------srijan--------------------------------#

    return fspeed_surv,hspeed_surv



def update_kalman_filter(kf, measurement):
    """
    Function to update Kalman filter with a new measurement
    """
    kf.predict()
    kf.update(measurement)



def sample_heading_test(fspeed_head,hspeed_head):
    """
    keeps fixed altitude and moves along a prescribed direction obtain from flow survey prior
    """
    global twistpub, twistmsg,rcmsg,rcpub
    # function for setting the flow direction obtained after surveying
    global sampling, sampling_time, sampling_t0, track_sampling_time
    global sample_along_heading
    global fspeed,hspeed,vspeed
    global horizontalerror, traverse_gain
    global horizontalerror_smoketrack, verticalerror_smoketrack, time_lastbox_smoketrack
    global smoketrack_pub
    global yawrate
    global print_iter

    sampling = True
    if print_flags: print('sampling = True')
    reverse_kalman = False

    print_iter = print_iter + 1

    x_speed = (math.cos(yaw)*fspeed_head + math.sin(yaw)*hspeed_head)*(0.25)
    y_speed = (math.sin(yaw)*fspeed_head - math.cos(yaw)*hspeed_head)*(0.25)

    # moving away from the source of the smoke
    if (time_lastbox_smoketrack != None and (rospy.Time.now() - time_lastbox_smoketrack < rospy.Duration(7.5))):
        z_angular = horizontalerror_smoketrack * (yawrate/5)
        z_speed = verticalerror_smoketrack * (3)
    else:
        print("Sampling ... Yawing using Reverse Kalman Filter")
        z_angular = - (kf.x[0, 0]) * (yawrate/2)
        z_speed = 0 
        reverse_kalman = True

    twistmsg.linear.z = z_speed
    twistmsg.angular.z = z_angular
    moveAirsimGimbal(1000, 1500)
    
    if time.time()-sampling_t0 < sampling_time:
        track_sampling_time = False
        if print_flags: print('track_sampling_time = False')
        twistpub.publish(twistmsg)
        rcpub.publish(rcmsg)
        rcpub.publish(rcmsg)
        if not reverse_kalman:
            x_speed = (math.cos(yaw)*fspeed_head + math.sin(yaw)*hspeed_head)*(0.25)
            y_speed = (math.sin(yaw)*fspeed_head - math.cos(yaw)*hspeed_head)*(0.25)
        else:
            x_speed = 0
            y_speed = 0
        twistmsg.linear.x = x_speed
        twistmsg.linear.y = y_speed
        twistpub.publish(twistmsg)
        rcpub.publish(rcmsg)
        rcpub.publish(rcmsg)
    elif time.time()-sampling_t0 >= sampling_time:
        track_sampling_time = True
        sample_along_heading = False
        if print_flags: print('track_sampling_time = True | sample_along_heading = False')
        print('Sampling Complete')        

    return



'''
def sample_heading_test(fspeed_head,hspeed_head):
    """
    keeps fixed altitude and moves along a prescribed direction obtain from flow survey prior
    """
    global twistpub, twistmsg,rcmsg,rcpub
    # function for setting the flow direction obtained after surveying
    global sampling, sampling_time, sampling_t0, track_sampling_time
    global sample_along_heading
    global fspeed,hspeed,vspeed
    global horizontalerror, traverse_gain
    global horizontalerror_smoketrack, verticalerror_smoketrack, time_lastbox_smoketrack
    global smoketrack_pub
    global yawrate
    global print_iter
    
    sampling = True
    if print_flags: print('sampling = True')
    reverse_kalman = False

    print_iter = print_iter + 1

    # start moving away from the source of the smoke
    if (time_lastbox_smoketrack != None and (rospy.Time.now() - time_lastbox_smoketrack < rospy.Duration(7.5))):
        fspeed_head = fspeed_head # fspeed_head determined by optical flow
        hspeed_head = hspeed_head + horizontalerror_smoketrack * (-15) # hspeed_head determined by optical flow and horizontalerror_smoketrack is determined by the segmentation/kalman filter
        z_speed = verticalerror_smoketrack * (3) # verticalerror_smoketrack is determined by segmentation only
        z_angular = horizontalerror_smoketrack * (yawrate/10)
        # z_angular = 0 # turning this off as the yaw motion is controlled by smoketrack_yaw()
        # previously z_angular = horizontalerror_smoketrack * (yawrate/5)        
    else:
        if print_stat: print("Sampling ... Yawing using Reverse Kalman Filter")
        hspeed_head = hspeed_head - (kf.x[0, 0]) * (5)
        fspeed_head = fspeed_head
        z_angular = - (kf.x[0, 0]) * (yawrate/2) # previously z_angular = - (kf.x[0, 0]) * (yawrate/2)
        z_speed = 0 
        reverse_kalman = True

    x_speed = (math.cos(yaw)*fspeed_head + math.sin(yaw)*hspeed_head)*(0.25)
    y_speed = (math.sin(yaw)*fspeed_head - math.cos(yaw)*hspeed_head)*(0.25)

    twistmsg.linear.x = x_speed
    twistmsg.linear.y = y_speed
    twistmsg.linear.z = z_speed
    twistmsg.angular.z = z_angular
    moveAirsimGimbal(1000, 1500)

    if time.time()-sampling_t0 < sampling_time:
        track_sampling_time = False
        if print_flags: print('track_sampling_time = False')
        twistpub.publish(twistmsg)
        #smoketrack_yaw()
        rcpub.publish(rcmsg)
        rcpub.publish(rcmsg)
    elif time.time()-sampling_t0 >= sampling_time:
        track_sampling_time = True
        sample_along_heading = False
        if print_flags: print('track_sampling_time = True | sample_along_heading = False')
        print('Sampling Complete')        

    return
    '''


def save_log():
    """
    writing data to csv
    """
    fid.write('%f,%f,%f,%f,%f,%f,%f,%f,%s,%s,%s,%s,%f,%f,%s,%f,%f,%f,%f,%f\n' % 
        (time.time(),gps_t,gps_x,gps_y,alt,gps_lat,gps_long,gps_alt_rel,str(surveying),str(sampling),str(move_up),str(above_object),pitchcommand,sizeerror,str(OPT_FLOW),flow_x,flow_y,vspeed,fspeed,hspeed))



if __name__ == '__main__':
    print("Initializing feedback node...")
    rospy.init_node('feedbacknode', anonymous=False)
    twist_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=1)
    
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
    arming = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
    resp = arming(arm)
    print(resp)
    time.sleep(5)
    print("Arming - 2nd attempt", end ="->")
    resp = arming(arm)
    print(resp)
    time.sleep(5)

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
    
    
    for i in range(150):
        #Start poition (X=-66495.023860,Y=49467.376329,Z=868.248719)
        # Move to set posiiton
        if (i+1)%10 == 0: print(f"Moving...{i+1}")
        go = PoseStamped()
        go.pose.position.x = -17
        go.pose.position.y = 10
        go.pose.position.z = 250 # previuosly 250 with 55 fov of AirSim Cam
        go.pose.orientation.z = 1
        twist_pub.publish(go)
        time.sleep(0.2)
    

    print("GOING AUTONOMOUS")
    time.sleep(5)
    try:
        dofeedbackcontrol()
    except rospy.ROSInterruptException:
        pass


