#!/usr/bin/python3
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
from filterpy.kalman import KalmanFilter
from geopy.distance import geodesic

# global EXECUTION
EXECUTION = rospy.get_param('EXECUTION', default='DEPLOYMENT') # 'SIMULATION' or 'DEPLOYMENT'

if EXECUTION == 'SIMULATION':
    import airsim
    from airsim_ros_pkgs.msg import GimbalAngleEulerCmd, GPSYaw



#----------------------- OPTIONS FOR MODES (Nate) -----------------------#
OPT_FLOW_MASTER = True # true means optical flow is used
top_down_mode = False  # different operating mode for motion if the drone is far above the feature
hybrid_mode = True # starts horizontal, moves toward object, then once above it, moves into the top-down-mode
yaw_mode = True # Previously True # whether or not to yaw the entire drone during motion
forward_scan_option = True # Previously False changes to True this is primarily for smoke generator and grenade experiments, otherwise hybrid mode should work well over a more static plume
fixed_heading_option = True # this mode tells the drone to sit above the smoke for a solid 5 seconds in order to get a heading from the mean flow direction, and then it goes to a fixed height and moves in that direction
controlled_descent_option = True
#----------------------- OPTIONS FOR MODES (Nate) -----------------------#




#----------------------- PRINT OPTIONS ----------------------------#
print_pitch = False
print_size_error = False
print_mode = False
print_vspeed = False
print_yawrate = False
print_flow=False
print_alt = False
#-----------------------(srijan)----------------------------#
print_stat = False
print_stat_test = True
print_flags = True
print_state = True
print_speeds = False
#----------------------- PRINT OPTIONS ----------------------------#



#----------------------- Global Parameters (srijan) ----------------------------#
global fspeed_head, hspeed_head
global source_gps_track
global horizontalerror_bbox, verticalerror_bbox, sizeerror_bbox
global horizontalerror_keypoint, verticalerror_keypoint
global head, slope_deg
global setpoint_global_pub
global proportional_gain
#----------------------- Global Parameters (srijan) ----------------------------#



#----------------------- Initialization (srijan) ----------------------------#
smoke_dir = ''
head, slope_deg = 0.0, 0.0
source_gps = [0.0, 0.0, 0.0] # lattitude, longitude and altitude
drone_gps = [0.0, 0.0, 0.0] # lattitude, longitude and altitude
time_lastbox = None
#----------------------- Initialization (srijan) ---------------------------- #



# ---------------------- Simulation Parameters (srijan) ------------------------------ #
sampling_time = 250
proportional_gain = 8
# ---------------------- Simulation Parameters (srijan) ------------------------------ #



# ---------------------------------------------------- Deployment Parameters (Nate) ---------------------------------------------------- #
# bounding box options
setpoint_size = 0.9 #fraction of frame that should be filled by target. Largest axis (height or width) used.
setpoint_size_approach = 1.5 # only relevant for hybrid mode, for getting close to target

# optical flow parameters
alt_flow = 264 # altitude at which to stop descent and keep constant for optical flow
alt_sampling = 264 # 1.5 # altitude setpoint at which to do a controlled sampling based on mean flow direction
alt_min = 260 # minimum allowable altitude

# gain values
size_gain = 1
yaw_gain = 1

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
fscan_speed = 0.3
sizeerror_flow_thresh = 0.1 # this is the fraction of the setpoint size that the error needs to be within in order to initiate optical flow

# initialize
guided_mode = False
horizontalerror_bbox = verticalerror_bbox = sizeerror_bbox = 0
horizontalerror_keypoint = verticalerror_keypoint = 0
vspeed = 0 # positive is upwards
hspeed = 0 # positive is to the right
fspeed = 0 # positive is forwards
flow_x = flow_y = flow_t = 0
yaw = 0
yawrate = 0 # previously 0 (not sue if this was the original value)
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
publish_rate = 0
# ---------------------------------------------------- Deployment Parameters (Nate) ---------------------------------------------------- #




# checked ok
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



# checked ok
def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion and 
    calculate quaternion components w, x, y, z
    """
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



# checked ok
def pose_callback(pose):
    global yaw, alt, gps_x, gps_y

    q = pose.pose.orientation
    r,p,y = euler_from_quaternion(q.x,q.y,q.z,q.w)
    yaw = y
    alt = pose.pose.position.z
    gps_x = pose.pose.position.x
    gps_y = pose.pose.position.y



# checked ok
def compass_hdg_callback(heading):
    """
    Returns drone's heading with respect to North
    """
    global head
    head = heading.data



# checked ok
def state_callback(state):
    """
    check if drone FCU is in LOITER or GUIDED mode
    """
    global guided_mode
    global print_state

    if state.mode == 'GUIDED':
        guided_mode = True
        if print_flags and print_state: 
            print("state.mode == 'GUIDED' -> guided_mode = True")
            print_state = False
    else:
        print("!!!!!!!!!!!!!!!! NOT OFFBOARD")
        guided_mode = False



# checked ok
def time_callback(gpstime):
    """
    returns the gps time
    """
    global gps_t
    gps_t = float(gpstime.time_ref.to_sec())



# checked ok
def gps_callback(gpsglobal):
    global gps_lat, gps_long, gps_alt, drone_gps

    gps_lat = gpsglobal.latitude
    gps_long = gpsglobal.longitude
    gps_alt = gpsglobal.altitude
    drone_gps[0], drone_gps[1], drone_gps[2] = gps_lat, gps_long, gps_alt


# checked ok
def rel_alt_callback(altrel):
    global gps_alt_rel
    gps_alt_rel = altrel.data # relative altitude just from GPS data



# NOT OK - MOVE_ABOVE_OBJECT removed not sure waht it was doing
def boundingbox_callback(box):
    global horizontalerror_bbox, verticalerror_bbox, sizeerror_bbox
    global time_lastbox
    global MOVE_ABOVE, OPT_FLOW

    # positive errors give right, up
    if box.bbox.center.x != -1:
        time_lastbox = rospy.Time.now()
        bboxsize = (box.bbox.size_x + box.bbox.size_y)/2 # take the average so that a very thin, long box will still be seen as small
        
        if not above_object: # different bbox size desired for approach and above stages for hybrid mode
            sizeerror_bbox = setpoint_size_approach - bboxsize # if box is smaller than setpoit, error is positive
        else:
            sizeerror_bbox = setpoint_size - bboxsize # if box is smaller than setpoit, error is positive

        if not OPT_FLOW: 
            horizontalerror_bbox = .5-box.bbox.center.x # if box center is on LHS of image, error is positive
            verticalerror_bbox = .5-box.bbox.center.y # if box center is on upper half of image, error is positive 

    return



# checked ok
def keypoint_callback(box):
    global horizontalerror_keypoint, verticalerror_keypoint
    global time_lastbox_keypoint
    global MOVE_ABOVE, OPT_FLOW
    
    # positive errors give right, up
    if box.bbox.center.x != -1:
        time_lastbox_keypoint = rospy.Time.now()  
        horizontalerror_keypoint = 0.5-box.bbox.center.x # if smoke source keypoint is on LHS of image, error is positive
        verticalerror_keypoint = 0.5-box.bbox.center.y # if smoke source keypoint is on upper half of image, error is positive 
    else:
        horizontalerror_keypoint = 0
        verticalerror_keypoint = 0
    
    return



# NOT OK - if not ABOVE_OBJECT commented not not sure why
def flow_callback(flow):
    global horizontalerror_bbox, verticalerror_bbox, time_lastbox
    global flow_x,flow_y,flow_t
    global OPT_FLOW,OPT_COMPUTE_FLAG
    
    # typical values might be around 10 pixels, depending on frame rate
    flow_x = flow.size_x # movement to the right in the image is positive
    flow_y = -flow.size_y # this is made negative so that positive flow_y means the object was moving toward the top of the image (RAFT returns negative y for this (i.e., toward smaller y coordinates))
    # now movement to the top is positive flow_y
    flow_t = float(gps_t)
    
    # adjust the feedback error using the optical flow
    if OPT_FLOW:
        OPT_COMPUTE_FLAG = True # this signals to later in the code that the first usable optical flow data can be pplied (instead of inheriting errors from bbox callback)
        horizontalerror_bbox = -flow_x # to be consistent with the bounding box error
        verticalerror_bbox = flow_y
        
        if print_flow:
            print('Flow x,y = %f,%f' % (flow_x,flow_y))
            print(f'horizontal error: {horizontalerror_bbox}, verticalerror: {verticalerror_bbox}')
        
        time_lastbox = rospy.Time.now()

    return



def dofeedbackcontrol():
    # ----------------------------- Nate ----------------------------- #
    global above_object, forward_scan
    global yaw_mode,OPT_FLOW,OPT_COMPUTE_FLAG,MOVE_ABOVE
    global move_up, USE_PITCH_ERROR
    global moving_to_set_alt
    global hspeed,vspeed,fspeed
    global yawrate
    global horizontalerror_bbox,verticalerror_bbox
    global twistpub, twistmsg
    # ----------------------------- Nate ----------------------------- #

    global gimbal
    global publish_rate
    global smoketrack_status_pub
    global fspeed_head, hspeed_head
    global kf, previous_yaw_measurements
    global head, source_gps, drone_gps
    global setpoint_global_pub
    global horizontalerror_keypoint, verticalerror_keypoint
    global yaw_pub, twist_stamped_pub
    global sampling_time_info_pub


    # ---------------------------------------------- Nate ---------------------------------------------- #
    # Initialize subscribers to mavros topics
    rospy.Subscriber('/mavros/state', State, state_callback)
    rospy.Subscriber('/mavros/local_position/pose', PoseStamped, pose_callback)
    rospy.Subscriber('/mavros/time_reference', TimeReference, time_callback)
    rospy.Subscriber('/mavros/global_position/global', NavSatFix, gps_callback)
    rospy.Subscriber('/mavros/global_position/rel_alt', Float64, rel_alt_callback)
    rospy.Subscriber('/mavros/global_position/compass_hdg', Float64, compass_hdg_callback)
    
    # Initialize subscribers to bounding box, flow, keypoints topics
    rospy.Subscriber('/bounding_box', Detection2D, boundingbox_callback)
    rospy.Subscriber('/flow', BoundingBox2D, flow_callback)
    rospy.Subscriber('/keypoints', Detection2D, keypoint_callback)
    
    # Initialize Publishers
    twistpub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=1)
    smoketrack_status_pub = rospy.Publisher('/smoketrack_status', String, queue_size=1)
    # ---------------------------------------------- Nate ---------------------------------------------- #
    
    

    global source_gps_track
    source_gps_track = True

    publish_rate = time.time()
    twistmsg = Twist()
    rate = rospy.Rate(20) # originally 10hz

    # feedback control loop
    while not rospy.is_shutdown():
        #smoketrack_status_pub.publish('Using Keypoints')
        if forward_scan_option and forward_scan: # both are true initially
            # in this mode, the drone will just start moving forward until it sees the smoke below it
            fspeed = fscan_speed
            vspeed = 0
            hspeed = 0

        #feedback control algorithm
        #don't publish if message is old
        
        if guided_mode:
            moving_to_set_alt = True
            smoketrack_status_pub.publish('Using Keypoints')

            hspeed = - (horizontalerror_keypoint) * (traverse_gain)
            fspeed = (verticalerror_keypoint) * (traverse_gain/1.5)
            alt_set_appr_speed = -0.5
            if print_stat_test: print(f'gps_alt: {gps_alt}, horizontalerror_keypoint: {horizontalerror_keypoint}, verticalerror_keypoint: {verticalerror_keypoint}')
            if print_stat_test: print(f'hspeed: {hspeed}, fspeed: {fspeed}')



        # out of loop, send commands       
        # check if altitude setpoint reached
        if fixed_heading_option and moving_to_set_alt:
            alt_diff = -(alt - alt_sampling)
            if print_flags: print(f'fixed_heading_option = True | moving_to_set_altitude = True')

            if abs(alt_diff) < 0.5:
                if print_stat_test: print(f'Reached setpoint alt at {alt} m')
                vspeed = 0 # desired alttitude reached
                moving_to_set_alt = False
                source_gps[0], source_gps[1], source_gps[2] = gps_lat, gps_long, gps_alt
                print(f'source_gps: {source_gps}')
                if print_flags : print('moving_to_set_alt = False')
            elif alt_diff < 0: # too low
                vspeed = abs(alt_set_appr_speed) # force to be positive
                if print_stat_test: print(f'Too Low ... Moving to setpoint alt at {vspeed} m/s')
            elif alt_diff > 0: # too high
                vspeed = -abs(alt_set_appr_speed) # force to be negative (move down)
                if print_stat_test: print(f'Too high ... Moving to setpoint alt at {vspeed} m/s')


        #bound controls to ranges
        fspeed = min(max(fspeed,-limit_speed),limit_speed) #lower bound first, upper bound second
        hspeed = min(max(hspeed,-limit_speed),limit_speed)
        vspeed = min(max(vspeed,-limit_speed_v),limit_speed_v) # vertical speed
        yawrate = min(max(yawrate,-limit_yawrate),limit_yawrate)

        twistmsg.linear.x = fspeed
        twistmsg.linear.y = hspeed
        twistmsg.linear.z = vspeed  
        twistmsg.angular.z = yawrate
        if print_stat: print(f'x_speed: {fspeed} | y_speed: {hspeed} | z_speed: {vspeed} | yaw_speed: {yawrate}')
        
        # publishing
        twistpub.publish(twistmsg)
        publish_rate = time.time()

        
        rate.sleep()
        


# NOT OK - although new PID controller can be used
def rise_up(dz = 5, vz=3):
    """
    simple loop to go up or down, usually to get above object
    """
    global twistpub, twistmsg
    
    print(f'Rising {dz}m at {vz}m/s...')
    
    twistmsg.linear.x = 0
    twistmsg.linear.y = 0
    twistmsg.linear.z = vz
    
    for i in range(int((dz/vz)/0.2)):
        twistpub.publish(twistmsg)
        time.sleep(0.2)
    
    twistmsg.linear.z = 0
    twistpub.publish(twistmsg)
    
    return



# checked ok
def survey_flow():
    """
    function for starting a survey of the flow from above, and then calling a heading to travel towards
    separated loop that keeps the drone fixed while observing the flow in the bounding box below
    """
    
    global surveying
    global twistpub, twistmsg
    global fspeed,hspeed,vspeed
    global yaw
    
    print('#-------------------Beginning survey---------------------#')

    survey_samples = 10 # originally 10
    survey_duration = 10 # hold position for 10 seconds
    
    twistmsg.linear.x = 0
    twistmsg.linear.y = 0
    twistmsg.linear.z = 0
    twistmsg.angular.z = 0 # stop any turning
    twistpub.publish(twistmsg)
    
    t1 = gps_t
    t0 = time.time()
    vx, vy = [], []

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
            twistpub.publish(twistmsg)

    fspeed_surv = np.nanmean(vy) # vertical velocity
    hspeed_surv = np.nanmean(vx)  # horizontal

    print('#-----------------------Got heading-----------------------#')
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

    #---------------------YAWING TOWARDS SMOKE FLOW---------------------------#
    print(f'yaw: {yaw}')
    
    i = 0
    
    twistmsg.linear.x = 0
    twistmsg.linear.y = 0
    twistmsg.linear.z = 0
    yaw_speed = 0.2
    precision = 0.1 # previously 0.009
    low_yawspeed = False
    
    if hspeed_surv > 0:
        yaw_speed = (- 0.2)

    dir_surv = math.atan2(fspeed_surv, hspeed_surv) + (3.141592653589793/2)
    print(f'dir_surv: {dir_surv}')
    twistmsg.angular.z = yaw_speed

    while True:
        i = i + 1
        formated_yawspeed = "{:.5f}".format(yaw_speed)
        
        if i%50000==0 and print_stat_test: print(f'Yaw: {yaw}, dirSurv: {dir_surv}: yawspeed: {formated_yawspeed}')
        
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
    #------------------------------YAWING COMPLETE--------------------------------#

    return fspeed_surv,hspeed_surv



if __name__ == '__main__':
    
    print("Initializing feedback node...")
    rospy.init_node('feedbackcontrol_node', anonymous=False)
    
    #global EXECUTION
    print(f'Executing in ==> {EXECUTION}')

    try:
        dofeedbackcontrol()
    except rospy.ROSInterruptException:
        pass