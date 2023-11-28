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
import argparse


# global EXECUTION
# EXECUTION = rospy.get_param('EXECUTION', default='DEPLOYMENT') # 'SIMULATION' or 'DEPLOYMENT'
EXECUTION = 'DEPLOYMENT'


#----------------------- OPTIONS FOR MODES (Nate) -----------------------#
OPT_FLOW_MASTER = True # true means optical flow is used
top_down_mode = False  # different operating mode for motion if the drone is far above the feature
hybrid_mode = True # starts horizontal, moves toward object, then once above it, moves into the top-down-mode
yaw_mode = True # Previously True # whether or not to yaw the entire drone during motion
USE_PITCH_ERROR = False # Previuosly True
forward_scan_option = True # Previously False changes to True this is primarily for smoke generator and grenade experiments, otherwise hybrid mode should work well over a more static plume
fixed_heading_option = True # this mode tells the drone to sit above the smoke for a solid 5 seconds in order to get a heading from the mean flow direction, and then it goes to a fixed height and moves in that direction
controlled_descent_option = True
#----------------------- OPTIONS FOR MODES (Nate) -----------------------#



# ---------------------------- Creating a saving directory ---------------------------- #
# create saving directory
tmp = datetime.datetime.now()
stamp = "%02d-%02d-%02d" %(tmp.year, tmp.month, tmp.day)

if EXECUTION == 'DEPLOYMENT':
    username = os.getlogin()
    maindir = Path(f'/home/{username}/1FeedbackControl')

runs_today = list(maindir.glob(f'*{stamp}*_fc-data'))
if runs_today:
    runs_today = [str(name) for name in runs_today]
    regex = 'run\\d\\d'
    runs_today=re.findall(regex,''.join(runs_today))
    runs_today = np.array([int(name[-2:]) for name in runs_today])
    new_run_num = max(runs_today)+1
else:
    new_run_num = 1

savedir = maindir.joinpath('%s_run%02d_fc-data' % (stamp,new_run_num))
os.makedirs(savedir)
fid = open(savedir.joinpath('Feedbackdata.csv'),'w')
fid.write('Timestamp_Jetson,Timestamp_GPS,GPS_x,GPS_y,GPS_z,GPS_lat,GPS_long,GPS_alt,GPS_alt_rel,test_speed,test_move_to_alt,test_yawrate,Vspeed,Fspeed,Hspeed,ver_speed,for_speed,hor_speed,yawrate,move_to_alt\n')
#          (time.time(),    gps_t,        gps_x,gps_y,alt,  gps_lat,gps_long,gps_alt,gps_alt_rel,test_speed,test_move_to_alt,test_yawrate,vspeed,fspeed,hspeed,ver_speed,for_speed,hor_speed,yawrate)
# ---------------------------- Creating a saving directory ---------------------------- #



#----------------------- PRINT OPTIONS ----------------------------#
PRINT_PITCH = False
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
print_flags = False
print_state = True
print_speeds = False
#----------------------- PRINT OPTIONS ----------------------------#



#----------------------- Global Parameters (srijan) ----------------------------#
global sampling_t0
global track_sampling_time
global fspeed_head, hspeed_head
global start_time_track, source_gps_track
global horizontalerror, verticalerror, sizeerror
global horizontalerror_smoketrack, verticalerror_smoketrack, sizeerror_smoketrack
global horizontalerror_keypoint, verticalerror_keypoint
global head, slope_deg
global setpoint_global_pub
global yawing_using_kalman_filter
global proportional_gain
global test_speed, test_move_to_alt
#----------------------- Global Parameters (srijan) ----------------------------#



#----------------------- Initialization (srijan) ----------------------------#
smoke_dir = ''
head, slope_deg = 0.0, 0.0
global source_gps, drone_gps
source_gps = [0.0, 0.0, 0.0] # lattitude, longitude and altitude
drone_gps = [0.0, 0.0, 0.0] # lattitude, longitude and altitude
global horizontalerror_smoketrack_list
horizontalerror_smoketrack_list =[]
time_lastbox, time_lastbox_smoketrack = None, None
yawing_using_kalman_filter = False
first_look_around = True
sample_along_heading = False
sampling_t0 = None
track_sampling_time = True
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
alt_flow = 3 # altitude at which to stop descent and keep constant for optical flow
alt_sampling = 5 # 1.5 # altitude setpoint at which to do a controlled sampling based on mean flow direction
alt_min = 1 # minimum allowable altitude

# gain values
size_gain = 1
yaw_gain = 1
gimbal_pitch_gain = -40 # previously -100, this needs to be adjusted depending on the frame rate for detection (20fps=-50gain, while saving video; 30fps=-25gain with no saving)
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
publish_rate = 0
# ---------------------------------------------------- Deployment Parameters (Nate) ---------------------------------------------------- #



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



def pose_callback(pose):
    """
    Returns Drone Pose: yaw, gps_x, gps_y and gps_z(alt)
    """
    global yaw, alt, gps_x, gps_y

    q = pose.pose.orientation
    # yaw = atan2(2.0*(q.y*q.z + q.w*q.x), q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z)
    _, _, y = euler_from_quaternion(q.x,q.y,q.z,q.w)
    yaw = y
    alt = pose.pose.position.z
    gps_x = pose.pose.position.x
    gps_y = pose.pose.position.y
    if print_stat:
        print(f'yaw: {yaw}, gps_x:{gps_x}, gps_y:{gps_y}, gps_z(alt): {alt}')



def compass_hdg_callback(heading):
    """
    Returns Drone's heading with respect to North
    """
    global head
    head = heading.data



def heading_btw_points():
    """
    Calculates bearing between two coordinates (smoke source and instantaneous drone location)
    https://mapscaping.com/how-to-calculate-bearing-between-two-coordinates/
    """
    global head, source_gps, drone_gps

    drone_heading = head
    # Format: point = (latitude_point, longitude_point)
    point_A = (math.radians(source_gps[0]), math.radians(source_gps[1]))
    point_B = (math.radians(drone_gps[0]), math.radians(drone_gps[1]))
    sin, cos = math.sin, math.cos

    # Formula : bearing  = math.atan2(math.sin(long2 - long1) * math.cos(lat2),
    #                                   math.cos(lat1) * math.sin(lat2) - math.sin(lat1) *
    #                                   math.cos(lat2) * math.cos(long2 - long1))
    bearing_AB = math.atan2(sin(point_B[1] - point_A[1]) * cos(point_B[0]),
                            cos(point_A[0]) * sin(point_B[0]) - sin(point_A[0]) *
                            cos(point_B[0]) * cos(point_B[1] - point_A[1]))

    heading_AB = math.degrees(bearing_AB)
    heading_AB = (heading_AB + 360) % 360
    diff_head = heading_AB - drone_heading

    if heading_AB > 270 and drone_heading < 90:
        diff_head = diff_head - 360
    elif heading_AB < 90 and drone_heading > 270:
        diff_head = 360 + diff_head

    yaw_correction = diff_head
    if print_stat:
        print(f'heading btw source [{point_A[0]:.8f}',
              f'{point_A[1]:.8f}] & drone[{point_B[0]:.8f}',
              f'{point_B[1]:.8f}] : {heading_AB:.4f} deg |',
              f'drone heading: {head} | Diff in heading: {diff_head:.2f}')

    return yaw_correction



def state_callback(state):
    """
    Check if drone FCU is in loiter or guided mode
    """
    global guided_mode, print_state

    if print_stat:
        print(f'state.mode={state.mode}')

    if state.mode == 'GUIDED':
        guided_mode = True
        if print_stat:
            print('GUIDED')
        if print_flags and print_state:
            print("state.mode == 'OFFBOARD' -> guided_mode = True")
            print_state = False
    else:
        print("!!!!!!!!!!!!!!!! NOT OFFBOARD")
        guided_mode = False
        if print_flags:
            print("state.mode == 'Not OFFBOARD' -> guided_mode = False")



def time_callback(gpstime):
    """
    Returns GPS time callback
    """
    global gps_t

    gps_t = float(gpstime.time_ref.to_sec())
    if print_stat:
        print(f"Time: {gps_t}")



def gps_callback(gpsglobal):
    """
    Returns GPS lat, long, alt (in order), drone_gps (lat, long, alt)
    """
    global gps_lat, gps_long, gps_alt, drone_gps

    gps_lat = gpsglobal.latitude
    gps_long = gpsglobal.longitude
    gps_alt = gpsglobal.altitude
    drone_gps[0], drone_gps[1], drone_gps[2] = gps_lat, gps_long, gps_alt

    if print_stat:
        print(f"gps_lat: {gps_lat}, gps_long: {gps_long}, gps_alt: {gps_alt}",
              f"drone_gps: {drone_gps}")



def rel_alt_callback(altrel):
    """
    Returns relative GPS altitude
    """
    global gps_alt_rel

    gps_alt_rel = altrel.data # relative altitude just from GPS data
    if print_stat:
        print(f"gps_alt_rel: {gps_alt_rel}")



def arguments():
    """
    Dealing with command line inputs
    """
    global test_speed, test_move_to_alt, test_yawrate
    global ver_speed, hor_speed, for_speed
    global altitude
    global yawrate

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--test_speed', type=bool, default=False, 
                        help='Perform speed test [True/False]')
    parser.add_argument('--ver_speed', type=float, default=0, help='vertical speed')
    parser.add_argument('--hor_speed', type=float, default=0, help='horizontal speed')
    parser.add_argument('--for_speed', type=float, default=0, help='forward speed')

    parser.add_argument('--test_move_to_alt', type=bool, default=False,
                        help='Test moving to setpoint altitude')
    parser.add_argument('--alt', type=float, default=0, help='setpoint altitude')

    parser.add_argument('--test_yawrate', type=bool, default=False,
                        help='Test yawrate')
    parser.add_argument('--yawrate', type=float, default=0, help='yawrate')

    args = parser.parse_args()

    test_speed = args.test_speed
    test_move_to_alt = args.test_move_to_alt
    test_yawrate = args.test_yawrate

    ver_speed = args.ver_speed
    hor_speed = args.hor_speed
    for_speed = args.for_speed
    altitude = args.alt
    yawrate = args.yawrate

    if test_speed:
        print(f"Running Speed Test: {test_speed}")
        print(f"vspeed:{ver_speed} | hspeed:{hor_speed} | vspeed:{for_speed}")
    if test_move_to_alt:
        print(f"Running Move to setpoint altitude test: {test_move_to_alt}")
        print(f"Altitude: {altitude}")
    if test_yawrate:
        print(f"Running Yawrate test: {test_yawrate}")
        print(f"yawrate: {yawrate}")

    return args



# Do not delete this function
def build_local_setpoint(x, y, z, yaw_):
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
    q_x, q_y, q_z, q_w= euler_to_quaternion(roll, pitch, yaw_)
    local_pose_setpoint.pose.orientation.x = q_x
    local_pose_setpoint.pose.orientation.y = q_y
    local_pose_setpoint.pose.orientation.z = q_z
    local_pose_setpoint.pose.orientation.w = q_w

    return local_pose_setpoint



def send_velocity_command(forward_speed, lateral_speed, vertical_speed, correction_yaw_rate):
    """
    send_velocity_command
    """
    global twist_stamped_pub, yaw

    cmd_vel = TwistStamped()
    cmd_vel.header.stamp = rospy.Time.now()

    x_speed = (math.cos(yaw)*forward_speed + math.sin(yaw)*lateral_speed)*(0.3)
    y_speed = (math.sin(yaw)*forward_speed - math.cos(yaw)*lateral_speed)*(0.3)
    z_speed = vertical_speed

    # Set linear velocity for forward, lateral, vertical motion (in the body frame)
    cmd_vel.twist.linear.x = x_speed
    cmd_vel.twist.linear.y = y_speed
    cmd_vel.twist.linear.z = z_speed
    # Set angular velocity for yaw motion (in the body frame)
    cmd_vel.twist.angular.z = correction_yaw_rate

    if print_stat:
        print(f'Command Velocity: x_speed: {x_speed} | y_speed: {y_speed} |',
              f'z_angular: {correction_yaw_rate}')

    # Publish the TwistStamped message
    twist_stamped_pub.publish(cmd_vel)



def dofeedbackcontrol():
    """
    main loop
    """
    # ----------------------------- Nate ----------------------------- #
    global pitchcommand, yawcommand
    global above_object, forward_scan
    global yaw_mode,OPT_FLOW,OPT_COMPUTE_FLAG,MOVE_ABOVE
    global move_up, USE_PITCH_ERROR
    global moving_to_set_alt
    global hspeed,vspeed,fspeed
    global yawrate
    global horizontalerror,verticalerror
    global twistpub, twistmsg,rcmsg,rcpub
    # ----------------------------- Nate ----------------------------- #

    global gimbal
    global publish_rate
    global smoketrack_pub, sample_along_heading
    global sampling_t0, track_sampling_time
    global fspeed_head, hspeed_head
    global kf, previous_yaw_measurements
    global head, source_gps, drone_gps
    global setpoint_global_pub
    global horizontalerror_keypoint, verticalerror_keypoint
    global yaw_pub, twist_stamped_pub
    global sampling_time_info_pub

    publish_rate = time.time()

    # ------------------------------------------ Nate ----------------------------------------- #
    #Initialize publishers/subscribers/node
    rospy.Subscriber('/mavros/local_position/pose', PoseStamped, pose_callback)
    rospy.Subscriber('/mavros/time_reference',TimeReference,time_callback)
    rospy.Subscriber('/mavros/global_position/global',NavSatFix,gps_callback)
    rospy.Subscriber('/mavros/global_position/rel_alt',Float64,rel_alt_callback)
    rospy.Subscriber('/mavros/state',State,state_callback)
    twistpub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=1)
    rcpub = rospy.Publisher('/mavros/rc/override', OverrideRCIn, queue_size=1)
    # ------------------------------------------ Nate ----------------------------------------- #

    # ------------------------------------------ New ------------------------------------------ #
    rospy.Subscriber('/mavros/global_position/compass_hdg',Float64,compass_hdg_callback)
    twist_stamped_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', 
                                        TwistStamped, queue_size=1)
    sampling_time_info_pub = rospy.Publisher('sampling_time_info_topic', Float64, queue_size=10)

    global start_time_track, source_gps_track
    start_time_track, source_gps_track = True, True
    # ------------------------------------------ New ------------------------------------------ #

    # control loop
    twistmsg = Twist()
    rcmsg = OverrideRCIn()
    rcmsg.channels = np.zeros(18,dtype=np.uint16).tolist()
    rate = rospy.Rate(20) # originally 10hz

    while not rospy.is_shutdown():
        if guided_mode:
        #rise_up()
            if test_speed:
                speed_test()
            if test_move_to_alt:
                move_to_set_alt_test()
            if test_yawrate:
                yawrate_test()
            # writing control states and data to csv
            save_log()
        rate.sleep()



def speed_test():
    """
    Run speed test
    """
    global ver_speed, hor_speed, for_speed
    print(f'ver_speed: {ver_speed} | hor_speed: {hor_speed} | for_speed : {for_speed}', end='\r')

    twistmsg.linear.x = hor_speed
    twistmsg.linear.y = for_speed
    twistmsg.linear.z = ver_speed
    twistmsg.angular.z = 0
    twistpub.publish(twistmsg)
    time.sleep(0.2)



def yawrate_test():
    """
    Run yawrate test
    """
    global yawrate
    print(f'yawrate : {yawrate}', end='\r')

    twistmsg.linear.x = 0
    twistmsg.linear.y = 0
    twistmsg.linear.z = 0
    twistmsg.angular.z = yawrate
    twistpub.publish(twistmsg)
    time.sleep(0.2)



def move_to_set_alt_test():
    """
    Run move_to_set_alt test
    """
    global altitide
    move_to_alt = altitude
    alt_diff = gps_alt - move_to_alt

    twistmsg.linear.x = 0
    twistmsg.linear.y = 0
    twistmsg.angular.z = 0

    if abs(alt_diff) < 0.3:
        twistmsg.linear.z = 0
        print("Setpoint altitude reached!")
    elif alt_diff < 1: # too low
        twistmsg.linear.z = 0.5
        print(f'Too Low! move_to_alt: {move_to_alt} | current_alt: {gps_alt: .3f} |',
              f' current rel_alt: {gps_alt_rel: .3f} | alt_diff : {alt_diff: .3f}', end='\r')
    elif alt_diff > 1: # too high
        twistmsg.linear.z = -0.5
        print(f'Too High! move_to_alt: {move_to_alt} | current_alt: {gps_alt: .3f} |',
              f' current rel_alt: {gps_alt_rel: .3f} | alt_diff : {alt_diff: .3f}', end='\r')

    twistpub.publish(twistmsg)
    time.sleep(0.2)



def rise_up(dz = 5,vz=3):
    """
    simple loop to go up or down, usually to get above object
    """
    global twistpub, twistmsg

    if print_stat:
        print(f'Rising {dz}m at {vz}m/s...')
    twistmsg.linear.x = 0
    twistmsg.linear.y = 0
    twistmsg.linear.z = vz

    for idx in range(int((dz/vz)/0.2)):
        twistpub.publish(twistmsg)
        time.sleep(0.2)

    twistmsg.linear.z = 0
    twistpub.publish(twistmsg)
    return



def save_log():
    """
    writing data to csv
    """
    global ver_speed, hor_speed, for_speed, yawrate

    # fid.write('%f,%f,%f,%f,%f,%f,%f,%f,%f,%s,%s,%s,%f,%f,%f,%f,%f,%f,%f,%f\n' %
    #     (time.time(),gps_t,gps_x,gps_y,alt,gps_lat,gps_long,gps_alt,gps_alt_rel,
    #       test_speed,test_move_to_alt,test_yawrate,vspeed,fspeed,hspeed,
    #       ver_speed,for_speed,hor_speed,yawrate,altitude))
    fid.write(f'{time.time()},{gps_t},{gps_x},{gps_y},{alt},{gps_lat},{gps_long}',
              f'{gps_alt},{gps_alt_rel},{test_speed},{test_move_to_alt},{test_yawrate}',
              f'{vspeed},{fspeed},{hspeed}',
              f'{ver_speed},{for_speed},{hor_speed},{yawrate},{altitude}\n')



if __name__ == '__main__':
    arguments()
    print("Initializing feedback node...")
    rospy.init_node('feedbackcontrol_node', anonymous=False)
    twist_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=1)

    print(f'Executing in ==> {EXECUTION}')
    try:
        dofeedbackcontrol()
    except rospy.ROSInterruptException:
        pass



# def boundingbox_callback(box):
#     """
#     returns detection bounding box size
#     horzontal_error and vertical_error
#     positive errors give right, up
#     if box center is on LHS of image, error is positive
#     if box center is on upper half, error is positive
#     """
#     global horizontalerror, verticalerror, sizeerror
#     global time_lastbox, pitchcommand, yawcommand
#     global MOVE_ABOVE, OPT_FLOW

#     # positive errors give right, up
#     if box.bbox.center.x != -1:
#         if print_stat:
#             print("Bounding box found (box.bbox.center.x != -1)")
#         time_lastbox = rospy.Time.now()
#         # take the average so that a very thin, long box will still be seen as small
#         bboxsize = (box.bbox.size_x + box.bbox.size_y)/2

#         # different bbox size desired for approach and above stages for hybrid mode
#         if not above_object:
#             if print_stat:
#                 print("Not above object ... ")
#             # if box is smaller than setpoint, error is positive
#             sizeerror = setpoint_size_approach - bboxsize
#         else:
#             if print_stat:
#                 print("above_object was set to true previously")
#             # if box is smaller than setpoit, error is positive
#             sizeerror = setpoint_size - bboxsize

#         if print_size_error:
#             print(f'Setpoint - bbox (sizeerror) = {sizeerror}')

#         if not OPT_FLOW:
#             if print_stat:
#                 print("OPT_FLOW is set to False: Optical flow not running")

#             # if box center is on LHS of image, error is positive
#             horizontalerror = .5-box.bbox.center.x
#             # if box center is on upper half of image, error is positive
#             verticalerror = .5-box.bbox.center.y

#             if print_stat:
#                 print(f'Horzontal error: {round(horizontalerror, 5)}',
#                       f'Vertical error: {round(verticalerror, 5)}')
#                 #print(f"Changed pitchcommand,delta: {round(pitchcommand, 5)},{round(pitchdelta, 5)}",
#                 #       f"changed yawcommand, delta: {round(yawcommand, 5)},{round(yawdelta, 5)}")

#         # if close and gimbal pitched upward, move to get above the object
#         if pitchcommand < pitch_thresh and bboxsize > 0.75:
#             MOVE_ABOVE = True
#             if print_stat:
#                 print('Gimbal pitched upward moving above to get above object ...',
#                       ': MOVE_ABOVE is set to True')
#     else:
#         if print_stat:
#             print("Bounding box not found (box.bbox.center.x = -1)")

#     return



# def segmentation_callback(box):
#     """
#     returns smoke segmentation predictions
#     """
#     global horizontalerror_smoketrack, verticalerror_smoketrack, sizeerror_smoketrack, horizontalerror_smoketrack_list
#     global time_lastbox_smoketrack, pitchcommand, yawcommand
#     global MOVE_ABOVE, OPT_FLOW
#     global kf, previous_yaw_measurements
#     global sampling
#     global yawing_using_kalman_filter

#     # positive errors give right, up
#     if box.bbox.center.x != -1 and box.bbox.size_x > 2000:
#         if print_stat and sampling:
#             print("Sampling ... Yawing using Segmentation")
#         yawing_using_kalman_filter  = False
#         time_lastbox_smoketrack = rospy.Time.now()
#         # take the average so that a very thin, long box will still be seen as small
#         # bboxsize = (box.bbox.size_x + box.bbox.size_y)/2

#         # different bbox size desired for approach and above stages for hybrid mode
#         if not above_object:
#             if print_stat:
#                 print("Not above object ... ")
#             # setpoint_size_approach - bboxsize # if box is smaller than setpoit, error is positive
#             sizeerror_smoketrack = 0
#         else:
#             if print_stat:
#                 print("above_object was set to true previously")
#             # setpoint_size - bboxsize # if box is smaller than setpoit, error is positive
#             sizeerror_smoketrack = 0

#         if print_size_error:
#             print(f'Setpoint - bbox (sizeerror) = {sizeerror_smoketrack}')

#         if not OPT_FLOW:
#             if print_stat:
#                 print("OPT_FLOW is set to False: Optical flow not running")
#             # if box center is on LHS of image, error is positive
#             horizontalerror_smoketrack = .5-box.bbox.center.x
#             # if box center is on upper half of image, error is positive
#             verticalerror_smoketrack = .5-box.bbox.center.y

#             if print_stat:
#                 print(f'Horzontal error: {round(horizontalerror_smoketrack, 5)}',
#                       f'Vertical error: {round(verticalerror, 5)}')

#         # if close and gimbal pitched upward, move to get above the object
#         # if pitchcommand < pitch_thresh and bboxsize > 0.75:
#         #     MOVE_ABOVE = True
#         #     if print_stat: print('Gimbal pitched upward moving above to get above object',
#         #                           '... : MOVE_ABOVE is set to True')

#     return



# def keypoint_callback(box):
#     """
#     returns smoke keypoints predictions
#     """
#     global horizontalerror_keypoint, verticalerror_keypoint
#     global time_lastbox_keypoint, MOVE_ABOVE, OPT_FLOW

#     # positive errors give right, up
#     if box.bbox.center.x != -1:
#         time_lastbox_keypoint = rospy.Time.now()
#         # if smoke source keypoint is on LHS of image, error is positive
#         horizontalerror_keypoint = 0.5-box.bbox.center.x
#         # if smoke source keypoint is on upper half of image, error is positive
#         verticalerror_keypoint = 0.5-box.bbox.center.y
#     else:
#         horizontalerror_keypoint = 0
#         verticalerror_keypoint = 0
#     return



# def flow_callback(flow):
#     """
#     Returns optical flow predictions
#     """
#     global horizontalerror, verticalerror,time_lastbox
#     global pitchcommand, yawcommand
#     global flow_x,flow_y,flow_t
#     global OPT_FLOW,OPT_COMPUTE_FLAG

#     # typical values might be around 10 pixels, depending on frame rate
#     flow_x = flow.size_x # movement to the right in the image is positive
#     # this is made negative so that positive flow_y means the object was moving toward the top of the image
#     # (RAFT returns negative y for this (i.e., toward smaller y coordinates))
#     flow_y = -flow.size_y
#     # now movement to the top is positive flow_y
#     flow_t = float(gps_t)

#     # adjust the feedback error using the optical flow
#     if OPT_FLOW:
#         if print_stat:
#             print('Doing optical flow feedback ...')
#         # this signals to later in the code that the first usable optical flow data can be applied
#         # (instead of inheriting errors from bbox callback)
#         OPT_COMPUTE_FLAG = True
#         horizontalerror = -flow_x # to be consistent with the bounding box error
#         verticalerror = flow_y
#         if not above_object: # this should never end up being called normally, just for debugging optical flow in side-view
#             # pitch
#             pitchdelta = verticalerror * gimbal_pitch_gain
#             pitchdelta = min(max(pitchdelta,-limit_pitchchange),limit_pitchchange)
#             pitchcommand += pitchdelta
#             pitchcommand = min(max(pitchcommand,1000),2000)
#             yawdelta = horizontalerror * gimbal_yaw_gain
#             yawdelta = min(max(yawdelta,-limit_yawchange),limit_yawchange)
#             yawcommand += yawdelta
#             yawcommand = min(max(yawcommand,1000),2000)
#         if print_flow:
#             if print_stat:
#                 print(f'Flow x,y = {flow_x},{flow_y}')
#                 print(f'horizontal error: {horizontalerror}, verticalerror: {verticalerror}')

#         time_lastbox = rospy.Time.now()
#     else:
#         if print_stat:
#             print("OPT_FLOW was turned off: OPT_FLOW = False")
#     return
