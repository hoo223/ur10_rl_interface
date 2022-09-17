#!/usr/bin/python
# -*- coding: utf8 -*- 

## standard library
import numpy as np
import gym
from gym.utils import seeding, EzPickle
from gym import spaces

## ros library
import rospy
import tf
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Float64, Bool

## custom library
#sys.path.append("/root/catkin_ws/src/ur10_python_interface/scripts") # 필요한 python 파일이 있는 경로 추가

# mode
INIT = 0
TELEOP = 1
TASK_CONTROL = 2
JOINT_CONTROL = 3
RSA = 4
MOVEIT = 5
IDLE = 6

## class definition
class UR10Env(gym.Env, EzPickle):
    """ This is a base class for ur10 manupulation tasks. It can be used to initialize
        launch file, controller type, tf listener and ros node.
        Methods:
        _step:
            This method returns position of the tip of the ur10 robot and next state.
            It publishes given action for 1/5f seconds. Note that, actions are considered
            as at the same order with the initial joint name list.

            Args:
                action: A length 6 iterable corresponding to each joint action.
            Return:
                tip_position: X, Y, Z coordinate of the tip of the ur10.
                next_state: Last values for the joint postion and their time derivatives.

        _reset:
            This method pause the gazebo and then resets the ur10 joint angles to the
            initial joint angles. After that it listens joint states for a single message
            to return it. It does not refreshs the ros time.

            Return:
                state: Joint position and time derivaties of the last state.

    """
    # metadata = {'render.modes': ['human']}

    def __init__(self, prefix='unity', FPS=100):

        ob_dim = 12
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(ob_dim,), dtype=np.float32)
        self.continuous = True
        if self.continuous:
            self.action_space = spaces.Box(-1, +1, (6,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(64)

        ## initialization
        try:
            rospy.init_node('ur10_env', anonymous=True)
        except:
            print("Already initializaed!")
        
        self.prefix = prefix
        self.FPS = FPS
        self.period = rospy.Duration(1./FPS)
        self.rate = rospy.Rate(FPS)
        self._state = None
        self.singular_threshold = 0.03
        self.action_msg = Float64MultiArray()

        ## publisher
        self.task_action_pub = rospy.Publisher(prefix+'/rsa_command', Float64MultiArray, queue_size=10)
        
        ## subscriber -> ros init 다음에
        joint_state_subscriber = rospy.Subscriber(prefix+"/joint_states", JointState, callback=self.joint_state_callback, queue_size=10)
        m_index_subscriber = rospy.Subscriber(prefix+"/m_index", Float64, callback=self.m_index_callback, queue_size=10)
        self_collision_subscriber = rospy.Subscriber(prefix+"/self_collision", Bool, callback=self.self_collision_callback, queue_size=10)
        ik_falied_subscriber = rospy.Subscriber(prefix+"/ik_failed", Bool, callback=self.ik_failed_callback, queue_size=10)
    
        ## tf listener
        self.base = prefix+"/base_link"
        self.end = prefix+"/tool_gripper" 
        now = rospy.Time.now()
        self.tf_listener = tf.TransformListener()

        print("Env Loaded!")


    def joint_state_callback(self, data):
        self._state = np.array(data.position + data.velocity)
        self._state_time = data.header.stamp
        # now = rospy.Time.now()
        # self.tf_listener.waitForTransform(self.base, self.end, now, rospy.Duration(5.0));
        # position, quaternion = self.tf_listener.lookupTransform(self.base, self.end, now)
        # self.tip_position = position
        # self._additional_state_callback()

    def m_index_callback(self, data):
        self.m_index = data.data

    def self_collision_callback(self, data):
        self.self_collision = data.data
        
    def ik_failed_callback(self, data):
        self.ik_failed = data.data

    def step(self, action):
        print(action)
        data = action.tolist()
        print(data)
        data.append(-1.0)
        self.action_msg.data = data
        ## publich the action message
        self.task_action_pub.publish(self.action_msg) 
        start_ros_time = rospy.Time.now()
        while True:     
            elapsed_time = rospy.Time.now() - start_ros_time
            if elapsed_time > self.period*(99.0/100):
                if elapsed_time > self.period:
                    break
                else:
                    rospy.sleep(self.period-elapsed_time)
                    break
            else:
                rospy.sleep(self.period/100.0)
            #self.rate.sleep()

        end_ros_time = rospy.Time.now()
        #print(end_ros_time - start_ros_time)
        ## update state
        rospy.wait_for_message(self.prefix+"/joint_states", JointState)
        ob_next = self._state

        ## update reward
        reward = self._get_reward(ob_next)

        ## update done
        if (reward <= -999):
            done = True
        else:
            done = False
            
        info = None

        return (ob_next, reward, done, info)


    def start_client(self):
        rospy.set_param(self.prefix+'/mode', TASK_CONTROL) 

    def reset(self):
        rospy.set_param(self.prefix+'/mode', INIT) 
        rospy.sleep(2)
        rospy.set_param(self.prefix+'/mode', RSA) 
        
    def _get_reward(self, state):
        """ Reward implementation"""
        reward = 0
        #rospy.wait_for_message('self_collision', Bool)
        self_collision = self.self_collision
        ik_failed = self.ik_failed
        m_index = self.m_index
        
        # penalty for self collision and ik failure
        if (self_collision == True) or (ik_failed == True) or (m_index < 0.03):
            reward -= 1000
            #print("self collision")
        else:  
            # reward for time
            reward = 0
            
        return reward

    def close(self):
        rospy.set_param(self.prefix+'/mode', INIT) 

