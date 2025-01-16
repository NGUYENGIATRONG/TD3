import numpy as np
import gym
from gym import spaces
from simulation import walking_controller_solo12
import math
import random
from collections import deque
import pybullet
from simulation import pybullet_client as pybullet_client
from utils import solo12_kinematics
import time
import pybullet_data
# import SlopedTerrainLinearPolicy.gym_sloped_terrain.envs.planeEstimation.get_terrain_normal as normal_estimator
from utils import get_terrain_normal as normal_estimator

start_time = time.time()
LEG_POSITION = ["fl_", "bl_", "fr_", "br_"]
KNEE_CONSTRAINT_POINT_RIGHT = [0.014, 0, 0.076]  # hip
KNEE_CONSTRAINT_POINT_LEFT = [0.0, 0.0, -0.077]  # knee
RENDER_HEIGHT = 720
RENDER_WIDTH = 960
PI = np.pi
no_of_points = 100

MOTOR_NAMES = [
    "motor_hip_fl", "motor_knee_fl", "motor_abduction_fl",
    "motor_hip_hr", "motor_knee_hr", "motor_abduction_hr",
    "motor_hip_fr", "motor_knee_fr", "motor_abduction_fr",
    "motor_hip_hl", "motor_knee_hl", "motor_abduction_hl"
]


def constrain_theta(theta):
    theta = np.fmod(theta, 2 * no_of_points)
    if theta < 0:
        theta = theta + 2 * no_of_points
    return theta


# def transform_action(action):
#     action = np.clip(action, -1, 1)
#     action[:4] = (action[:4] + 1) / 2  # Step lengths are positive always
#     action[:4] = action[:4] * 0.136  # Max steplength = 0.136
#     action[4:8] = action[4:8] * np.pi / 2  # PHI can be [-pi/2, pi/2]
#     action[8:12] = 0.07 * (action[8:12] + 1) / 2  # elipse center y is positive always
#     action[12:16] = action[12:16] * 0.04  # x
#     action[16:20] = action[16:20] * 0.035  # Max allowed Z-shift due to abduction limits is 3.5cm
#     action[17] = -action[17]
#     action[19] = -action[19]
#
#     return action


# def add_noise(sensor_value, sd=0.04):
#     """
#     Adds sensor noise of user defined standard deviation in current sensor_value
#     """
#     noise = np.random.normal(0, sd, 1)
#     sensor_value = sensor_value + noise[0]
#     return sensor_value


class Solo12PybulletEnv(gym.Env):

    def __init__(self,
                 render=True,
                 default_pos=(0, 0, 0.33),
                 default_ori=(0, 0, 0, 1),
                 on_rack=False,
                 gait='trot',
                 phase=(0, no_of_points, no_of_points, 0),
                 # [FR, FL, BR, BL]
                 action_dim=20,
                 end_steps=1000,
                 stairs=False,
                 downhill=False,
                 seed_value=100,
                 wedge=False,
                 imu_noise=False,                 pd_control_enabled=True,
                 step_length=0.08,
                 step_height=0.06,
                 motor_kp=30.5,
                 motor_kd=0.68,
                 deg=5):

        # global phase

        # self.gait = gait
        self.pd_control_enabled = pd_control_enabled
        self.x_init = 0
        self.y_init = -0.23
        self.step_length = step_length / 2
        self.step_height = step_height
        self.motor_offset = [np.pi / 2, 0, 0]
        self.phase = phase
        self.plane = None
        self.solo12 = None
        self.new_fric_val = None
        self.FrontMass = None
        self.BackMass = None
        self._motor_id_list = None
        self._joint_name_to_id = None
        self.wedge_halfheight = None
        self.wedgePos = None
        self.wedgeOrientation = None
        self.robot_landing_height = None
        self.wedge = None
        self._is_stairs = stairs
        self._is_wedge = wedge
        self._is_render = render
        self._on_rack = on_rack
        self.rh_along_normal = 0.24
        self.no_of_points = 100
        self.kinematic = solo12_kinematics.Solo12Kinematic()
        self.leg_name_to_sol_branch_Solo12 = {'fl': 1, 'fr': 1, 'hl': 0, 'hr': 0}
        self.seed_value = seed_value
        random.seed(self.seed_value)
        if self._is_render:
            self._pybullet_client = pybullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = pybullet_client.BulletClient()

        self._theta = 0

        self._frequency = -3.5
        self.termination_steps = end_steps
        self.downhill = downhill

        # PD gains
        self._kp = motor_kp
        self._kd = motor_kd

        self.dt = 0.005
        self._frame_skip = 25
        self._n_steps = 0
        self._action_dim = action_dim

        self._obs_dim = 11

        self.action = np.zeros(self._action_dim)

        self._last_base_position = [0, 0, 0]
        self.last_yaw = 0
        self._distance_limit = float("inf")

        self.current_com_height = 0.243

        # wedge_parameters
        self.wedge_start = 0.5
        self.wedge_halflength = 2

        if gait == 'trot':
            self.phase = [0, self.no_of_points, self.no_of_points, 0]
        elif gait == 'walk':
            self.phase = [0, self.no_of_points, 3 * self.no_of_points / 2, self.no_of_points / 2]
        self._walkcon = walking_controller_solo12.WalkingController(gait_type=gait, phase=self.phase)
        self.inverse = False
        self._cam_dist = 1.0
        self._cam_yaw = 0.0
        self._cam_pitch = 0.0

        self.avg_vel_per_step = 0
        self.avg_omega_per_step = 0

        self.linearV = 0
        self.angV = 0
        self.prev_vel = [0, 0, 0]
        self.perturb_steps = 0
        self.x_f = 0
        self.y_f = 0

        self.clips = 100

        self.friction = 0.75
        self.ori_history_length = 3
        self.ori_history_queue = deque([0] * 3 * self.ori_history_length,
                                       maxlen=3 * self.ori_history_length)  # observation queue

        self.step_disp = deque([0] * 100, maxlen=100)
        self.stride = 5

        self.incline_deg = deg
        self.incline_ori = 0

        self.prev_incline_vec = (0, 0, 1)

        self.terrain_pitch = []
        self.add_IMU_noise = imu_noise

        self.INIT_POSITION = default_pos
        self.INIT_ORIENTATION = default_ori
        self.desired_height = 0

        self.support_plane_estimated_pitch = 0
        self.support_plane_estimated_roll = 0

        self.pertub_steps = 0
        self.x_f = 0
        self.y_f = 0

        ## Gym env related mandatory variables
        # self._obs_dim = 3 * self.ori_history_length + 2  # [r,p,y]x previous time steps, suport plane roll and pitch
        # observation_high = np.array([np.pi / 2] * self._obs_dim)
        # observation_low = -observation_high
        # self.observation_space = spaces.Box(observation_low, observation_high)

        action_high = np.array([1] * self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high)

        self.hard_reset()

        self.Set_Randomization(default=True, idx1=2, idx2=2)
        self.height = []
        abduction_low = np.radians(-45)
        abduction_high = np.radians(45)
        other_motor_low = np.radians(-90)
        other_motor_high = np.radians(90)

        action_low = np.array([other_motor_low, other_motor_low, abduction_low,
                               other_motor_low, other_motor_low, abduction_low,
                               other_motor_low, other_motor_low, abduction_low,
                               other_motor_low, other_motor_low, abduction_low], dtype=np.float32)

        action_high = np.array([other_motor_high, other_motor_high, abduction_high,
                                other_motor_high, other_motor_high, abduction_high,
                                other_motor_high, other_motor_high, abduction_high,
                                other_motor_high, other_motor_high, abduction_high], dtype=np.float32)

        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        observation_dim = len(self.GetObservation())
        observation_low = -np.inf * np.ones(observation_dim, dtype=np.float32)
        observation_high = np.inf * np.ones(observation_dim, dtype=np.float32)

        self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)
        if self._is_stairs:
            boxHalfLength = 0.1
            boxHalfWidth = 1
            boxHalfHeight = 0.015
            sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
                                                                   halfExtents=[boxHalfLength, boxHalfWidth,
                                                                                boxHalfHeight])
            boxOrigin = 0.3
            n_steps = 15
            self.stairs = []
            for i in range(n_steps):
                step = self._pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                             basePosition=[boxOrigin + i * 2 * boxHalfLength, 0,
                                                                           boxHalfHeight + i * 2 * boxHalfHeight],
                                                             baseOrientation=[0.0, 0.0, 0.0, 1])
                self.stairs.append(step)
                self._pybullet_client.changeDynamics(step, -1, lateralFriction=0.8)

    def hard_reset(self):
        """
        Function to
        1) Set simulation parameters which remains constant throughout the experiments
        2) load urdf of plane, wedge and robot in initial conditions
        """
        self._pybullet_client.resetSimulation()
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(300))
        self._pybullet_client.setTimeStep(self.dt / self._frame_skip)

        self.plane = self._pybullet_client.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath())
        # self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 0.9])
        self._pybullet_client.setGravity(0, 0, -9.81)

        if self._is_wedge:

            wedge_halfheight_offset = 0.01

            self.wedge_halfheight = wedge_halfheight_offset + 1.5 * np.tan(np.radians(self.incline_deg)) / 2.0
            self.wedgePos = [0, 0, self.wedge_halfheight]
            self.wedgeOrientation = self._pybullet_client.getQuaternionFromEuler([0, 0, self.incline_ori])

            if not self.downhill:
                wedge_model_path = "simulation/Wedges/uphill/urdf/wedge_" + str(
                    self.incline_deg) + ".urdf"

                self.INIT_ORIENTATION = self._pybullet_client.getQuaternionFromEuler(
                    [np.radians(self.incline_deg) * np.sin(self.incline_ori),
                     -np.radians(self.incline_deg) * np.cos(self.incline_ori), 0])

                self.robot_landing_height = wedge_halfheight_offset + 0.28 + np.tan(
                    np.radians(self.incline_deg)) * abs(self.wedge_start)

                self.INIT_POSITION = [self.INIT_POSITION[0], self.INIT_POSITION[1], self.robot_landing_height]

            else:
                wedge_model_path = "simulation/Wedges/downhill/urdf/wedge_" + str(
                    self.incline_deg) + ".urdf"

                self.robot_landing_height = wedge_halfheight_offset + 0.28 + np.tan(
                    np.radians(self.incline_deg)) * 1.5

                self.INIT_POSITION = [0, 0, self.robot_landing_height]  # [0.5, 0.7, 0.3] #[-0.5,-0.5,0.3]

                self.INIT_ORIENTATION = [0, 0, 0, 1]

            self.wedge = self._pybullet_client.loadURDF(wedge_model_path, self.wedgePos, self.wedgeOrientation)

            self.SetWedgeFriction(0.7)

        model_path = 'simulation/solo12/solo12.urdf'
        self.solo12 = self._pybullet_client.loadURDF(model_path, self.INIT_POSITION, self.INIT_ORIENTATION)

        self._joint_name_to_id, self._motor_id_list = self.BuildMotorIdList()

        if self._on_rack:
            self._pybullet_client.createConstraint(
                self.solo12, -1, -1, -1, self._pybullet_client.JOINT_FIXED,
                [0, 0, 0], [0, 0, 0], [0, 0, 0.3])

        self._pybullet_client.resetBasePositionAndOrientation(self.solo12, self.INIT_POSITION, self.INIT_ORIENTATION)
        self._pybullet_client.resetBaseVelocity(self.solo12, [0, 0, 0], [0, 0, 0])
        self.reset_standing_position()
        # self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
        self.SetFootFriction(self.friction)

    def reset_standing_position(self):
        self.ResetPoseForAbd()
        self.ResetLeg()

    def reset(self, **kwargs):
        """
        This function resets the environment
        Note : Set_Randomization() is called before reset() to either randomize or set environment in default conditions.
        """
        self._theta = 0
        self._last_base_position = [0, 0, 0]

        self._pybullet_client.resetBasePositionAndOrientation(self.solo12, self.INIT_POSITION, self.INIT_ORIENTATION)
        self._pybullet_client.resetBaseVelocity(self.solo12, [0, 0, 0], [0, 0, 0])
        self.reset_standing_position()

        self._n_steps = 0
        return self.GetObservation()

    def set_wedge_friction(self, friction):
        """
        Hàm này điều chỉnh hệ số ma sát của miếng cản
        :param friction: hệ số ma sát mong muốn của miếng cản
        :return:
        """
        self._pybullet_client.changeDynamics(self.wedge, -1, lateralFriction=friction)

    def BuildMotorIdList(self):
        """
        function to map joint_names with respective motor_ids as well as create a list of motor_ids
        Ret:
        joint_name_to_id : Dictionary of joint_name to motor_id
        motor_id_list	 : List of joint_ids for respective motors in order [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA ]
        """
        num_joints = self._pybullet_client.getNumJoints(self.solo12)  #12
        joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.solo12, i)
            joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

            # adding abduction

        motor_id_list = [joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES]
        return joint_name_to_id, motor_id_list

    def GetMotorAngles(self):
        '''
        This function returns the current joint angles in order [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA ]
        '''
        motor_ang = [self._pybullet_client.getJointState(self.solo12, motor_id)[0] for motor_id in self._motor_id_list]
        return motor_ang#12

    def GetMotorVelocities(self):
        """
        This function returns the current joint velocities in order [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA ]
        """
        motor_vel = [self._pybullet_client.getJointState(self.solo12, motor_id)[1] for motor_id in self._motor_id_list]
        return motor_vel#12

    def GetBasePosAndOrientation(self):
        """
        This function returns the robot torso position(X,Y,Z) and orientation(Quaternions) in world frame
        """
        position, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.solo12))
        print(orientation)
        return position, orientation#(3,4)

    def GetBaseAngularVelocity(self):
        '''
        This function returns the robot base angular velocity in world frame
        Ret: list of 3 floats
        '''
        base_velocity = self._pybullet_client.getBaseVelocity(self.solo12)
        print(base_velocity)
        return base_velocity[1]

    def GetBaseLinearVelocity(self):
        """
        This function returns the robot base linear velocity in world frame
        Ret: list of 3 floats
        """
        base_velocity = self._pybullet_client.getBaseVelocity(self.solo12)
        return base_velocity[0]

    def get_motor_torques(self):
        motor_ang = [self._pybullet_client.getJointState(self.solo12, motor_id)[3] for motor_id in self._motor_id_list]
        return motor_ang

    def SetFootFriction(self, foot_friction):
        """
        This function modify coefficient of friction of the robot feet
        Args :
        foot_friction :  desired friction coefficient of feet
        Ret  :
        foot_friction :  current coefficient of friction
        """
        FOOT_LINK_ID = [2, 5, 8, 11]
        for link_id in FOOT_LINK_ID:
            self._pybullet_client.changeDynamics(
                self.solo12, link_id, lateralFriction=foot_friction)
        return foot_friction


    def apply_postion_control(self, desired_angles):
        for motor_id, angle in zip(self._motor_id_list, desired_angles):
            self.set_desired_motor_angle_by_id(motor_id, angle)

    def set_desired_motor_angle_by_id(self, motor_id, desired_angle):
        self._pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                    jointIndex=motor_id,
                                                    controlMode=self._pybullet_client.POSITION_CONTROL,
                                                    targetPosition=desired_angle,
                                                    positionGain=1,
                                                    velocityGain=1,
                                                    force=3)

    def set_desired_motor_angle_by_name(self, motor_name, desired_angle):
        self.set_desired_motor_angle_by_id(self._joint_name_to_id[motor_name], desired_angle)

    def apply_Ext_Force(self, x_f, y_f, link_index=1, visulaize=False, life_time=0.01):
        """
        function to apply external force on the robot
        Args:
            x_f  :  external force in x direction
            y_f  : 	external force in y direction
            link_index : link index of the robot where the force need to be applied
            visulaize  :  bool, whether to visulaize external force by arrow symbols
            life_time  :  life time of the visualization
         """
        force_applied = [x_f, y_f, 0]
        self._pybullet_client.applyExternalForce(self.solo12, link_index, forceObj=[x_f, y_f, 0], posObj=[0, 0, 0],
                                                 flags=self._pybullet_client.LINK_FRAME)
        f_mag = np.linalg.norm(np.array(force_applied))

        if visulaize and f_mag != 0.0:
            point_of_force = self._pybullet_client.getLinkState(self.solo12, link_index)[0]

            lam = 1 / (2 * f_mag)
            dummy_pt = [point_of_force[0] - lam * force_applied[0],
                        point_of_force[1] - lam * force_applied[1],
                        point_of_force[2] - lam * force_applied[2]]
            self._pybullet_client.addUserDebugText(str(round(f_mag, 2)) + " N", dummy_pt, [0.13, 0.54, 0.13],
                                                   textSize=2, lifeTime=life_time)
            self._pybullet_client.addUserDebugLine(point_of_force, dummy_pt, [0, 0, 1], 3, lifeTime=life_time)

    # def compute_motor_angles(self, x, y, z, leg_name):
    #     """
    #     Compute angles from x,y,z
    #     :param x: x coordinate
    #     :param y: y coordinate
    #     :param z: z coordinate
    #     :param leg_name: leg name
    #     :return: a list contain motor angles
    #     """
    #     return list(self.kinematic.inverse_kinematics(x, y, z, self.leg_name_to_sol_branch_Solo12[leg_name]))

    # def gen_signal(self, t, phase):
    #     """Generates a modified sinusoidal reference leg trajectory with half-circle shape.
    #
    #     Args:
    #       t: Current time in simulation.
    #       phase: The phase offset for the periodic trajectory.
    #
    #
    #     Returns:
    #       The desired leg x and y angle at the current time.
    #     """
    #     period = 1 / self._frequency
    #     theta = (2 * np.pi / period * t + phase) % (2 * np.pi)
    #
    #     x = -self.step_length * np.cos(theta) + self.x_init
    #     if theta > np.pi:
    #         y = self.y_init
    #     else:
    #         y = self.step_height * np.sin(theta) + self.y_init
    #     return x, y

    # def signal(self, t):
    #     """Generates the trotting gait for the robot.
    #
    #     Args:
    #       t: Current time in simulation.
    #
    #     Returns:
    #       A numpy array of the reference leg positions.
    #     """
    #     # Generates the leg trajectories for the two digonal pair of legs.
    #     ext_first_pair, sw_first_pair = self.gen_signal(t, phase=0)
    #     ext_second_pair, sw_second_pair = self.gen_signal(t, phase=np.pi)
    #
    #     motors_fl = self.compute_motor_angles(ext_first_pair, sw_first_pair, 0, "fl")
    #     motors_hr = self.compute_motor_angles(ext_first_pair, sw_first_pair, 0, "hr")
    #     motors_fr = self.compute_motor_angles(ext_second_pair, sw_second_pair, 0, "fr")
    #     motors_hl = self.compute_motor_angles(ext_second_pair, sw_second_pair, 0, "hl")
    #
    #     motors_fl[0] += self.motor_offset[0]
    #     motors_hr[0] += self.motor_offset[0]
    #     motors_fr[0] += self.motor_offset[0]
    #     motors_hl[0] += self.motor_offset[0]
    #
    #     trotting_signal = np.array([*motors_fl, *motors_hr, *motors_fr, *motors_hl])
    #     return trotting_signal

    # def get_time_since_reset(self):
    #     return self._n_steps * self.dt

    def SetLinkMass(self, link_idx, mass=0):
        """
        Function to add extra mass to front and back link of the robot

        Args:
            link_idx : link index of the robot whose weight to need be modified
            mass     : value of extra mass to be added

        Ret:
            new_mass : mass of the link after addition
        Note : Presently, this function supports addition of masses in the front and back link only (0, 11)
        """
        link_mass = self._pybullet_client.getDynamicsInfo(self.solo12, link_idx)[0]
        if link_idx == 0:
            link_mass = mass + 1.1
            self._pybullet_client.changeDynamics(self.solo12, 0, mass=link_mass)
        elif link_idx == 11:
            link_mass = mass + 1.1
            self._pybullet_client.changeDynamics(self.solo12, 11, mass=link_mass)

        return link_mass

    def getlinkmass(self, link_idx):
        """
        function to retrieve mass of any link
        Args:
            link_idx : link index of the robot
        Ret:
            m[0] : mass of the link
        """
        m = self._pybullet_client.getDynamicsInfo(self.solo12, link_idx)
        return m[0]

    def Set_Randomization(self, default=False, idx1=0, idx2=0, idx3=1, idx0=0, idx11=0, idxc=2, idxp=0, deg=5, ori=0):
        '''
        This function helps in randomizing the physical and dynamics parameters of the environment to robustify the policy.
        These parameters include wedge incline, wedge orientation, friction, mass of links, motor strength and external perturbation force.
        Note : If default argument is True, this function set above mentioned parameters in user defined manner
        '''
        if default:
            frc = [0.55, 0.6, 0.8]
            extra_link_mass = [0, 0.05, 0.1, 0.15]
            cli = [5.2, 6, 7, 8]
            pertub_range = [0, -60, 60, -100, 100]
            self.pertub_steps = 150
            self.x_f = 0
            self.y_f = pertub_range[idxp]
            self.incline_deg = deg + 2 * idx1
            self.incline_ori = ori + PI / 6 * idx2
            self.new_fric_val = frc[idx3]
            self.friction = self.SetFootFriction(self.new_fric_val)
            self.FrontMass = self.SetLinkMass(0, extra_link_mass[idx0])
            self.BackMass = self.SetLinkMass(11, extra_link_mass[idx11])
            self.clips = cli[idxc]

        else:
            avail_deg = [5, 7, 9, 11]
            extra_link_mass = [0, .05, 0.1, 0.15]
            pertub_range = [0, -60, 60, -100, 100]
            cli = [5, 6, 7, 8]
            self.pertub_steps = 150  # random.randint(90,200) #Keeping fixed for now
            self.x_f = 0
            self.y_f = pertub_range[random.randint(0, 4)]
            self.incline_deg = avail_deg[random.randint(0, 3)]
            self.incline_ori = (PI / 12) * random.randint(0, 6)  # resolution of 15 degree
            self.new_fric_val = np.round(np.clip(np.random.normal(0.6, 0.08), 0.55, 0.8), 2)
            self.friction = self.SetFootFriction(self.new_fric_val)
            i = random.randint(0, 3)
            self.FrontMass = self.SetLinkMass(0, extra_link_mass[i])
            i = random.randint(0, 3)
            self.BackMass = self.SetLinkMass(11, extra_link_mass[i])
            self.clips = np.round(np.clip(np.random.normal(6.5, 0.4), 5, 8), 2)

    def randomize_only_inclines(self, default=False, idx1=0, idx2=0, deg=7, ori=0):
        '''
        This function only randomizes the wedge incline and orientation and is called during training without Domain Randomization
        '''
        if default:
            self.incline_deg = deg + 2 * idx1
            self.incline_ori = ori + PI / 12 * idx2

        else:
            avail_deg = [7, 9, 11, 13, 15]
            self.incline_deg = avail_deg[random.randint(0, 4)]
            self.incline_ori = (PI / 12) * random.randint(0, 3)  # resolution of 15 degree

    @staticmethod
    def boundYshift(x, y):
        '''
        This function bounds Y shift with respect to current X shift
        Args:
             x : absolute X-shift
             y : Y-Shift
        Ret :
              y : bounded Y-shift
        '''
        if x > 0.5619:
            if y > 1 / (0.5619 - 1) * (x - 1):
                y = 1 / (0.5619 - 1) * (x - 1)
        return y

    def getYXshift(self, yx):
        '''
        This function bounds X and Y shifts in a trapezoidal workspace
        '''
        y = yx[:4]
        x = yx[4:]
        for i in range(0, 4):
            y[i] = self.boundYshift(abs(x[i]), y[i])
            y[i] = y[i] * 0.038
            x[i] = x[i] * 0.0418
        yx = np.concatenate([y, x])
        return yx

    def transform_action(self, action):
        '''
        Transform normalized actions to scaled offsets
        Args:
            action : 20 dimensional 1D array of predicted action values from policy in following order :
                     [(step lengths of FR, FL, BR, BL), (steer angles of FR, FL, BR, BL),
                      (Y-shifts of FR, FL, BR, BL), (X-shifts of FR, FL, BR, BL),
                      (Z-shifts of FR, FL, BR, BL)]
        Ret :
            action : scaled action parameters

        Note : The convention of Cartesian axes for leg frame in the codebase follow this order, Y points up, X forward and Z right.
               While in research paper we follow this order, Z points up, X forward and Y right.
        '''

        action = np.clip(action, -1, 1)

        action[:4] = (action[:4] + 1) / 2  # Step lengths are positive always

        action[:4] = action[:4] * 2 * 0.068 * 2  # Max steplength = 2x0.068

        action[4:8] = action[4:8] * PI / 2  # PHI can be [-pi/2, pi/2]

        action[8:12] = (action[8:12] + 1) / 2  # el1ipse center y is positive always

        action[8:16] = self.getYXshift(action[8:16]) * 3

        action[16:20] = action[16:20] * 0.035 * 4
        action[17] = -action[17]
        action[19] = -action[19]
        return action

    def get_foot_contacts(self):
        '''
        Retrieve foot contact information with the supporting ground and any special structure (wedge/stairs).
        Ret:
            foot_contact_info : 8 dimensional binary array, first four values denote contact information of feet [FR, FL, BR, BL] with the ground
            while next four with the special structure.
        '''
        foot_ids = [2, 5, 8, 11]
        foot_contact_info = np.zeros(8)

        for leg in range(4):
            contact_points_with_ground = self._pybullet_client.getContactPoints(self.plane, self.solo12, -1,
                                                                                foot_ids[leg])
            if len(contact_points_with_ground) > 0:
                foot_contact_info[leg] = 1

            if self._is_wedge:
                contact_points_with_wedge = self._pybullet_client.getContactPoints(self.wedge, self.solo12, -1,
                                                                                   foot_ids[leg])
                if len(contact_points_with_wedge) > 0:
                    foot_contact_info[leg + 4] = 1

            if self._is_stairs:
                for steps in self.stairs:
                    contact_points_with_stairs = self._pybullet_client.getContactPoints(steps, self.solo12, -1,
                                                                                        foot_ids[leg])
                    if len(contact_points_with_stairs) > 0:
                        foot_contact_info[leg + 4] = 1

        return foot_contact_info

    # def apply_action(self, action):
    #     # Todo: change action
    #     # motor_commands = self.walking_controller.test_elip(theta=self.theta)
    #     # motor_commands = np.array(motor_commands)
    #
    #     # Update theta
    #     # omega = 2 * self.no_of_points * self.frequency
    #     # self.theta = np.fmod(omega * self.dt + self.theta, 2 * self.no_of_points)
    #
    #     force_visualizing_counter = 0
    #     action = np.array(action)
    #     print("action: ", action)
    #     action = transform_action(action)
    #     print(action)
    #     action = self.transform_action(action)

    # # Apply action
    # for _ in range(self._frame_skip):
    #     if self.pd_control_enabled:
    #         self.apply_pd_control(action)
    #     else:
    #         self.apply_pd_control(action)
    #     self._pybullet_client.stepSimulation()
    #     if self._n_steps % 300 == 0:
    #         force_visualizing_counter += 1
    #         link = np.random.randint(0, 11)
    #         pertub_range = [0, -120, 120, -200, 200]
    #         y_force = pertub_range[np.random.randint(0, 4)]
    #         if force_visualizing_counter % 10 == 0:
    #             self.apply_Ext_Force(x_f=0, y_f=y_force, link_index=1, visulaize=True, life_time=0.2)
    #
    # self._n_steps += 1

    def step(self, action):
        # self.apply_action(action)
        action = self.transform_action(action)
        self.do_simulation(action, n_frames=self._frame_skip)
        # print(action)
        ob = self.GetObservation()
        reward, done = self._get_reward()
        return ob, reward, done, {}

    def CurrentVelocities(self):
        '''
        Returns robot's linear and angular velocities
        Ret:
            radial_v  : linear velocity
            current_w : angular velocity
        '''
        current_w = self.GetBaseAngularVelocity()[2]
        current_v = self.GetBaseLinearVelocity()
        radial_v = np.sqrt(current_v[0] ** 2 + current_v[1] ** 2)
        return radial_v, current_w

    def do_simulation(self, action, n_frames):
        """
        Chuyển đổi các tham số hành động thành các lệnh động cơ tương ứng
        với sự hỗ trợ của một bộ điều khiển quỹ đạo elip
        :param action:
        :param n_frames:
        :return:
        """
        omega = 2 * self.no_of_points * self._frequency
        omega = 2 * self.no_of_points * self._frequency
        leg_m_angle_cmd = self._walkcon.run_elip(self._theta, self.action)

        self._theta = constrain_theta(omega * self.dt + self._theta)

        m_angle_cmd_ext = np.array(leg_m_angle_cmd)

        force_visualizing_counter = 0

        for _ in range(n_frames):
            _ = self.apply_pd_control(m_angle_cmd_ext)
            self._pybullet_client.stepSimulation()

            if self.perturb_steps <= self._n_steps <= self.perturb_steps + self.stride:
                pass
                force_visualizing_counter += 1
                if force_visualizing_counter % 7 == 0:
                    self.apply_Ext_Force(self.x_f, self.y_f, visulaize=False, life_time=0.1)
                else:
                    self.apply_Ext_Force(self.x_f, self.y_f, visulaize=False)

        contact_info = self.get_foot_contacts()
        pos, ori = self.GetBasePosAndOrientation()

        rot_mat = self._pybullet_client.getMatrixFromQuaternion(ori)
        rot_mat = np.array(rot_mat)
        rot_mat = np.reshape(rot_mat, (3, 3))

        plane_normal, self.support_plane_estimated_roll, self.support_plane_estimated_pitch = \
            normal_estimator.vector_method_solo12(self.prev_incline_vec, contact_info, self.GetMotorAngles(), rot_mat)
        self.prev_incline_vec = plane_normal

        # line_id = self._pybullet_client.addUserDebugLine([0, 0, 0], plane_normal, lineColorRGB=[1, 0, 0], lineWidth=2)
        # if 'line_id' in globals():
        #     self._pybullet_client.removeUserDebugItem(line_id)

        self._n_steps += 1

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])

        base_pos, _ = self.GetBasePosAndOrientation()
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px).reshape(RENDER_WIDTH, RENDER_HEIGHT, 4)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self, pos, orientation):
        '''
        Check termination conditions of the environment
        Args:
            pos 		: current position of the robot's base in world frame
            orientation : current orientation of robot's base (Quaternions) in world frame
        Ret:
            done 		: return True if termination conditions satisfied
        '''
        done = False
        RPY = self._pybullet_client.getEulerFromQuaternion(orientation)

        if self._n_steps >= self.termination_steps:
            done = True
        else:
            if abs(RPY[0]) > np.radians(30):
                print('Oops, Robot about to fall sideways! Terminated')
                done = True

            if abs(RPY[1]) > np.radians(35):
                print('Oops, Robot doing wheely! Terminated')
                done = True

            if pos[2] > 0.9:
                print('Robot was too high! Terminated')
                done = True

        return done

    def _get_reward(self):
        '''
        Calculates reward achieved by the robot for RPY stability, torso height criterion and forward distance moved on the slope:
        Ret:
            reward : reward achieved
            done   : return True if environment terminates

        '''
        wedge_angle = np.radians(self.incline_deg)
        robot_height_from_support_plane = 0.26
        pos, ori = self.GetBasePosAndOrientation()

        RPY_orig = self._pybullet_client.getEulerFromQuaternion(ori)
        RPY = np.round(RPY_orig, 4)

        current_height = round(pos[2], 5)
        self.current_com_height = current_height
        standing_penalty = 3

        desired_height = robot_height_from_support_plane / np.cos(wedge_angle) + np.tan(wedge_angle) * (
                (pos[0]) * np.cos(self.incline_ori) + 0.5)

        roll_reward = np.exp(-45 * ((RPY[0] - self.support_plane_estimated_roll) ** 2))
        pitch_reward = np.exp(-45 * ((RPY[1] - self.support_plane_estimated_pitch) ** 2))
        yaw_reward = np.exp(-40 * (RPY[2] ** 2))
        height_reward = np.exp(-800 * (desired_height - current_height) ** 2)

        x = pos[0]
        y = pos[1]
        x_l = self._last_base_position[0]
        y_l = self._last_base_position[1]
        self._last_base_position = pos

        step_distance_x = (x - x_l)
        step_distance_y = abs(y - y_l)

        done = self._termination(pos, ori)
        if done:
            reward = 0
        else:
            reward = round(yaw_reward, 4) + round(pitch_reward, 4) + round(roll_reward, 4) \
                     + round(height_reward, 4) + 100 * round(step_distance_x, 4) - 20 * round(step_distance_y, 4)

            '''
            #Penalize for standing at same position for continuous 150 steps
            self.step_disp.append(step_distance_x)

            if(self._n_steps>150):
                if(sum(self.step_disp)<0.035):
                    reward = reward-standing_penalty
            '''

        return reward, done

    def reward(self):
        pos, ori = self.GetBasePosAndOrientation()
        rpy_orig = self._pybullet_client.getEulerFromQuaternion(ori)
        rpy = np.round(rpy_orig, 4)
        x_reward = pos[0] - self._last_base_position[0]
        y_reward = -np.abs(pos[1])
        roll_reward = -np.abs(np.degrees(rpy[0]))
        pitch_reward = -np.abs(np.degrees(rpy[1]))
        yaw_reward = -np.abs(np.degrees(rpy[2]))

        done = self._termination(pos, ori)

        if done:
            reward = -20
        else:
            reward = 2 * x_reward + y_reward + roll_reward + pitch_reward + yaw_reward

        return reward, done

    def apply_pd_control(self, motor_commands):
        motor_vel_commands = np.zeros(12)
        qpos_act = self.GetMotorAngles()
        qvel_act = self.GetMotorVelocities()
        applied_motor_torque = self._kp * (motor_commands - qpos_act) + self._kd * (motor_vel_commands - qvel_act)
        applied_motor_torque = np.clip(np.array(applied_motor_torque), -self.clips, self.clips)
        applied_motor_torque = applied_motor_torque.tolist()

        for motor_id, motor_torque in zip(self._motor_id_list, applied_motor_torque):
            self.SetMotorTorqueById(motor_id, motor_torque)
        return applied_motor_torque

    @staticmethod
    def add_noise(sensor_value, SD=0.04):
        """
        Adds sensor noise of user defined standard deviation in current sensor_value
        """
        noise = np.random.normal(0, SD, 1)
        sensor_value = sensor_value + noise[0]
        return sensor_value

    def GetObservation(self):
        '''
        This function returns the current observation of the environment for the interested task
        Ret:
            obs : [R(t-2), P(t-2), Y(t-2), R(t-1), P(t-1), Y(t-1), R(t), P(t), Y(t), estimated support plane (roll, pitch) ]
        '''
        # motor_angles = np.array(self.GetMotorAngles(), dtype=np.float32)
        # motor_velocities = np.array(self.GetMotorVelocities(), dtype=np.float32)
        pos, ori = self.GetBasePosAndOrientation()
        RPY = self._pybullet_client.getEulerFromQuaternion(ori)
        RPY = np.round(RPY, 5)

        for val in RPY:
            if self.add_IMU_noise:
                val = self.add_noise(val)
            self.ori_history_queue.append(val)

        obs = np.concatenate(
            (self.ori_history_queue, [self.support_plane_estimated_roll, self.support_plane_estimated_pitch])).ravel()

        return obs

    def estimate_terrain(self):
        contact_info = self.get_foot_contacts()
        pos, ori = self.GetBasePosAndOrientation()
        rot_mat = self._pybullet_client.getMatrixFromQuaternion(ori)
        rot_mat = np.array(rot_mat)
        rot_mat = np.reshape(rot_mat, (3, 3))

        (plane_normal,
         self.support_plane_estimated_roll,
         self.support_plane_estimated_pitch) = normal_estimator.vector_method_solo12(self.prev_incline_vec,
                                                                                     contact_info,
                                                                                     self.GetMotorAngles(),
                                                                                     rot_mat)
        self.prev_incline_vec = plane_normal



    def SetWedgeFriction(self, friction):
        """
        This function modify friction coefficient of the wedge
        Args :
        foot_friction :  desired friction coefficient of the wedge
        """
        self._pybullet_client.changeDynamics(
            self.wedge, -1, lateralFriction=friction)

    def SetMotorTorqueById(self, motor_id, torque):
        """
        function to set motor torque for respective motor_id
        """
        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.solo12,
            jointIndex=motor_id,
            controlMode=self._pybullet_client.TORQUE_CONTROL,
            force=torque)


    def ResetLeg(self):
        """
        function to reset hip and knee joints' state
        Args:
             # leg_id 		  : denotes leg index
             # add_constraint   : bool to create constraints in lower joints of five bar leg mechanisim
             # standstilltorque : value of initial torque to set in hip and knee motors for standing condition
        """
        self._pybullet_client.resetJointState(self.solo12,
                                              self._joint_name_to_id["motor_hip_fl"],
                                              targetValue=-0.7, targetVelocity=0)
        self._pybullet_client.resetJointState(self.solo12,
                                              self._joint_name_to_id["motor_knee_fl"],
                                              targetValue=1.4, targetVelocity=0)
        self._pybullet_client.resetJointState(self.solo12,
                                              self._joint_name_to_id["motor_hip_fr"],
                                              targetValue=-0.7, targetVelocity=0)
        self._pybullet_client.resetJointState(self.solo12,
                                              self._joint_name_to_id["motor_knee_fr"],
                                              targetValue=1.4, targetVelocity=0)
        self._pybullet_client.resetJointState(self.solo12,
                                              self._joint_name_to_id["motor_hip_hl"],
                                              targetValue=0.7, targetVelocity=0)
        self._pybullet_client.resetJointState(self.solo12,
                                              self._joint_name_to_id["motor_knee_hl"],
                                              targetValue=-1.4, targetVelocity=0)
        self._pybullet_client.resetJointState(self.solo12,
                                              self._joint_name_to_id["motor_hip_hr"],
                                              targetValue=0.7, targetVelocity=0)
        self._pybullet_client.resetJointState(self.solo12,
                                              self._joint_name_to_id["motor_knee_hr"],
                                              targetValue=-1.4, targetVelocity=0)

        if self.pd_control_enabled:
            self._pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                        jointIndex=self._joint_name_to_id["motor_hip_fl"],
                                                        controlMode=self._pybullet_client.VELOCITY_CONTROL,
                                                        force=0, targetVelocity=0)
            self._pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                        jointIndex=self._joint_name_to_id["motor_knee_fl"],
                                                        controlMode=self._pybullet_client.VELOCITY_CONTROL,
                                                        force=0, targetVelocity=0)
            self._pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                        jointIndex=self._joint_name_to_id["motor_hip_fr"],
                                                        controlMode=self._pybullet_client.VELOCITY_CONTROL,
                                                        force=0, targetVelocity=0)
            self._pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                        jointIndex=self._joint_name_to_id["motor_knee_fr"],
                                                        controlMode=self._pybullet_client.VELOCITY_CONTROL,
                                                        force=0, targetVelocity=0)
            self._pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                        jointIndex=self._joint_name_to_id["motor_hip_hl"],
                                                        controlMode=self._pybullet_client.VELOCITY_CONTROL,
                                                        force=0, targetVelocity=0)
            self._pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                        jointIndex=self._joint_name_to_id["motor_knee_hl"],
                                                        controlMode=self._pybullet_client.VELOCITY_CONTROL,
                                                        force=0, targetVelocity=0)
            self._pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                        jointIndex=self._joint_name_to_id["motor_hip_hr"],
                                                        controlMode=self._pybullet_client.VELOCITY_CONTROL,
                                                        force=0, targetVelocity=0)
            self._pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                        jointIndex=self._joint_name_to_id["motor_knee_hr"],
                                                        controlMode=self._pybullet_client.VELOCITY_CONTROL,
                                                        force=0, targetVelocity=0)
        else:
            self.set_desired_motor_angle_by_name("motor_hip_fl", desired_angle=-0.7)
            self.set_desired_motor_angle_by_name("motor_knee_fl", desired_angle=1.4)

            self.set_desired_motor_angle_by_name("motor_hip_fr", desired_angle=-0.7)
            self.set_desired_motor_angle_by_name("motor_knee_fr", desired_angle=1.4)

            self.set_desired_motor_angle_by_name("motor_hip_hl", desired_angle=0.7)
            self.set_desired_motor_angle_by_name("motor_knee_hl", desired_angle=-1.4)

            self.set_desired_motor_angle_by_name("motor_hip_hr", desired_angle=0.7)
            self.set_desired_motor_angle_by_name("motor_knee_hr", desired_angle=-1.4)

    def ResetPoseForAbd(self):
        '''
        Reset initial conditions of abduction joints
        '''
        self._pybullet_client.resetJointState(self.solo12,
                                              self._joint_name_to_id["motor_abduction_fl"],
                                              targetValue=0, targetVelocity=0)
        self._pybullet_client.resetJointState(self.solo12,
                                              self._joint_name_to_id["motor_abduction_fr"],
                                              targetValue=0, targetVelocity=0)
        self._pybullet_client.resetJointState(self.solo12,
                                              self._joint_name_to_id["motor_abduction_hl"],
                                              targetValue=0, targetVelocity=0)
        self._pybullet_client.resetJointState(self.solo12,
                                              self._joint_name_to_id["motor_abduction_hr"],
                                              targetValue=0, targetVelocity=0)
        if self.pd_control_enabled:
            self._pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                        jointIndex=self._joint_name_to_id["motor_abduction_fl"],
                                                        controlMode=self._pybullet_client.VELOCITY_CONTROL,
                                                        force=0, targetVelocity=0)
            self._pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                        jointIndex=self._joint_name_to_id["motor_abduction_fr"],
                                                        controlMode=self._pybullet_client.VELOCITY_CONTROL,
                                                        force=0, targetVelocity=0)
            self._pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                        jointIndex=self._joint_name_to_id["motor_abduction_hl"],
                                                        controlMode=self._pybullet_client.VELOCITY_CONTROL,
                                                        force=0, targetVelocity=0)
            self._pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                        jointIndex=self._joint_name_to_id["motor_abduction_hr"],
                                                        controlMode=self._pybullet_client.VELOCITY_CONTROL,
                                                        force=0, targetVelocity=0)
        else:
            self.set_desired_motor_angle_by_name("motor_abduction_fl", desired_angle=0)
            self.set_desired_motor_angle_by_name("motor_abduction_fr", desired_angle=0)
            self.set_desired_motor_angle_by_name("motor_abduction_hl", desired_angle=0)
            self.set_desired_motor_angle_by_name("motor_abduction_hr", desired_angle=0)
