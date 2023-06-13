import gym
from gym  import error, spaces, utils
from gym.utils import seeding
import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random

class PandaEnv(gym.Env):
    metadata = {'render_modes': ['human'],  "render_fps": 4}

    def __init__(self):
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])
        self.action_space = spaces.Box(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1]))
        self.observation_space = spaces.Box(np.array([-1, -1, -1, -1, -1, -1, -1, -1]), np.array([1, 1, 1, 1, 1, 1, 1, 1]))
        self.pandaUid = None
        self.objectUid = None
     
    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
        dv = 0.05
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]

        currentPose = p.getLinkState(self.pandaUid, 11)
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        #jointPoses = p.calculateInverseKinematics(self.pandaUid,11,newPosition, orientation)
        num_joints = p.getNumJoints(self.pandaUid)
        dof = p.getNumJoints(self.pandaUid) - 1
        joints = list(range(dof))
        joint_states = p.getJointStates(self.pandaUid, joints)
        joint_positions = [state[0] for state in joint_states]
        #for jointIndex in range(num_joints):
        #    p.resetJointState(self.pandaUid, jointIndex, self.joint_positions[jointIndex])
        #    p.setJointMotorControl2(self.pandaUid, jointIndex, p.POSITION_CONTROL,
        #                            targetPosition=joint_positions[jointIndex], force=1)
        # Calculate the desired position for the robot's end effector based on the object's position
        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        desired_position = [state_object[0], state_object[1], state_object[2] + 0.1]  # Adjust the z-coordinate as needed

        # Calculate the joint poses for the robot's arm to reach the desired end effector position
        joint_poses = p.calculateInverseKinematics(self.pandaUid, 11, desired_position)

        # Set the joint positions of the robot's arm to move towards the desired position
        p.setJointMotorControlArray(bodyUniqueId=self.pandaUid,
                                    jointIndices=list(range(len(joint_poses))),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_poses)
        for _ in range(10):
            p.stepSimulation()

        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        if state_object[2]>0.01:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        info = {"information":state_object}

        #observation = np.array(state_robot + state_fingers)
        state_robot_normalized = np.array(state_robot) / 2.0  # Normalize robot state values between -1 and 1
        state_fingers_normalized = np.array(state_fingers)  # Fingers state is already within [-1, 1]

        observation = np.concatenate((state_robot_normalized, state_fingers_normalized))
        
        return observation, reward, done, info
   
    
    def reset(self):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setGravity(0, 0, -9.8)
        urdfRootPath = pybullet_data.getDataPath()

        planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])

        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),flags=p.URDF_USE_SELF_COLLISION,  useFixedBase=True)
        for i in range(7):
            p.resetJointState(self.pandaUid, i, rest_poses[i])

        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])

        trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"), basePosition=[0.45, 0, 0])
        self.endEffectorPos = [0.537, 0.0, 0.5]
        self.endEffectorAngle=0
        state_object= [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.05]
        self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid, 9)[0], p.getJointState(self.pandaUid, 10)[0])

        state_robot_normalized = np.array(state_robot) / 2.0
        state_fingers_normalized = np.array(state_fingers)

        observation = np.concatenate((state_robot_normalized, state_fingers_normalized))

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return observation
    
    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array
    def close(self):
        p.disconnect()