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
        self.useInverseKinematics = 1
        self.pandaEndEffectorIndex = 7
        self.useNullSpace = 21
        self.useOrientation = 1
        self.useSimulation =1
        self.maxVelocity = 0.35
        self.maxForce = 200
        self.endEffectorAngle
        self.fingerAForce = 2
        self.fingerTipForce = 2
    '''    
    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]

        currentPose = p.getLinkState(self.pandaUid, 11)
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        jointPoses = p.calculateInverseKinematics(self.pandaUid,11,newPosition, orientation)

        p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])

        p.stepSimulation()

        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        if state_object[2]>0.45:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        
        observation = state_robot + state_fingers
        
        info = {"information": state_object}

        return observation, reward, done, info
    '''
   
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
        #cubeHalfExtents = [0.03, 0.03, 0.03]  # Adjust the size of the cube as needed
        #cubeMass = 100
        ## Define the range for random cube position
        #min_position = [0.45, -0.3, 0.05]  # Minimum position [x, y, z] within the table boundaries
        #max_position = [0.65, 0.3, 0.05]  # Maximum position [x, y, z] within the table boundaries

        # Generate random cube position within the specified range
        #cubePosition = [
        #    random.uniform(min_position[0], max_position[0]),
        #    random.uniform(min_position[1], max_position[1]),
        #    random.uniform(min_position[2], max_position[2])
        #    ]
        #cubeOrientation = p.getQuaternionFromEuler([0, 0, 0])  # Adjust the orientation of the cube as needed
        #cubeVisualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=cubeHalfExtents)
        #cubeCollisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=cubeHalfExtents)
        #self.objectUid = p.createMultiBody(
        #    baseMass=cubeMass,
        #    baseCollisionShapeIndex=cubeCollisionShapeId,
        #    baseVisualShapeIndex=cubeVisualShapeId,
        #    basePosition=cubePosition,
        #    baseOrientation=cubeOrientation
        #)
    
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid, 9)[0], p.getJointState(self.pandaUid, 10)[0])

        state_robot_normalized = np.array(state_robot) / 2.0
        state_fingers_normalized = np.array(state_fingers)

        observation = np.concatenate((state_robot_normalized, state_fingers_normalized))

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return observation
    
    def applyAction(self, motorCommands):
        pandaNumDofs = 7
        ll = [-7]*pandaNumDofs
        #upper limits for null space (todo: set them to proper range)
        ul = [7]*pandaNumDofs
        #joint ranges for null space (todo: set them to proper range)
        jr = [7]*pandaNumDofs
        #restposes for null space
        jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
        rp = jointPositions
        if (self.useInverseKinematics):
            dx = motorCommands[0]
            dy = motorCommands[1]
            dz = motorCommands[2]
            da = motorCommands[3]
            fingerAngle = motorCommands[4]
            state = p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)
            actualEndEffectorPos = state[0]

            self.endEffectorPos[0] = self.endEffectorPos[0] + dx
            if (self.endEffectorPos[0] > 0.65):
                self.endEffectorPos[0] = 0.65
            if (self.endEffectorPos[0] < 0.50):
                self.endEffectorPos[0] = 0.50
                self.endEffectorPos[1] = self.endEffectorPos[1] + dy
            if (self.endEffectorPos[1] < -0.17):
                self.endEffectorPos[1] = -0.17
            if (self.endEffectorPos[1] > 0.22):
                self.endEffectorPos[1] = 0.22
            self.endEffectorPos[2] = self.endEffectorPos[2] + dz

            self.endEffectorAngle = self.endEffectorAngle + da
            pos = self.endEffectorPos
            orn = p.getQuaternionFromEuler([0, -math.pi, 0])
            
            jointPoses = p.calculateInverseKinematics(self.pandaUid, self.pandaEndEffectorIndex, pos,orn)
            if (self.useSimulation):
                for i in range(self.pandaEndEffectorIndex + 1):
                    #print(i)
                    p.setJointMotorControl2(bodyUniqueId=self.pandaUid,
                                  jointIndex=i,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=jointPoses[i],
                                  targetVelocity=0,
                                  force=self.maxForce,
                                  maxVelocity=self.maxVelocity,
                                  positionGain=0.3,
                                  velocityGain=1)
            else:
                #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
                for i in range(self.numJoints):
                    p.resetJointState(self.pandaUid, i, jointPoses[i])
            #fingers
            p.setJointMotorControl2(self.pandaUid,
                              7,
                              p.POSITION_CONTROL,
                              targetPosition=self.endEffectorAngle,
                              force=self.maxForce)
            p.setJointMotorControl2(self.pandaUid,
                              8,
                              p.POSITION_CONTROL,
                              targetPosition=-fingerAngle,
                              force=self.fingerAForce)
            p.setJointMotorControl2(self.pandaUid,
                              11,
                              p.POSITION_CONTROL,
                              targetPosition=fingerAngle,
                              force=self.fingerBForce)

            p.setJointMotorControl2(self.pandaUid,
                              10,
                              p.POSITION_CONTROL,
                              targetPosition=0,
                              force=self.fingerTipForce)
            p.setJointMotorControl2(self.pandaUid,
                              13,
                              p.POSITION_CONTROL,
                              targetPosition=0,
                              force=self.fingerTipForce)
        else:
            for action in range(len(motorCommands)):
                motor = self.motorIndices[action]
                p.setJointMotorControl2(self.pandaUid,
                                motor,
                                p.POSITION_CONTROL,
                                targetPosition=motorCommands[action],
                                force=self.maxForce)

        return

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