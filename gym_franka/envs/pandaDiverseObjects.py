import os
import gym
import numpy as np
import pybullet as p
import pybullet_data
import time
from . import panda_env
import gym_franka
import glob
import pdb
import distutils.dir_util
from pkg_resources import parse_version
from gym import spaces

class PandaDiverseObject(gym_franka):
    """A class for Franka Emika Panda environment with diverse objects.

        In each episode some objects are chosen from a set of 1000 diverse objects.
        These 1000 objects are split 90/10 into a train and test set."""
    
    def __init__(self,
                urdfRoot = pybullet_data.getDataPath(), 
                actionRepeat=80,
                isEnableSelfCollision=True,
                renders=False,
                isDiscrete=False,
                maxSteps=8,
                dv=0.06,
                removeHeightHack=False,
                blockRandom=0.3,
                cameraRandom=0,
                width=48,
                height=48,
                numObjects=5,
                isTest=False):
        
        self._isDiscrete = isDiscrete
        self._timeStep = 1. / 240.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._dv = dv
        self._p = p
        self._removeHeightHack = removeHeightHack
        self._blockRandom = blockRandom
        self._cameraRandom = cameraRandom
        self._width = width
        self._height = height
        self._numObjects = numObjects
        self._isTest = isTest
    
        if self._renders:
            self.cid = p.connect(p.SHARED_MEMORY)
            if (self.cid < 0):
                self.cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
            self.cid = p.connect(p.DIRECT)
        self.seed()

        if (self._isDiscrete):
            if self._removeHeightHack:
                self.action_space = spaces.Discrete(9)
            else:
                self.action_space = spaces.Discrete(7)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(3,))  # dx, dy, da
            if self._removeHeightHack:
                self.action_space = spaces.Box(low=-1, high=1, shape=(4,))  # dx, dy, dz, da
        self.observation_space = spaces.Box(low=0, high=255, shape=(self._height,
                                                                self._width,
                                                                3))
        self.viewer = None

