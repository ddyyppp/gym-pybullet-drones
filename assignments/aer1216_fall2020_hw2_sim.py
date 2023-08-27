"""Simulation script for assignment 2.

The script uses the control defined in file `aer1216_fall2020_hw2_ctrl.py`.

Example
-------
To run the simulation, type in a terminal:

    $ python aer1216_fall2020_hw2_sim.py

"""
import time
import random
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
import cv2
#### Uncomment the following 2 lines if "module gym_pybullet_drones cannot be found"
#import sys
#sys.path.append('../')

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.enums import DroneModel
from aer1216_fall2020_hw2_ctrl import HW2Control
from aer1216_fall2020_hw2_ctrl import MAPPO

DURATION = 2000
"""int: The duration of the simulation in seconds."""
GUI = True
"""bool: Whether to use PyBullet graphical interface."""
RECORD = False
"""bool: Whether to save a video under /files/videos. Requires ffmpeg"""

if __name__ == "__main__":
    frames = []
    #### Create the ENVironment ################################

    ENV = CtrlAviary(num_drones=2,
                     drone_model=DroneModel.CF2P,
                     initial_xyzs=np.array([ [.0, -.2, 1.0] , [0.0,0.2,1.0] ]),
                     gui=GUI,
                     record=RECORD,
                     freq = 480
                     )

    PYB_CLIENT = ENV.getPyBulletClient()

    #### Initialize the LOGGER #################################
    LOGGER = Logger(logging_freq_hz=ENV.SIM_FREQ,num_drones=2,)



    ppo_agent = MAPPO(state_dim = 25, action_dim = 11,agent_num=2, lr_actor = 0.001, lr_critic = 0.001, gamma=0.95, \
                    K_epochs = 15, eps_clip = 0.3, action_std_init = 0.01)
    #ppo_agent.load('Test2.pt')
    #### Initialize the ACTION #################################
    ACTION = {}
    ACTION_sim = {}
    OBS = ENV.reset()

    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=90, cameraPitch=0, cameraTargetPosition=[0, 0, 1])

    TargetSTATE = np.zeros(3)
    TargetSTATE[0] = 1
    TargetSTATE[1] = 0
    TargetSTATE[2] = 2
    STATE = OBS["0"]["state"]
    quaternion = STATE[3:7]
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    rotation = rotation_matrix.reshape(-1)
   # STATE1 = np.hstack((STATE[0:3], STATE[10:13], STATE[13:16] * 2 * np.pi / 360, rotation, np.zeros(7))) - np.hstack(
    #    (TargetSTATE, np.zeros(16)))
    STATE0 = np.hstack(
        (STATE[0:3], STATE[10:16], rotation, np.zeros(7))) - np.hstack(
        (TargetSTATE, np.zeros(22)))

    STATE = OBS["1"]["state"]
    quaternion = STATE[3:7]
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    rotation = rotation_matrix.reshape(-1)
    STATE1 = np.hstack(
        (STATE[0:3], STATE[10:16], rotation, np.zeros(7))) - np.hstack(
        (TargetSTATE, np.zeros(22)))
    obs_states = np.hstack((STATE0, STATE1))
    ACTION["0"] = ppo_agent.select_action(STATE0,obs_states,0)
    ACTION["1"] = ppo_agent.select_action(STATE1,obs_states,1)

    #### Run the simulation ####################################
    START = time.time()
    rp_old0 = 0
    rp_old1 = 0
    for i in range(0, DURATION*ENV.SIM_FREQ):


        #### Step the simulation ###################################
        ACTION_sim["0"] = (ACTION["0"][:4]+1.05)*ENV.MAX_RPM/2.1
        ACTION_sim["1"] = (ACTION["1"][:4]+1.05)*ENV.MAX_RPM/2.1
        OBS, _, _, _ = ENV.step(ACTION_sim)
        STATE = OBS["0"]["state"]
        #rewards = 20-np.linalg.norm(STATE1[:6])
        #rp = (STATE[:3]-StartSTATE).dot(TargetSTATE-StartSTATE)/np.linalg.norm(TargetSTATE-StartSTATE)
        rp = -np.linalg.norm(TargetSTATE-STATE[:3])
        rewards = rp - rp_old0 - 0.05 * (np.linalg.norm(STATE[13:16]))
        #rp_old0 = rp
        if STATE[0] > 10:
            rewards -= np.exp(min((STATE[0]-10),500))
        if STATE[1] > 10:
            rewards -= np.exp(min((STATE[1]-10),500))
        if STATE[2] > 10:
            rewards -= np.exp(min((STATE[2]-10),500))
        if STATE[0] < -10:
            rewards -= np.exp(min(-(STATE[0]+10),500))
        if STATE[1] < -10:
            rewards -= np.exp(min(-(STATE[1]+10),500))
        if STATE[2] < -10:
            rewards -= np.exp(min(-(STATE[2]+10),500))
        ppo_agent.buffer[0].rewards.append(rewards)


        if i % (ENV.SIM_FREQ*2) == ENV.SIM_FREQ*2-1:
            ppo_agent.buffer[0].is_terminals.append(True)

        else:
            ppo_agent.buffer[0].is_terminals.append(False)
        if i % (ENV.SIM_FREQ/10) == ENV.SIM_FREQ/10-1:
            print("rewards0", rewards)


        quaternion = STATE[3:7]
        rotation = R.from_quat(quaternion)
        rotation_matrix = rotation.as_matrix()
        rotation = rotation_matrix.reshape(-1)
        #STATE1 = np.hstack((STATE[0:3], STATE[10:13],STATE[13:16]*2*np.pi/360, rotation, np.zeros(7))) - np.hstack((TargetSTATE, np.zeros(16)))
        STATE0 = np.hstack(
            (STATE[0:3], STATE[10:16], rotation, np.zeros(7))) - np.hstack(
            (TargetSTATE, np.zeros(22)))

        STATE = OBS["1"]["state"]
        # rewards = 20-np.linalg.norm(STATE1[:6])
        # rp = (STATE[:3]-StartSTATE).dot(TargetSTATE-StartSTATE)/np.linalg.norm(TargetSTATE-StartSTATE)
        rp = -np.linalg.norm(TargetSTATE - STATE[:3])
        rewards = rp - rp_old1 - 0.05 * (np.linalg.norm(STATE[13:16]))
        #rp_old1 = rp
        if STATE[0] > 10:
            rewards -= np.exp(min((STATE[0] - 10), 500))
        if STATE[1] > 10:
            rewards -= np.exp(min((STATE[1] - 10), 500))
        if STATE[2] > 10:
            rewards -= np.exp(min((STATE[2] - 10), 500))
        if STATE[0] < -10:
            rewards -= np.exp(min(-(STATE[0] + 10), 500))
        if STATE[1] < -10:
            rewards -= np.exp(min(-(STATE[1] + 10), 500))
        if STATE[2] < -10:
            rewards -= np.exp(min(-(STATE[2] + 10), 500))
        ppo_agent.buffer[1].rewards.append(rewards)

        if i % (ENV.SIM_FREQ * 2) == ENV.SIM_FREQ * 2 - 1:
            ppo_agent.buffer[1].is_terminals.append(True)
            ENV.render()
            OBS = ENV.reset()
        else:
            ppo_agent.buffer[1].is_terminals.append(False)
        if i % (ENV.SIM_FREQ / 10) == ENV.SIM_FREQ / 10 - 1:
            ppo_agent.train()
            print("rewards1", rewards)

        quaternion = STATE[3:7]
        rotation = R.from_quat(quaternion)
        rotation_matrix = rotation.as_matrix()
        rotation = rotation_matrix.reshape(-1)
        # STATE1 = np.hstack((STATE[0:3], STATE[10:13],STATE[13:16]*2*np.pi/360, rotation, np.zeros(7))) - np.hstack((TargetSTATE, np.zeros(16)))
        STATE1 = np.hstack(
            (STATE[0:3], STATE[10:16], rotation, np.zeros(7))) - np.hstack(
            (TargetSTATE, np.zeros(22)))
        #STATE1 = np.hstack((STATE[0:3], STATE[10:13], STATE[7:10],  np.zeros(7)))
        obs_states = np.hstack((STATE0,STATE1))
        ACTION["0"] = ppo_agent.select_action(STATE0, obs_states, 0)
        ACTION["1"] = ppo_agent.select_action(STATE1, obs_states, 1)




        #LOGGER.log(drone=0, timestamp=i/ENV.SIM_FREQ, state=STATE)


        #### Sync the simulation ###################################
        if GUI:
            sync(i, START, ENV.TIMESTEP)

    #### Close the ENVironment #################################
    ENV.close()

    #### Save the simulation results ###########################
    #LOGGER.save()
    ppo_agent.save('Test3.pt')
    #### Plot the simulation results ###########################
    #LOGGER.plot()
