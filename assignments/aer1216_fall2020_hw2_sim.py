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

#### Uncomment the following 2 lines if "module gym_pybullet_drones cannot be found"
import sys
sys.path.append('../')

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.enums import DroneModel
from aer1216_fall2020_hw2_ctrl import HW2Control

DURATION = 3
"""int: The duration of the simulation in seconds."""
GUI = True
"""bool: Whether to use PyBullet graphical interface."""
RECORD = False
"""bool: Whether to save a video under /files/videos. Requires ffmpeg"""

if __name__ == "__main__":

    #### Create the ENVironment ################################

    ENV = CtrlAviary(num_drones=3,
                     drone_model=DroneModel.CF2P,
                     initial_xyzs=np.array([ [.0, -.2, 1.0], [.0, .2, 1.0], [.3, .2, 1.0] ]),
                     gui=GUI,
                     record=RECORD
                     )

    PYB_CLIENT = ENV.getPyBulletClient()

    #### Initialize the LOGGER #################################
    LOGGER = Logger(logging_freq_hz=ENV.SIM_FREQ,
                    num_drones=3,
                    )

    #### Initialize the CONTROLLERS ############################
    CTRL_0 = HW2Control(env=ENV,
                        control_type=0
                        )
    CTRL_1 = HW2Control(env=ENV,
                        control_type=1
                        )
    CTRL_2 = HW2Control(env=ENV,
                        control_type=2
                        )

    #### Initialize the ACTION #################################
    ACTION = {}
    OBS = ENV.reset()
    #p.setCollisionFilterPair(bodyUniqueIdA=ENV.DRONE_IDS[0], bodyUniqueIdB=ENV.DRONE_IDS[1], linkIndexA=-1,
    #                         linkIndexB=-1, enableCollision=True)
    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=90, cameraPitch=0, cameraTargetPosition=[0, 0, 1])
    p.changeDynamics(bodyUniqueId=ENV.DRONE_IDS[0], linkIndex=-1, restitution=0)
    p.changeDynamics(bodyUniqueId=ENV.DRONE_IDS[1], linkIndex=-1, restitution=0)
    p.changeDynamics(bodyUniqueId=ENV.DRONE_IDS[2], linkIndex=-1, restitution=0)
    STATE = OBS["0"]["state"]
    ACTION["0"] = CTRL_0.compute_control(current_position=STATE[0:3],
                                         current_velocity=STATE[10:13],
                                         current_rpy=STATE[7:10],
                                         target_position=STATE[0:3],
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )
    STATE = OBS["1"]["state"]
    ACTION["1"] = CTRL_1.compute_control(current_position=STATE[0:3],
                                         current_velocity=STATE[10:13],
                                         current_rpy=STATE[7:10],
                                         target_position=STATE[0:3],
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )
    STATE = OBS["2"]["state"]
    ACTION["2"] = CTRL_2.compute_control(current_position=STATE[0:3],
                                         current_velocity=STATE[10:13],
                                         current_rpy=STATE[7:10],
                                         target_position=STATE[0:3],
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )

    #### Initialize the target trajectory ######################
    TARGET_POSITION1 = np.array([[0, +(1*i/DURATION/ENV.SIM_FREQ), 1.0] for i in range(DURATION*ENV.SIM_FREQ)])
    TARGET_POSITION2 = np.array([[0, -(1*i/DURATION/ENV.SIM_FREQ), 1.0] for i in range(DURATION * ENV.SIM_FREQ)])
    TARGET_VELOCITY1 = np.zeros([DURATION * ENV.SIM_FREQ, 3])
    TARGET_VELOCITY2 = np.zeros([DURATION * ENV.SIM_FREQ, 3])
    TARGET_ACCELERATION1 = np.zeros([DURATION * ENV.SIM_FREQ, 3])
    TARGET_ACCELERATION2 = np.zeros([DURATION * ENV.SIM_FREQ, 3])

    #### Derive the target trajectory to obtain target velocities and accelerations
    TARGET_VELOCITY1[1:, :] = (TARGET_POSITION1[1:, :] - TARGET_POSITION1[0:-1, :]) / ENV.SIM_FREQ
    TARGET_VELOCITY2[1:, :] = (TARGET_POSITION2[1:, :] - TARGET_POSITION2[0:-1, :]) / ENV.SIM_FREQ
    TARGET_ACCELERATION1[1:, :] = (TARGET_VELOCITY1[1:, :] - TARGET_VELOCITY1[0:-1, :]) / ENV.SIM_FREQ
    TARGET_ACCELERATION2[1:, :] = (TARGET_VELOCITY2[1:, :] - TARGET_VELOCITY2[0:-1, :]) / ENV.SIM_FREQ

    #### Run the simulation ####################################
    START = time.time()
    for i in range(0, DURATION*ENV.SIM_FREQ):

        ### Secret control performance booster #####################
        #if i/ENV.SIM_FREQ>3 and i%30==0 and i/ENV.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [random.gauss(0, 0.3), random.gauss(0, 0.3), 3], p.getQuaternionFromEuler([random.randint(0, 360),random.randint(0, 360),random.randint(0, 360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################

        OBS, _, _, _ = ENV.step(ACTION)

        #### Compute control for drone 0 ###########################
        STATE = OBS["0"]["state"]
        ACTION["0"] = CTRL_0.compute_control(current_position=STATE[0:3],
                                         current_velocity=STATE[10:13],
                                         current_rpy=STATE[7:10],
                                         target_position=TARGET_POSITION1[i, :]+ np.array([-.0, -.2, .0]),
                                         target_velocity=TARGET_VELOCITY1[i, :],
                                         target_acceleration=TARGET_ACCELERATION1[i, :]
                                         )
        #### Log drone 0 ###########################################
        LOGGER.log(drone=0, timestamp=i/ENV.SIM_FREQ, state=STATE)
        #### Compute control for drone 1 ###########################
        STATE = OBS["1"]["state"]
        ACTION["1"] = CTRL_1.compute_control(current_position=STATE[0:3],
                                         current_velocity=STATE[10:13],
                                         current_rpy=STATE[7:10],
                                         target_position=TARGET_POSITION2[i, :] + np.array([-.0, .2, .0]),
                                         target_velocity=TARGET_VELOCITY2[i, :],
                                         target_acceleration=TARGET_ACCELERATION2[i, :]
                                         )
        #### Log drone 1 ###########################################
        LOGGER.log(drone=1, timestamp=i/ENV.SIM_FREQ, state=STATE)
        #### Compute control for drone 2 ###########################
        STATE = OBS["2"]["state"]
        ACTION["2"] = CTRL_2.compute_control(current_position=STATE[0:3],
                                             current_velocity=STATE[10:13],
                                             current_rpy=STATE[7:10],
                                             target_position=TARGET_POSITION2[i, :] + np.array([.3, .2, .0]),
                                             target_velocity=TARGET_VELOCITY2[i, :],
                                             target_acceleration=TARGET_ACCELERATION2[i, :]
                                             )
        #### Log drone 2 ###########################################
        LOGGER.log(drone=2, timestamp=i/ENV.SIM_FREQ, state=STATE)

            #### Printout ##############################################
        if i*10%ENV.SIM_FREQ == 0:
            ENV.render()

        #### Sync the simulation ###################################
        if GUI:
            sync(i, START, ENV.TIMESTEP)

    #### Close the ENVironment #################################
    ENV.close()

    #### Save the simulation results ###########################
    LOGGER.save()

    #### Plot the simulation results ###########################
    LOGGER.plot()
