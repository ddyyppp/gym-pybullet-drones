import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import tkinter as tk
from ENV import Env
from MAPPO import MAPPO


device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()


def message_transform(drones, ppo, states):
    action = {}
    obs_states = states.reshape(-1)
    processed_drones = set()
    sum_drone = 0
    sum_drone_old = -1
    while sum_drone < len(drones):

        if sum_drone == sum_drone_old:
            for i, drone in enumerate(drones):
                if i in processed_drones:
                    continue
                message = np.zeros((12))
                state = np.hstack((states[i, :], message))
                action[i] = ppo.select_action(state, obs_states, i)
                sum_drone += 1
                processed_drones.add(i)
                drone.message = action[i][-12:]
                break
        else:
            sum_drone_old = sum_drone

        for i, drone in enumerate(drones):
            if i in processed_drones:
                continue
            if drone.link == -1:
                message = np.zeros((12))
                state = np.hstack((states[i,:],message))
                action[i] = ppo.select_action(state, obs_states, i)
                sum_drone += 1
                processed_drones.add(i)
                drone.message = action[i][-12:]
            else:
                if drones[drone.link].message is not None:
                    message = drones[drone.link].message
                    state = np.hstack((states[i, :], message))
                    action[i] = ppo.select_action(state, obs_states, i)
                    sum_drone += 1
                    processed_drones.add(i)
                    drone.message = action[i][-12:]
    for i, drone in enumerate(drones):
        drone.message = None
    return action




if __name__ == "__main__":
    num_drones_per_side = 3
    spacing = 0.2

    initial_positions = np.array([
        [i * spacing, j * spacing]
        for i in range(num_drones_per_side)
        for j in range(num_drones_per_side)
    ]) - spacing * (num_drones_per_side - 1) / 2
    env = Env(num_drones=9,
                 initial_xyzs=initial_positions,
                 record=False,
                 )

    ppo_agent = MAPPO(state_dim = 19, action_dim = 16, obs_dim = 63, agent_num=9, lr_actor = 0.001, lr_critic = 0.001, gamma=0.95,
                    K_epochs = 15, eps_clip = 0.3, action_std_init = 0.01)
    ppo_agent.load('Test.pt')
    time = 0
    DURATION = 300
    obs = env.reset(initial_positions)
    SIM_FREQ = 240
    for i in range(0, DURATION*SIM_FREQ):
        Action = message_transform(env.drones, ppo_agent, obs)
        env.link_step(Action)
        obs, rewards, done = env.action_step(Action)
        for j in range(env.num_drones):
            ppo_agent.buffer[j].rewards.append(rewards)
            if i % (SIM_FREQ * 2) == SIM_FREQ * 2 - 1:
                ppo_agent.buffer[j].is_terminals.append(True)
            else:
                ppo_agent.buffer[j].is_terminals.append(False)
        if i % (SIM_FREQ * 2) == SIM_FREQ * 2 - 1:
            initial_positions = np.array([
                [i * spacing, j * spacing]
                for i in range(num_drones_per_side)
                for j in range(num_drones_per_side)
            ]) - spacing * (num_drones_per_side - 1) / 2
            obs = env.reset(initial_positions)
        if i % (SIM_FREQ / 20) == SIM_FREQ / 20 - 1:
            ppo_agent.train()
            print("i", i, "rewards", rewards)

        env.render()

    ppo_agent.save('Test.pt')
