"""Control implementation for assignment 2.

The controller used the simulation in file `aer1216_fall2020_hw2_sim.py`.

Example
-------
To run the simulation, type in a terminal:

    $ python aer1216_fall2020_hw2_sim.py

Notes
-----
To-dos
    Search for word "Objective" this file (there are 4 occurrences)
    Fill appropriate values in the 3 by 3 matrix self.matrix_u2rpm.
    Compute u_1 for the linear controller and the second nonlinear one
    Compute u_2

"""
import numpy as np
from gym_pybullet_drones.envs.BaseAviary import BaseAviary


import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()

class HW2Control():
    """Control class for assignment 2."""

    ################################################################################

    def __init__(self,
                 env: BaseAviary,
                 control_type: int=0
                 ):
        """ Initialization of class HW2Control.

        Parameters
        ----------
        env : BaseAviary
            The PyBullet-based simulation environment.
        control_type : int, optional
            Choose between implementation of the u1 computation.

        """
        self.g = env.G
        """float: Gravity acceleration, in meters per second squared."""
        self.mass = env.M
        """float: The mass of quad from environment."""
        self.inertia_xx = env.J[0][0]
        """float: The inertia of quad around x axis."""
        self.arm_length = env.L
        """float: The inertia of quad around x axis."""
        self.timestep = env.TIMESTEP
        """float: Simulation and control timestep."""
        self.last_rpy = np.zeros(3)
        """ndarray: Store the last roll, pitch, and yaw."""
        self.kf_coeff = env.KF
        """float: RPMs to force coefficient."""
        self.km_coeff = env.KM
        """float: RPMs to torque coefficient."""
        self.CTRL_TYPE = control_type
        """int: Flag switching beween implementations of u1."""
        self.p_coeff_position = {}
        """dict[str, float]: Proportional coefficient(s) for position control."""
        self.d_coeff_position = {}
        """dict[str, float]: Derivative coefficient(s) for position control."""

        ############################################################
        ############################################################
        #### HOMEWORK CODE (START) #################################
        ############################################################
        ############################################################

        # Objective 1 of 4: fill appropriate values in the 3 by 3 matrix
        self.matrix_u2rpm = np.array([ [2,   1,   1],
                                       [0,   1,  -1],
                                       [2,  -1,  -1] 
                                      ])
        """ndarray: (3, 3)-shaped array of ints to determine motor rpm from force and torque."""

        ############################################################
        ############################################################
        #### HOMEWORK CODE (END) ###################################
        ############################################################
        ############################################################

        self.matrix_u2rpm_inv = np.linalg.inv(self.matrix_u2rpm)

        self.p_coeff_position["z"] = 0.7 * 0.7
        self.d_coeff_position["z"] = 2 * 0.5 * 0.7
        #
        self.p_coeff_position["y"] = 0.7 * 0.7
        self.d_coeff_position["y"] = 2 * 0.5 * 0.7
        #
        self.p_coeff_position["r"] = 0.7 * 0.7
        self.d_coeff_position["r"] = 2 * 2.5 * 0.7

        self.reset()

    ################################################################################

    def reset(self):
        """ Resets the controller counter."""
        self.control_counter = 0

    ################################################################################

    def compute_control(self,
                        current_position,
                        current_velocity,
                        current_rpy,
                        target_position,
                        target_velocity=np.zeros(3),
                        target_acceleration=np.zeros(3),
                        ):
        """Computes the propellers' RPMs for the target state, given the current state.

        Parameters
        ----------
        current_position : ndarray
            (3,)-shaped array of floats containing global x, y, z, in meters.
        current_velocity : ndarray
            (3,)-shaped array of floats containing global vx, vy, vz, in m/s.
        current_rpy : ndarray
            (3,)-shaped array of floats containing roll, pitch, yaw, in rad.
        target_position : ndarray
            (3,)-shaped array of float containing global x, y, z, in meters.
        target_velocity : ndarray, optional
            (3,)-shaped array of floats containing global, in m/s.
        target_acceleration : ndarray, optional
            (3,)-shaped array of floats containing global, in m/s^2.

        Returns
        -------
        ndarray
            (4,)-shaped array of ints containing the desired RPMs of each propeller.
        """
        self.control_counter += 1

        #### Compute roll, pitch, and yaw rates ####################
        current_rpy_dot = (current_rpy - self.last_rpy) / self.timestep

        ##### Calculate PD control in y, z #########################
        y_ddot = self.pd_control(target_position[1],
                                 current_position[1],
                                 target_velocity[1],
                                 current_velocity[1],
                                 target_acceleration[1],
                                 "y"
                                 )
        z_ddot = self.pd_control(target_position[2],
                                 current_position[2],
                                 target_velocity[2],
                                 current_velocity[2],
                                 target_acceleration[2],
                                 "z"
                                 )

        ##### Calculate desired roll and rates given by PD #########
        desired_roll = -y_ddot / self.g
        desired_roll_dot = (desired_roll - current_rpy[0]) / 0.004
        self.old_roll = desired_roll
        self.old_roll_dot = desired_roll_dot
        roll_ddot = self.pd_control(desired_roll, 
                                    current_rpy[0],
                                    desired_roll_dot, 
                                    current_rpy_dot[0],
                                    0,
                                    "r"
                                    )

        ############################################################
        ############################################################
        #### HOMEWORK CODE (START) #################################
        ############################################################
        ############################################################
        
        # Variables that you might use
        #   self.g
        #   self.mass
        #   self.inertia_xx
        #   y_ddot
        #   z_ddot
        #   roll_ddot
        #   current_rpy[0], current_rpy[1], current_rpy[2]
        # Basic math and NumPy
        #   sine(x) -> np.sin(x)
        #   cosine(x) -> np.cos(x)
        #   x squared -> x**2
        #   square root of x -> np.sqrt(x)

        ##### Calculate thrust and moment given the PD input #######
        if self.CTRL_TYPE == 0:
            #### Linear Control ########################################
            # Objective 2 of 4: compute u_1 for the linear controller
            u_1 = self.mass * (self.g + z_ddot)

        elif self.CTRL_TYPE == 1:
            #### Nonlinear Control 1 ###################################
            u_1 = self.mass * (self.g + z_ddot) / np.cos(current_rpy[0])

        elif self.CTRL_TYPE == 2:
            #### Nonlinear Control 2 ###################################
            # Objective 3 of 4: compute u_1 for the second nonlinear controller
            u_1 = self.mass * np.sqrt(y_ddot**2+(self.g + z_ddot)**2)

        # Objective 4 of 4: compute u_2
        u_2 = self.inertia_xx * roll_ddot

        ############################################################
        ############################################################
        #### HOMEWORK CODE (END) ###################################
        ############################################################
        ############################################################
        
        ##### Calculate RPMs #######################################
        u = np.array([ [u_1 / self.kf_coeff],
                       [u_2 / (self.arm_length*self.kf_coeff)],
                       [0] ])
        propellers_rpm = np.dot(self.matrix_u2rpm_inv, u)

        #### Command the turn rates of propellers 1 and 3 ##########
        if (propellers_rpm[1, 0]) <= 0:
            propellers_1_rpm = 0
        else:
            propellers_1_rpm = np.sqrt(propellers_rpm[1, 0])
        if (propellers_rpm[2, 0]) <= 0:
            propellers_3_rpm = 0
        else:
            propellers_3_rpm = np.sqrt(propellers_rpm[2, 0])

        #### For motion in the Y-Z plane, assign the same turn rates to prop. 0 and 2
        if propellers_rpm[0, 0] <= 0:
            propellers_0_and_2_rpm = 0
        else:
            propellers_0_and_2_rpm = np.sqrt(propellers_rpm[0, 0])

        #### Print relevant output #################################
        if self.control_counter%(1/self.timestep) == 0:
            print("current_position", current_position)
            print("current_velocity", current_velocity)
            print("target_position", target_position)
            print("target_velocity", target_velocity)
            print("target_acceleration", target_acceleration)

        #### Store the last step's roll, pitch, and yaw ############
        self.last_rpy = current_rpy

        return np.array([propellers_0_and_2_rpm, propellers_1_rpm,
                         propellers_0_and_2_rpm, propellers_3_rpm])

    ################################################################################

    def pd_control(self,
                   desired_position,
                   current_position,
                   desired_velocity,
                   current_velocity,
                   desired_acceleration,
                   opt
                   ):
        """Computes PD control for the acceleration minimizing position error.

        Parameters
        ----------
        desired_position :
            float: Desired global position.
        current_position :
            float: Current global position.
        desired_velocity :
            float: Desired global velocity.
        current_velocity :
            float: Current global velocity.
        desired_acceleration :
            float: Desired global acceleration.

        Returns
        -------
        float
            The commanded acceleration.
        """
        u = desired_acceleration + \
            self.d_coeff_position[opt] * (desired_velocity - current_velocity) + \
            self.p_coeff_position[opt] * (desired_position - current_position)

        return u






import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class Buffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,) , action_std_init ** 2).to(device)
    # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, action_dim), nn.Tanh()
        )
    # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def act(self, state):

        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):

        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init):

        self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = Buffer()

        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
        self.set_action_std(self.action_std)
    def select_action(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)


        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))







class MABuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.obsstates = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.obsstates[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class MAActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, agent_num, action_std_init):
        super(MAActorCritic, self).__init__()
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,) , action_std_init ** 2).to(device)
    # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, action_dim), nn.ReLU(),
        )
    # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim*agent_num, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def act(self, state,obsstates):

        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(obsstates)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state,obsstates , action):

        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(obsstates)

        return action_logprobs, state_values, dist_entropy
class MAPPO:
    def __init__(self, state_dim, action_dim, agent_num, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init):

        self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.agent_num = agent_num
        self.buffer = {}
        for i in range(agent_num):
            self.buffer[i] = MABuffer()

        self.policy = MAActorCritic(state_dim, action_dim, agent_num, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = MAActorCritic(state_dim, action_dim, agent_num, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
        self.set_action_std(self.action_std)
    def select_action(self, state, obsstate, n):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            obsstate = torch.FloatTensor(obsstate).to(device)
            action, action_logprob, state_val = self.policy_old.act(state, obsstate)

        self.buffer[n].states.append(state)
        self.buffer[n].actions.append(action)
        self.buffer[n].obsstates.append(obsstate)
        self.buffer[n].logprobs.append(action_logprob)
        self.buffer[n].state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()

    def update(self, buffer):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)


        old_states = torch.squeeze(torch.stack(buffer.states, dim=0)).detach().to(device)
        old_obsstates = torch.squeeze(torch.stack(buffer.obsstates, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_obsstates, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer


    def train(self):
        for i in range(self.agent_num):
            self.update(self.buffer[i])
            self.buffer[i].clear()
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))