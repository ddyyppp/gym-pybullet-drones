import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import tkinter as tk



device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()


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
    def __init__(self, state_dim, action_dim, obs_dim, agent_num, action_std_init):
        super(MAActorCritic, self).__init__()
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,) , action_std_init ** 2).to(device)
    # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, action_dim*2), nn.Identity(),
        )
    # Critic
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

#    def set_action_std(self, new_action_std):
#        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def act(self, state,obsstates):
        action_parameters = self.actor(state)
        action_mean = action_parameters[..., :self.action_dim]
        action_std_dev = nn.Softplus()(action_parameters[..., self.action_dim:])
        cov_mat = torch.diag_embed(action_std_dev ** 2 + 1e-7)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(obsstates)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, obsstates, action):
        action_parameters = self.actor(state)
        action_mean = action_parameters[..., :self.action_dim]
        action_std_dev = nn.Softplus()(action_parameters[..., self.action_dim:])
        cov_mat = torch.diag_embed(action_std_dev ** 2 + 1e-7)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(obsstates)

        return action_logprobs, state_values, dist_entropy
class MAPPO:
    def __init__(self, state_dim, action_dim, obs_dim, agent_num, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init):

        self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.agent_num = agent_num
        self.buffer = {}
        for i in range(agent_num):
            self.buffer[i] = MABuffer()

        self.policy = MAActorCritic(state_dim, action_dim, obs_dim, agent_num, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = MAActorCritic(state_dim, action_dim, obs_dim, agent_num, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()


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
        action = action.detach().cpu().numpy().flatten()
        max_action_value = np.array([0.034, 0.034, 7.043e-5])
        action[:3] = np.clip(action[:3], -max_action_value, max_action_value)

        return action

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

