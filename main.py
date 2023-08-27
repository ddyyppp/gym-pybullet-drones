import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import tkinter as tk



device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()

class Drone_2D():
    def __init__(self, xy, angular, velocity, angular_velocity, dronesgroup):
        self.length = 0.059
        self.xy = xy
        self.angular = angular
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.link = -1
        self.linked = []
        self.dronesgroup = dronesgroup
        self.message = None
        self.linkposition = [xy + 2*self.length*np.array([np.cos(self.angular),np.sin(self.angular)]),
                             xy - 2*self.length*np.array([np.cos(self.angular),np.sin(self.angular)]),
                             xy + 2*self.length*np.array([np.sin(self.angular),np.cos(self.angular)]),
                             xy - 2*self.length*np.array([np.sin(self.angular),np.cos(self.angular)])]

    #def update(self,period, acceleration, angular_acceleration):
    #    self.xy += self.velocity * period + 0.5 * acceleration * (period ** 2)
    #    self.angular += self.angular_velocity * period + 0.5 * angular_acceleration * (period ** 2)
    #    self.velocity += acceleration * period
    #    self.angular_velocity += angular_acceleration * period
    #    self.angular = self.angular % (np.pi/2)
    #    self.linkposition = [self.xy + 2 * self.length * np.array([np.cos(self.angular), np.sin(self.angular)]),
    #                         self.xy - 2 * self.length * np.array([np.cos(self.angular), np.sin(self.angular)]),
    #                         self.xy + 2 * self.length * np.array([np.sin(self.angular), np.cos(self.angular)]),
    #                         self.xy - 2 * self.length * np.array([np.sin(self.angular), np.cos(self.angular)])]
    def update(self):
        self.linkposition = [self.xy + 2 * self.length * np.array([np.cos(self.angular), np.sin(self.angular)]),
                             self.xy - 2 * self.length * np.array([np.cos(self.angular), np.sin(self.angular)]),
                             self.xy + 2 * self.length * np.array([np.sin(self.angular), np.cos(self.angular)]),
                             self.xy - 2 * self.length * np.array([np.sin(self.angular), np.cos(self.angular)])]

    @staticmethod
    def print_header():
        header = ("| Drone ID | Position (x,y) | Angular (deg) | Velocity | Angular Vel. |"
                  " Link Index | Linked Drones | Drone Group | Link Pos. |")
        print(header)
        print('-' * len(header))

    def print_status(self, drone_id):
        position_str = f"({self.xy[0]:.2f}, {self.xy[1]:.2f})"
        angular_str = f"{np.degrees(self.angular):.2f}°"
        velocity_str = f"({self.velocity[0]:.2f}, {self.velocity[1]:.2f})"
        angular_velocity_str = f"{self.angular_velocity:.2f}"
        link_str = f"{self.link}"
        linked_str = ', '.join(map(str, self.linked))
        group_str = f"{self.dronesgroup}"
        link_pos_str = ', '.join([f"({pos[0]:.2f}, {pos[1]:.2f})" for pos in self.linkposition])

        status = (f"| {drone_id:8} | {position_str:15} | {angular_str:13} | {velocity_str:10} |"
                  f" {angular_velocity_str:12} | {link_str:10} | {linked_str:14} | {group_str:11} | {link_pos_str} |")
        print(status)

class Env():
    """Multi-drone environment """

    def __init__(self,
                 num_drones: int=1,
                 initial_xyzs=None,
                 period: float=1/240,
                 record=False,
                 ):
        self.num_drones = num_drones
        self.period = period
        self.M = 0.034
        self.L3D = np.array([
            [4.134e-5, 0.0, 0.0],
            [0.0, 4.134e-5, 0.0],
            [0.0, 0.0, 7.043e-5]
            ])
        self.L2D = 7.043e-5
        self.INIT_XYZS = initial_xyzs.copy()
        self.length = 0.059
        self.drones_groups = [[i] for i in range(num_drones)]
        self.drones = []
        for i in range(num_drones):
            drone = Drone_2D(initial_xyzs[i],0.0,np.array(([0.0,0.0])),0.0, i)
            self.drones.append(drone)

        self.window = tk.Tk()
        self.window.title('Drone Display')
        self.canvas = tk.Canvas(self.window, bg='white', width=1000, height=1000)
        self.canvas.pack()


        if record == True:
            self.history = [[] for _ in range(num_drones)]

    def render(self):
        self.canvas.delete("all")

        for drone in self.drones:
            x, y = drone.xy * 100 + 500
            angle = drone.angular
            color = 'blue' if (drone.linked is not None and drone.link != -1) else 'red'


            x1, y1 = x + 5 * np.cos(angle) - 5 * np.sin(angle), y + 5 * np.sin(angle) + 5 * np.cos(angle)
            x2, y2 = x - 5 * np.cos(angle) - 5 * np.sin(angle), y - 5 * np.sin(angle) + 5 * np.cos(angle)
            x3, y3 = x - 5 * np.cos(angle) + 5 * np.sin(angle), y - 5 * np.sin(angle) - 5 * np.cos(angle)
            x4, y4 = x + 5 * np.cos(angle) + 5 * np.sin(angle), y + 5 * np.sin(angle) - 5 * np.cos(angle)


            self.canvas.create_polygon(x1, y1, x2, y2, x3, y3, x4, y4, fill=color)

        self.window.update_idletasks()
        self.window.update()

    def link_step(self, action):
        # action -- force_x force_y Torque
        for i in range(self.num_drones):
            if action[i][3] > 0 and self.drones[i].link == -1:
                self.link(i)
            if action[i][3] < 0 and self.drones[i].link != -1:
                self.unlink(i)
        self.update_groups()
    def action_step(self, action):
        for i in self.drones_groups:
            self._apply_action(i,action)
        rewards,done = self.get_rewards()
        obs = self.get_observation()
        return obs, rewards, done


    def _apply_action(self,drones_group,action):
        center_x = 0.0
        center_y = 0.0
        velocity_x_sum = 0.0
        velocity_y_sum = 0.0
        force_x_sum = 0.0
        force_y_sum = 0.0
        inertia_sum = 0.0
        torque_sum = 0.0
        angular_velocity_sum = 0.0
        centroid_angular=0.0

        num_drones = len(drones_group)
        if num_drones == 0:
            return
        for drone_id in drones_group:
            center_x += self.drones[drone_id].xy[0]
            center_y += self.drones[drone_id].xy[1]
            velocity_x_sum += self.drones[drone_id].velocity[0]
            velocity_y_sum += self.drones[drone_id].velocity[1]



        center_x /= num_drones
        center_y /= num_drones
        centroid_velocity = np.array([velocity_x_sum / num_drones, velocity_y_sum / num_drones])

        new_center_old = np.array([center_x, center_y])

        for drone_id in drones_group:
            position = self.drones[drone_id].xy - new_center_old
            velocity = self.drones[drone_id].velocity - centroid_velocity
            radius = np.linalg.norm(position)
            relative_velocity = np.cross(position, velocity) / radius ** 2 if radius != 0 else 0
            angular_velocity_sum += relative_velocity
            centroid_angular += self.drones[drone_id].angular

        centroid_angular_velocity = angular_velocity_sum / num_drones

        for drone_id in drones_group:
            force_x = action[drone_id][0] - 0.01 * self.drones[drone_id].velocity[0]
            force_x_sum += force_x
            force_y = action[drone_id][1] - 0.01 * self.drones[drone_id].velocity[1]
            force_y_sum += force_y
            torque = action[drone_id][2] + (self.drones[drone_id].xy[0]-center_x) * force_y - (self.drones[drone_id].xy[1]-center_y) * force_x - 0.01 * self.drones[drone_id].angular_velocity
            torque_sum += torque
            inertia_sum += self.L2D + self.M * np.linalg.norm(self.drones[drone_id].xy - new_center_old)**2

        acceleration_centroid = np.array([force_x_sum, force_y_sum]) / self.M / num_drones
        angular_acceleration = torque_sum / inertia_sum


        new_center = new_center_old+ self.period * centroid_velocity + 0.5*acceleration_centroid*self.period**2
        centroid_velocity += self.period * acceleration_centroid
        centroid_angular_delta = centroid_angular_velocity*self.period + 0.5*angular_acceleration*self.period**2
        centroid_angular_velocity += self.period * angular_acceleration



        for drone_id in drones_group:

            position = self.drones[drone_id].xy - new_center_old
            theta = centroid_angular_delta
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta), np.cos(theta)]])
            new_position = rotation_matrix.dot(position)

            self.drones[drone_id].angular = np.arctan2(position[1], position[0]) + theta
            self.drones[drone_id].xy = new_position + new_center
            self.drones[drone_id].angular_velocity = centroid_angular_velocity

            r = np.array(self.drones[drone_id].xy) - new_center
            omega = np.array([0, 0, self.drones[drone_id].angular_velocity])
            r_3D = np.array([r[0], r[1], 0])
            relative_velocity = np.cross(omega, r_3D)
            self.drones[drone_id].velocity = relative_velocity[:2] + centroid_velocity
            self.drones[drone_id].update()
        #for drone_id in drones_group:
        #    r = np.array(self.drones[drone_id].xy) - new_center
        #    omega = np.array([0, 0, self.drones[drone_id].angular_velocity])
        #    r_3D = np.array([r[0],r[1],0])
        #    relative_velocity = np.cross(omega, r_3D)
        #    centripetal_acceleration = self.drones[drone_id].angular_velocity ** 2 * r
        #    coriolis_acceleration = 2 * np.cross(omega, relative_velocity)
        #    coriolis_acceleration = coriolis_acceleration[:2]
        #    acceleration = acceleration_centroid + angular_acceleration * r + centripetal_acceleration + coriolis_acceleration
        #    self.drones[drone_id].update(self.period, acceleration, angular_acceleration)

  #  def _dgl(self, force_x_sum, force_y_sum, ):
  #      pass

    def link(self,i):
        for j in range(self.num_drones):
            drone1 = self.drones[i]
            drone2 = self.drones[j]
            distance = np.linalg.norm(drone1.xy - drone2.xy)
            angle_difference = abs(drone1.angular - drone2.angular) % (np.pi / 2)
            angle_difference = min(angle_difference, (np.pi / 2) - angle_difference)
            if self.length <= distance <= 2.5*self.length and angle_difference <= np.radians(np.pi/4) and self.drones[j].link!=i:
                self.drones[i].link = j
                self.drones[j].linked.append(i)
                self._adjust_link_position(i,j,self.drones[i].dronesgroup)
               # self.drones[i].dronesgroup = self.drones[j].dronesgroup

    def _adjust_link_position(self, i, j, pre_dronesgroup):
        #self.drones[i].angular = self.drones[j].angular

        min_distance = float('inf')
        closest_link_position = None

        for k in range(4):
            distance = np.linalg.norm(self.drones[j].linkposition[k] - self.drones[i].xy)
            if distance < min_distance:
                min_distance = distance
                closest_link_position = self.drones[j].linkposition[k]
        delta_xy = closest_link_position - self.drones[i].xy
       # self.drones[i].xy = closest_link_position

        for rest_drones in range(self.num_drones):
            if self.drones[rest_drones].dronesgroup == pre_dronesgroup:
                self.drones[rest_drones].xy += delta_xy
                self.drones[rest_drones].angular = self.drones[j].angular
                self.drones[rest_drones].dronesgroup = self.drones[j].dronesgroup


    def unlink(self,i):
        j = self.drones[i].link
        self.drones[i].link = -1
        self.drones[i].dronesgroup = i
        self.drones[j].linked.remove(i)
        next_drones = self.drones[i].linked
        while next_drones:
            temp = []
            for next in next_drones:
                temp += (self.drones[next].linked)
                self.drones[next].dronesgroup = i
            next_drones = temp

    def update_groups(self):
        self.drones_groups = [[] for _ in range(self.num_drones)]
        for i in range(self.num_drones):
            group_index = self.drones[i].dronesgroup
            self.drones_groups[group_index].append(i)

    def check_collisions(self):
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                drone1 = self.drones[i]
                drone2 = self.drones[j]

                distance = np.linalg.norm(drone1.xy - drone2.xy)

                if distance < 2 * self.length:
                    corners1 = self.get_corners(drone1)
                    corners2 = self.get_corners(drone2)

                    if self.check_collision(corners1, corners2):
                        print(f"Collision detected between drone {i} and drone {j}!")
                        return True

    def get_corners(self, drone):
        length = self.length
        corners = []
        for i in range(4):
            angle = drone.angular + i * np.pi / 2
            offset = np.array([length * np.cos(angle), length * np.sin(angle)])
            corners.append(drone.xy + offset)
        return corners

    def check_collision(self, corners1, corners2):
        for corners in [corners1, corners2]:
            for i in range(4):
                normal = np.array([corners[i][1] - corners[(i + 1) % 4][1],
                                   corners[(i + 1) % 4][0] - corners[i][0]])
                normal /= np.linalg.norm(normal)

                min1, max1 = float('inf'), float('-inf')
                min2, max2 = float('inf'), float('-inf')
                for j in range(4):
                    projection1 = np.dot(normal, corners1[j])
                    projection2 = np.dot(normal, corners2[j])
                    min1, max1 = min(min1, projection1), max(max1, projection1)
                    min2, max2 = min(min2, projection2), max(max2, projection2)

                if max1 < min2 or max2 < min1:
                    return False
        return True

    def get_rewards(self):
        done = 0
        reward = 10

       # x_positions = [drone.xy[0] for drone in self.drones]
       # x_length = max(x_positions) - min(x_positions)
       # reward += x_length * 5

        scaling_factor = 0.1
        for drone in self.drones:
            distance_to_origin = (drone.xy[0] ** 2 + drone.xy[1] ** 2) ** 0.5
            reward -= distance_to_origin * scaling_factor

        assembled_count = sum(1 for drone in self.drones if drone.linked)
        reward += assembled_count * 10

        #if self.check_collisions():
        #    reward -= 100
        #    done = 1

        return reward, done
    def get_observation(self):
        obs = []
        for drone in self.drones:
            drone_obs = np.concatenate([
                drone.xy,
                [drone.angular],
                drone.velocity,
                [drone.angular_velocity],
                [drone.link]
            ])
            obs.append(drone_obs)
        return np.array(obs)


    def reset(self,initial_positions):
        self.drones = []
        self.drones_groups = [[i] for i in range(self.num_drones)]
        for i in range(self.num_drones):
            drone = Drone_2D(initial_positions[i], 0.0, np.array([0.0, 0.0]), 0.0, i)
            self.drones.append(drone)
        return self.get_observation()


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
    ENV = Env(num_drones=9,
                 initial_xyzs=initial_positions,
                 record=False,
                 )

    ppo_agent = MAPPO(state_dim = 19, action_dim = 16, obs_dim = 63, agent_num=9, lr_actor = 0.001, lr_critic = 0.001, gamma=0.95,
                    K_epochs = 15, eps_clip = 0.3, action_std_init = 0.01)
    ppo_agent.load('Test.pt')
    time = 0
    DURATION = 300
    obs = ENV.reset(initial_positions)
    SIM_FREQ = 240
    for i in range(0, DURATION*SIM_FREQ):
        Action = message_transform(ENV.drones, ppo_agent, obs)
        ENV.link_step(Action)
        obs, rewards, done = ENV.action_step(Action)
        for j in range(ENV.num_drones):
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
            obs = ENV.reset(initial_positions)
        if i % (SIM_FREQ / 20) == SIM_FREQ / 20 - 1:
            ppo_agent.train()
            print("i", i, "rewards", rewards)

        ENV.render()

    ppo_agent.save('Test.pt')



