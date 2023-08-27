import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import tkinter as tk


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
        self.linkposition = [xy + 2 * self.length * np.array([np.cos(self.angular), np.sin(self.angular)]),
                             xy - 2 * self.length * np.array([np.cos(self.angular), np.sin(self.angular)]),
                             xy + 2 * self.length * np.array([np.sin(self.angular), np.cos(self.angular)]),
                             xy - 2 * self.length * np.array([np.sin(self.angular), np.cos(self.angular)])]

    # def update(self,period, acceleration, angular_acceleration):
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
                 num_drones: int = 1,
                 initial_xyzs=None,
                 period: float = 1 / 240,
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
            drone = Drone_2D(initial_xyzs[i], 0.0, np.array(([0.0, 0.0])), 0.0, i)
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
            self._apply_action(i, action)
        rewards, done = self.get_rewards()
        obs = self.get_observation()
        return obs, rewards, done

    def _apply_action(self, drones_group, action):
        center_x = 0.0
        center_y = 0.0
        velocity_x_sum = 0.0
        velocity_y_sum = 0.0
        force_x_sum = 0.0
        force_y_sum = 0.0
        inertia_sum = 0.0
        torque_sum = 0.0
        angular_velocity_sum = 0.0
        centroid_angular = 0.0

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
            torque = action[drone_id][2] + (self.drones[drone_id].xy[0] - center_x) * force_y - (
                        self.drones[drone_id].xy[1] - center_y) * force_x - 0.01 * self.drones[
                         drone_id].angular_velocity
            torque_sum += torque
            inertia_sum += self.L2D + self.M * np.linalg.norm(self.drones[drone_id].xy - new_center_old) ** 2

        acceleration_centroid = np.array([force_x_sum, force_y_sum]) / self.M / num_drones
        angular_acceleration = torque_sum / inertia_sum

        new_center = new_center_old + self.period * centroid_velocity + 0.5 * acceleration_centroid * self.period ** 2
        centroid_velocity += self.period * acceleration_centroid
        centroid_angular_delta = centroid_angular_velocity * self.period + 0.5 * angular_acceleration * self.period ** 2
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
        # for drone_id in drones_group:
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

    def link(self, i):
        for j in range(self.num_drones):
            drone1 = self.drones[i]
            drone2 = self.drones[j]
            distance = np.linalg.norm(drone1.xy - drone2.xy)
            angle_difference = abs(drone1.angular - drone2.angular) % (np.pi / 2)
            angle_difference = min(angle_difference, (np.pi / 2) - angle_difference)
            if self.length <= distance <= 2.5 * self.length and angle_difference <= np.radians(np.pi / 4) and \
                    self.drones[j].link != i:
                self.drones[i].link = j
                self.drones[j].linked.append(i)
                self._adjust_link_position(i, j, self.drones[i].dronesgroup)
                return
            # self.drones[i].dronesgroup = self.drones[j].dronesgroup

    def _adjust_link_position(self, i, j, pre_dronesgroup):
        # self.drones[i].angular = self.drones[j].angular

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

    def unlink(self, i):
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

        # if self.check_collisions():
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

    def reset(self, initial_positions):
        self.drones = []
        self.drones_groups = [[i] for i in range(self.num_drones)]
        for i in range(self.num_drones):
            drone = Drone_2D(initial_positions[i], 0.0, np.array([0.0, 0.0]), 0.0, i)
            self.drones.append(drone)
        return self.get_observation()
