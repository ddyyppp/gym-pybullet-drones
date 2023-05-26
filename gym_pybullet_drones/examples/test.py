import numpy as np


class Dipole:
    def __init__(self, pos, moment):
        self.pos = np.array(pos)
        self.moment = moment


class Drone:
    def __init__(self, pos, a, b, c, moment):
        # 磁偶极在笼子角上的位置
        self.dipoles = [Dipole([pos[0] + i * a / 2, pos[1] + j * b / 2, pos[2] + k * c / 2], moment)
                        for i in [-1, 1] for j in [-1, 1] for k in [-1, 1]]


def compute_force(dipole1, dipole2):
    r = dipole2.pos - dipole1.pos
    r_norm = np.linalg.norm(r)
    r_unit = r / r_norm
    force = (3 * np.dot(dipole2.moment, r_unit) * r_unit - dipole2.moment) / r_norm ** 3
    return force


def compute_torque(dipole, force):
    return np.cross(dipole.pos, force)


def compute_total_force_and_torque(drone1, drone2):
    total_force = np.array([0.0, 0.0, 0.0])
    total_torque = np.array([0.0, 0.0, 0.0])

    for dipole1 in drone1.dipoles:
        for dipole2 in drone2.dipoles:
            force = compute_force(dipole1, dipole2)
            total_force += force
            total_torque += compute_torque(dipole1, force)
    return total_force, total_torque


# 这里是一些参数的示例，你可以根据实际情况进行修改
drone_position1 = [0, 0, 0]  # 无人机1的位置
drone_position2 = [2, 2, 2]  # 无人机2的位置
cage_size_a = 1.0  # 笼子的长度
cage_size_b = 1.0  # 笼子的宽度
cage_size_c = 1.0  # 笼子的高度
dipole_moment = 1.0  # 磁偶极矩

drone1 = Drone(drone_position1, cage_size_a, cage_size_b, cage_size_c, dipole_moment)
drone2 = Drone(drone_position2, cage_size_a, cage_size_b, cage_size_c, dipole_moment)

total_force, total_torque = compute_total_force_and_torque(drone1, drone2)

print("Total force:", total_force)
print("Total torque:", total_torque)