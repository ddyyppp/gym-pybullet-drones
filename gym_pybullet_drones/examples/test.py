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



    def __magnetForce(self,i,j):
        """PyBullet implementation of a ground effect model.

        Inspired by the analytical model used for comparison in (Shi et al., 2019).

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        for ii in range(6):
            for jj in range(6):
                delta = np.linalg.norm(np.array(self.pos[i, 0:3]) - np.array(self.pos[j, 0:3]))
                if delta < 1: # Ignore the magnet of drones more than 1 meters away
                    # Clauculate the position of the 4 ecke
                    length = 0.3
                    height = 0.1

                    magnetForce = [0, 0, 10]
    ################################################################################

   # import numpy as np
   #
#
   # # 真空磁导率
   # mu0 = 4 * np.pi * 1e-7
#
   # # 定义长方体的维度和中心位置
   # box_dimensions = np.array([1, 1, 1])  # 例子：长、宽、高都为1，需要根据实际情况修改
   # box_center1 = np.array([0, 0, 0])  # 例子：第一个长方体的中心坐标，需要根据实际情况修改
   # box_center2 = np.array([2, 2, 2])  # 例子：第二个长方体的中心坐标，需要根据实际情况修改
#
   # # 定义长方体的RPY角度
   # rpy_angles1 = np.array([0, 0, 0])  # 例子：第一个长方体的RPY角度，需要根据实际情况修改
   # rpy_angles2 = np.array([0, 0, 0])  # 例子：第二个长方体的RPY角度，需要根据实际情况修改
#
   # # 定义磁偶极的磁矩
   # magnet_moments = np.array([1, 1, 1, 1, 1, 1])  # 例子：所有磁偶极的磁矩都为1，需要根据实际情况修改
#
   # # 计算长方体的角的相对位置
   # corner_offsets = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1]]) - 0.5
   # corner_offsets *= box_dimensions
#
   # # 用RPY角度计算旋转矩阵
   # rotation1 = R.from_euler('xyz', rpy_angles1)
   # rotation2 = R.from_euler('xyz', rpy_angles2)
#
   # # 用旋转矩阵和中心位置计算每个角的实际位置
   # magnet_positions1 = np.dot(corner_offsets, rotation1.as_matrix().T) + box_center1
   # magnet_positions2 = np.dot(corner_offsets, rotation2.as_matrix().T) + box_center2
#
   # n_magnets = len(magnet_positions1)
#
   # # 初始化总力和总力矩为零
   # total_force = np.array([0, 0, 0])
   # total_torque1 = np.array([0, 0, 0])
   # total_torque2 = np.array([0, 0, 0])
#
   # # 计算每一对磁偶极间的作用力和力矩
   # for i in range(n_magnets):
   #     for j in range(n_magnets):
   #         r = magnet_positions2[j] - magnet_positions1[i]  # 位置矢量
   #         r_norm = np.linalg.norm(r)  # 距离
   #         r_hat = r / r_norm  # 方向
#
   #         # 计算力的大小和方向
   #         force_magnitude = mu0 / (4 * np.pi) * (magnet_moments[i] * magnet_moments[j]) / r_norm ** 2
   #         force = force_magnitude * r_hat
#
   #         # 添加到总力
   #         total_force += force
#
   #         # 计算并添加到总力矩
   #         total_torque1 += np.cross(magnet_positions1[i], force)
   #         total_torque2 += np.cross(magnet_positions2[j], -force)
#
   # print("Total force: ", total_force)
   # print("Total torque on box 1: ", total_torque1)
   # print("Total torque on box 2: ", total_torque2)