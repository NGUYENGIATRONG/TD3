from math import atan2
import  math
import numpy as np
from numpy.ma.core import arctan

class Serial2RKin:
    def __init__(self,
                 base_pivot=(0, 0),
                 link_lengths=(0.11, 0.11, 0.2, 0.2, 0.05)):
        self.link_lengths = link_lengths
        self.base_pivot = base_pivot

    def inverse_kinematics(self, ee_pos, branch=1):
        q = np.zeros(2, float)
        x_y_points = np.array(ee_pos)
        [x, y] = x_y_points.tolist()
        q1_temp = None
        [l1, l2, l3, l4, d] = self.link_lengths
        r1 = np.sqrt((x + d / 2) ** 2 + y ** 2)
        r2 = np.sqrt((x - d / 2) ** 2 + y ** 2)
        # print(2 * l1 * r1)
        # print(2 * l2 * r2)
        # print(l2 ** 2 + r2 ** 2 - l4 ** 2)
        # Đầu tiên, kiểm tra xem vị trí đầu cuối có nằm trong không gian làm việc của bộ điều khiển hay không:
        if (l1 ** 2 + r1 ** 2 - l3 ** 2 > 2 * l1 * r1) or ((2 * l2 * r2) < (l2 ** 2 + r2 ** 2 - l4 ** 2)):
            print("Point is outside the workspace")
            valid = False
            return valid, q

            # c = l1 ** 2 - l2 ** 2 - x ** 2 - y ** 2
            # if branch == 1:#nhanh1
            #     q1_temp = np.arctan2(y, x) + np.arccos(-c / np.sqrt(a ** 2 + b ** 2))##phi2
            # elif branch == 2:#nhanh2
            #     q1_temp = np.arctan2(y, x) - np.arccos(-c / np.sqrt(a ** 2 + b ** 2))##phi4
            #
            # q[0] = np.arctan2(y - l2 * np.sin(q1_temp), x - l2 * np.cos(q1_temp))##phi3 neu la nhanh 2 phi 1 neu la nhanh 1
            # q[1] = q1_temp - q[0] #phi2-phi1 or phi4-phi3
        cos_theta1 = (l1 ** 2 + r1 ** 2 - l3 ** 2) / (2 * l1 * r1)
        q[0] = atan2(y, x + d / 2) - math.acos(cos_theta1)
        cos_theta2 = (l2 ** 2 + r2 ** 2 - l4 ** 2) / (2 * l2 * r2)
        q[1] = atan2(y, x - d / 2) + math.acos(cos_theta2)
        # Check if the end-effector point lies in the workspace of the manipulator

        valid = True

        return valid, q

    def jacobian(self, q):
        """
        Provides the Jacobian matrix for the end-effector
        Args:
        --- q : The joint angles of the manipulator [q_hip, q_knee]
        where the angle q_knee is specified relative to the thigh link
        Returns:
        --- mat : A 2x2 velocity Jacobian matrix of the manipulator
        """
        [l1, l2] = self.link_lengths
        mat = np.zeros([2, 2])
        mat[0, 0] = -l1 * np.sin(q[0]) - l2 * np.sin(q[0] + q[1])
        mat[0, 1] = - l2 * np.sin(q[0] + q[1])
        mat[1, 0] = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
        mat[1, 1] = l2 * np.cos(q[0] + q[1])
        return mat


class SpotKinematics:
    """
    SpotKinematics class by NNQ
    """
    def __init__(self,
                 base_pivot1=(0, 0),
                 base_pivot2=(0.05, 0),
                 link_parameters=(0.11, 0.11, 0.2, 0.2, 0.05)):#0.25=l2+l3
        self.base_pivot1 = base_pivot1
        self.base_pivot2 = base_pivot2
        self.link_parameters = link_parameters

    def inverse2d(self, ee_pos):
        """
        2D inverse kinematics
        :param ee_pos: end_effector position
        :return:
        """
        valid = False
        q = np.zeros(2)
        [l1, l2, l3, l4, d] = self.link_parameters
        # [l, _] = self.base_pivot2

        leg = Serial2RKin(self.base_pivot1, [l1, l2, l3, l4, d])
        # leg2 = Serial2RKin(self.base_pivot2, [l3, l4])

        valid1, q1 = leg.inverse_kinematics(ee_pos)
        # q1[0]=phi1,q1[1]=phi2-phi1
        # q2[0]=phi3,q2[1]=phi4-phi3
        if not valid1:
            return valid, q1

        # ee_pos_new = [ee_pos[0] - l * np.cos(q1[0] + q1[1]), ee_pos[1] - l * np.sin(q1[0] + q1[1])]
        # valid2, q2 = leg2.inverse_kinematics(ee_pos_new, branch=2)
        # if not valid2:
        #     return valid, q

        valid = True
        # q = [q1[0], q1[0], q1[0] + q1[1], q2[0] + q2[1]]#[phi1,phi3,phi2,phi4]
        q = [q1[0], q1[1]]
        return valid, q

    def inverse_kinematics(self, x, y, z):
        """
        Spot's inverse kinematics
        :param x: x position
        :param y: y position
        :param z: z position
        :return:
        """
        motor_abduction = np.arctan2(z, -y)
        _, [motor_hip, motor_knee] = self.inverse2d([x, y])  # motorhip=phi1, motor knee =phi3

        if motor_hip > 0:
            motor_hip = -2 * np.pi + motor_hip

        return [motor_hip, motor_knee, motor_abduction]

    def forward_kinematics(self, q):
        """
        Spot's forward kinematics
        :param q: [hip_angle, knee_angle]
        :return: end-effector position
        """
        [l1, l2, l3, l4, d] = self.link_parameters
        [l, _] = self.base_pivot2

        x1 = -d / 2 + l1 * np.cos(q[0])
        y1 = l1 * np.sin(q[0])

        x2 = d / 2 + l2 * np.cos(q[1])
        y2 = l2 * np.sin(q[1])
        D = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if D > (l3 + l4) or D < abs(l3 - l4):
            print('Không tồn tại nghiệm cho cấu hình này. Kiểm tra lại góc theta1 và theta2.')
        a = (l3 ** 2 - l4 ** 2 + D ** 2) / (2 * D)
        h = np.sqrt(l3 ** 2 - a ** 2)
        xm = x1 + a * (x2 - x1) / D
        ym = y1 + a * (y2 - y1) / D
        x = xm + h * (y2 - y1) / D
        y = ym - h * (x2 - x1) / D
        ee_pos = [x, y]

        vaild = True
        return vaild, ee_pos

kinematic = SpotKinematics()
theta1 = np.radians(-146.12159007)
theta2 = np.radians(-61.16792781)
x = -0.05
y = -0.25
z = 0
ee_pos = [x, y, z]
q = [theta1, theta2]
print(np.degrees(kinematic.inverse_kinematics(x, y, z)))
print(kinematic.forward_kinematics(q))