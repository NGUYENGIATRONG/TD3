# ### Walking controller
# Written by Shishir Kolathaya shishirk@iisc.ac.in
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from future.standard_library import import_

# from test_spot_policy import step_height

"""Utilities for realizing walking controllers."""
import sys
sys.path.append("/home/giatrong/PycharmProjects/pythonProject/simulation")


from dataclasses import dataclass
from collections import namedtuple
from utils import spot_kinematic
import numpy as np
from simulation import bullet_client
from simulation import spot_pybullet_env
# from simulation import spot_pybullet_env as spot_env_module
# from spot_pybullet_env import SpotEnv as e

from scipy.spatial.transform import Rotation as R
no_of_points = 100
# env = spot_env_module.SpotEnv()
_pybullet_client = bullet_client.BulletClient()


def constrain_theta(theta):
    """
    Lấy phần dư của phép chia theta / 2 * no_of_points

    :param theta:
    :return:
    """
    theta = np.fmod(theta, 2 * no_of_points)
    if theta < 0:
        theta = theta + 2 * no_of_points
    return theta


@dataclass
class LegData:
    name: str
    motor_hip: float = 0.0
    motor_knee: float = 0.0
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    step_length: float = 0.0
    step_height: float = 0.0
    x_shift = 0.0
    y_shift = 0.0


@dataclass
class RobotData:
    front_right: float
    front_left: float
    back_right: float
    back_left: float


class WalkingController:
    def __init__(self,
                 gait_type='trot',
                 phase=(0, 0, 0, 0)
                 ):
        self._phase = RobotData(front_right=phase[0], front_left=phase[1], back_right=phase[2], back_left=phase[3])
        self.front_left = LegData('fl')
        self.front_right = LegData('fr')
        self.back_left = LegData('bl')
        self.back_right = LegData('br')
        self.gait_type = gait_type

        self.MOTOROFFSETS_Spot = [np.radians(160.2), np.radians(43.28)]

        self.Spot_kinematics = spot_kinematic.SpotKinematics()

        self.step_length_1 = []
        self.step_length_2 = []
        self.step_length_3 = []
        self.step_length_4 = []

    def update_leg_theta(self, theta):
        """
        Tùy thuộc vào dáng đi, theta cho mỗi chân sẽ được tính toán
        :param theta:
        :return:
        """
        self.front_right.theta = constrain_theta(theta + self._phase.front_right)
        self.front_left.theta = constrain_theta(theta + self._phase.front_left)
        self.back_right.theta = constrain_theta(theta + self._phase.back_right)
        self.back_left.theta = constrain_theta(theta + self._phase.back_left)

    def initialize_elipse_shift(self, x_shift, y_shift):
        self.front_right.x_shift = x_shift[0]
        self.front_left.x_shift = x_shift[1]
        self.back_right.x_shift = x_shift[2]
        self.back_left.x_shift = x_shift[3]

        self.front_right.y_shift = y_shift[0]
        self.front_left.y_shift = y_shift[1]
        self.back_right.y_shift = y_shift[2]
        self.back_left.y_shift = y_shift[3]

    def initialize_leg_state(self, theta, step_length, test=False):
        """
        Khởi tạo tất cả các tham số của các quỹ đạo chân

        :param theta: tham số chu kỳ quỹ đạo theta
        :param action: các tham số điều biến quỹ đạo được dự đoán bởi chính sách
        :param test: test IK
        :return: namedtuple('legs', 'front_right front_left back_right back_left')
        """
        Legs = namedtuple('legs', 'front_right front_left back_right back_left')
        legs = Legs(front_right=self.front_right, front_left=self.front_left, back_right=self.back_right,
                    back_left=self.back_left)

        self.update_leg_theta(theta)

        if test is False:
            leg_sl = step_length # fr fl br bl

            self._update_leg_step_length_val(leg_sl)
            # self.initialize_elipse_shift(action[4:8], action[8:12])

        return legs

    def _update_leg_step_length_val(self, step_length):
        """

        :param step_length: step length of each leg trajectories
        :return:
        """
        self.front_right.step_length = step_length[0]
        self.front_left.step_length = step_length[1]
        self.back_right.step_length = step_length[2]
        self.back_left.step_length = step_length[3]

    def _update_leg_step_height_val(self, step_height):
        """
        Cập nhật giá trị chiều cao bước chân (step_height) cho từng chân.

        :param step_height: danh sách chiều cao bước cho mỗi chân (front_right, front_left, back_right, back_left)
        """
        self.front_right.step_height = step_height[0]
        self.front_left.step_height = step_height[1]
        self.back_right.step_height = step_height[2]
        self.back_left.step_height = step_height[3]

    def run_elliptical_traj_spot(self, theta, step_length,step_mode):
        """
        Bộ điều khiển quỹ đạo bán-ellipse

        :param theta: tham số chu kỳ quỹ đạo theta
        :param action: các tham số điều chỉnh quỹ đạo được dự đoán bởi policy
        :return: danh sách vị trí của động cơ cho hành động mong muốn
        """
        from simulation import spot_pybullet_env
        # from simulation.spot_pybullet_env import SpotEnv
        # env = SpotEnv()
        legs = self.initialize_leg_state(theta, step_length)
        print(step_mode)
        # ori = env.get_base_pos_and_orientation()[1]
        # euler_angles = R.from_quat(ori).as_euler('xyz', degrees=True)
        # pitch_angle = euler_angles[1]
        # print(pitch_angle)
        step_height = 0.08
        x_center = 0.02
        y_center = -0.29

        x = y = 0
        phase_offset = np.radians(0.2)
        print(step_height)
        for leg in legs:
            leg_theta = (leg.theta / (2 * no_of_points)) * 2 * np.pi
            leg.r = leg.step_length / 2
            # print(leg.step_height)
            if self.gait_type == "trot":
                x = -leg.r * np.cos(leg_theta) + x_center + leg.x_shift
                if leg_theta > np.pi:
                    flag = 0
                else:
                    flag = 1
                y = step_height * np.sin(leg_theta) * flag + y_center + leg.y_shift
                # print(y)
                if leg.name in ['fr', 'fl']:
                    y-=-0.00
                    if step_mode == 3:
                        y -=0.002
                    # if pitch_angle > 5:  # Giả sử góc lớn hơn 5 độ là lên dốc
                    #     y -= 0.05
                # if leg.name in ['br', 'bl']:
                #     y += 0.01
                #     # if pitch_angle > 15:  # Giả sử góc lớn hơn 5 độ là lên dốc
                #     #     y -= 0.08

            leg.x, leg.y = x, y

            leg.motor_hip, leg.motor_knee, _ = self.Spot_kinematics.inverse_kinematics(leg.x, leg.y, 0)

            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS_Spot[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS_Spot[1]

        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_right.motor_hip+phase_offset,
                            legs.front_right.motor_knee+phase_offset,
                            legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_right.motor_hip,
                            legs.back_right.motor_knee]

        return leg_motor_angles

    def run_elliptical(self, theta, test):
        """
        Run elipse trajectory with IK
        """
        legs = self.initialize_leg_state(theta, action=None, test=test)
        pos,ori = e.get_base_pos_and_orientation()
        # Parameters for elip --------------------
        step_length = 0.07
        step_height = 0.05
        x_center = 0.02
        y_center = -0.29
        # ----------------------------------------

        x = y = 0
        for leg in legs:
            leg_theta = (leg.theta / (2 * no_of_points)) * 2 * np.pi
            leg.r = step_length / 2
            if self.gait_type == "trot":
                x = -leg.r * np.cos(leg_theta) + x_center
                if leg_theta > np.pi:
                    flag = 0
                else:
                    flag = 1
                y = step_height * np.sin(leg_theta) * flag + y_center

            leg.x, leg.y = x, y
            leg.motor_hip, leg.motor_knee, _ = self.Spot_kinematics.inverse_kinematics(leg.x, leg.y, 0)

            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS_Spot[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS_Spot[1]

        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_right.motor_hip,
                            legs.front_right.motor_knee,
                            legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_right.motor_hip,
                            legs.back_right.motor_knee]

        return leg_motor_angles
    #
    # def plot_trajectory(self, theta, step_length, no_of_points):
    #     """
    #     Vẽ đường đi của chân 'fl_' (Front Left) dựa trên công thức tính tọa độ từ góc theta và step_length.
    #     :param theta: Góc ban đầu của chuyển động bước.
    #     :param step_length: Chiều dài bước chân.
    #     :param no_of_points: Số điểm cần vẽ trên đường đi.
    #     """
    #     # Tọa độ trung tâm và tham số
    #     legs = self.initialize_leg_state(theta, step_length)
    #
    #     x_center = 0.02
    #     y_center = -0.29
    #     step_height = 0.08
    #
    #     for leg in legs:
    #         leg.r = leg.step_length / 2
    #
    #         # Duyệt qua các điểm trên elipse
    #         for i in range(no_of_points):
    #             leg_theta_start = (i / no_of_points) * 2 * np.pi
    #             leg_theta_end = ((i + 1) / no_of_points) * 2 * np.pi
    #
    #             x_start = -leg.r * np.cos(leg_theta_start) + x_center + leg.x_shift
    #             y_start = (step_height * np.sin(
    #                 leg_theta_start) if leg_theta_start <= np.pi else 0) + y_center + leg.y_shift
    #
    #             x_end = -leg.r * np.cos(leg_theta_end) + x_center + leg.x_shift
    #             y_end = (step_height * np.sin(leg_theta_end) if leg_theta_end <= np.pi else 0) + y_center + leg.y_shift
    #             print(f"start {x_start,y_start}")
    #             print(f"end {x_end,y_end}")
    #             # Vẽ đoạn thẳng từ điểm hiện tại tới điểm tiếp theo
    #             _pybullet_client.addUserDebugLine(
    #                 lineFromXYZ=[x_start, y_start, 0],  # Tọa độ bắt đầu
    #                 lineToXYZ=[x_end, y_end, 0],  # Tọa độ kết thúc
    #                 lineColorRGB=[1, 0, 0],  # Màu đỏ
    #                 lineWidth=10.0,
    #                 lifeTime=10
    #             )
    #
