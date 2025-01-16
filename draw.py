from simulation import bullet_client
import pybullet_data

_pybullet_client = bullet_client.BulletClient()
def draw_trajectory_link_3(self, duration, interval=0.1, line_color=[0, 1, 0], line_width=2):
    """
    Vẽ đường đi của link 3 (fr_lower_hip_joint) sử dụng addUserDebugLine.
    :param duration: Thời gian để vẽ đường đi (giây).
    :param interval: Khoảng thời gian giữa các lần lấy tọa độ (giây).
    :param line_color: Màu của đường vẽ (mặc định là màu xanh lá cây).
    :param line_width: Độ dày của đường vẽ.
    """
    prev_position = [0,0,0]
    current_position =[0,0,10]
    print(prev_position, current_position)
    _pybullet_client.addUserDebugLine(
                lineFromXYZ=prev_position,
                lineToXYZ=current_position,
                lineColorRGB=line_color,
                lineWidth=line_width,
                lifeTime=0  # Đường sẽ tồn tại mãi mãi
            )

        # Cập nhật tọa độ trước đó

        # Đợi một khoảng thời gian trước khi lấy tọa độ tiếp theo
