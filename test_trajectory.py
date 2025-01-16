import matplotlib.pyplot as plt
import numpy as np
from utils import spot_kinematic

# Khởi tạo đối tượng SpotKinematics
spot = spot_kinematic.SpotKinematics()
base1 = spot.base_pivot1
base2 = spot.base_pivot2
[l1, l2, l3, l4, d] = spot.link_parameters
pi = np.pi

motor_hip = []
motor_knee = []


def draw(p1, p2):
    """
    Hàm vẽ mô phỏng động học chân
    :param p1: tọa độ x của bàn chân
    :param p2: tọa độ y của bàn chân
    """
    # Tính toán động học nghịch
    valid, [q1, q2] = spot.inverse2d(ee_pos=[p1, p2])
    if not valid:
        print("Invalid position, skipping.")
        return

    # Tính tọa độ các điểm
    x1 = l1 * np.cos(q1) - d / 2
    y1 = l1 * np.sin(q1)
    x2 = l2 * np.cos(q2) + d / 2
    y2 = l2 * np.sin(q2)

    # Vẽ các đoạn thẳng
    plt.plot([-d / 2, x1], [0, y1], 'r-', linewidth=2, label='Motor 1 to Joint 1')  # Động cơ 1 đến khớp 1
    plt.plot([+d / 2, x2], [0, y2], 'b-', linewidth=2, label='Motor 2 to Joint 2')  # Động cơ 2 đến khớp 2
    plt.plot([x1, p1], [y1, p2], 'g-', linewidth=2, label='Joint 1 to Foot')       # Khớp 1 đến bàn chân
    plt.plot([x2, p1], [y2, p2], 'm-', linewidth=2, label='Joint 2 to Foot')       # Khớp 2 đến bàn chân

    # Vẽ các điểm
    plt.plot(-d / 2, 0, 'ko', markersize=10, label='Motor 1')  # Động cơ 1
    plt.plot(d / 2, 0, 'ko', markersize=10, label='Motor 2')   # Động cơ 2
    plt.plot(x1, y1, 'ro', markersize=10, label='Joint 1')     # Khớp 1
    plt.plot(x2, y2, 'bo', markersize=10, label='Joint 2')     # Khớp 2
    plt.plot(p1, p2, 'go', markersize=10, label='Foot')        # Bàn chân

    # Tùy chỉnh đồ thị
    plt.title('Leg Trajectory')
    plt.xlabel('X-axis (m)')
    plt.ylabel('Y-axis (m)')
    plt.xlim(-0.15, 0.2)
    plt.ylim(-0.35, 0.01)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)

    # Hiển thị đồ thị
    plt.pause(0.0001)
    plt.clf()  # Xóa đồ thị trước khi vẽ lần tiếp theo


# Cài đặt tham số quỹ đạo
no_of_points = 100
step_length = 0.1
radius = step_length / 2
theta = 0
y_center = -0.26
foot_clearance = 0.05
x_shift = 0.01
y_shift = 0
count = 5

# Tạo quỹ đạo
x_traj = []
y_traj = []

for _ in range(count * 80):
    leg_theta = (theta / (2 * no_of_points)) * 2 * np.pi

    t = -radius * np.cos(leg_theta) + x_shift
    x_traj.append(t)
    if leg_theta > np.pi:
        flag = 0
    else:
        flag = 1
    z = foot_clearance * np.sin(leg_theta) * flag + y_center + y_shift
    y_traj.append(z)

    theta += 2.5
    theta = np.fmod(theta, 2 * no_of_points)
    draw(t, z)

# Test vị trí cụ thể
# draw(0.00811, -0.22082)
