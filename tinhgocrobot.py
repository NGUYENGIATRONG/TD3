import matplotlib.pyplot as plt
import numpy as np
from utils import spot_kinematic  # Giả sử bạn có module này như đã nêu trước đó

# Khởi tạo kinematics
spot = spot_kinematic.SpotKinematics()
base1 = spot.base_pivot1
base2 = spot.base_pivot2
[l1, l2, l3, l4] = spot.link_parameters  # Các thông số độ dài liên kết (0.11, 0.25, 0.11, 0.2)

# Hàm xoay một điểm quanh gốc một góc 30 độ
def rotate_point(x, y, angle_deg):
    angle_rad = np.radians(angle_deg)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    rotated_point = np.dot(rotation_matrix, np.array([x, y]))
    return rotated_point[0], rotated_point[1]

# Hàm vẽ cấu hình cho vị trí đầu vào (p1, p2)
def draw_input_position(p1, p2):
    # Sử dụng động học ngược để tính toán các góc khớp
    valid, [q1, q2, q3, q4] = spot.inverse2d(ee_pos=[p1, p2])
    if not valid:
        print("Điểm nằm ngoài phạm vi làm việc.")
        return

    if q1 > 0:
        q1 = -2 * np.pi + q1

    # In ra các giá trị q1 và q2
    print(f"Giá trị q1 (radian): {q1:.3f}")
    print(f"Giá trị q2 (radian): {q2:.3f}")

    # Tính toán tọa độ các khớp
    x_00, y_00 = 0, 0
    x_01, y_01 = base2[0], 0  # Trục xoay của khớp thứ hai

    x1, y1 = l1 * np.cos(q1), l1 * np.sin(q1)
    x2, y2 = x1 + l2 * np.cos(q3), y1 + l2 * np.sin(q3)
    x3, y3 = base2[0] + l3 * np.cos(q2), base2[1] + l3 * np.sin(q2)
    x4, y4 = x3 + l4 * np.cos(q4), y3 + l4 * np.sin(q4)

    # Tạo figure và ax
    fig, ax = plt.subplots()

    # Vẽ cấu hình chân robot
    ax.plot([x_00, x_01], [y_00, y_01], 'bo-')
    ax.plot([x_00, x1], [y_00, y1], 'ro-', label='Liên kết 1')
    ax.plot([x1, x2], [y1, y2], 'go-', label='Liên kết 2')
    ax.plot([x_01, x3], [y_01, y3], 'co-', label='Liên kết 3')
    ax.plot([x3, x4], [y3, y4], 'mo-', label='Liên kết 4')

    # Cài đặt đồ thị để hình ở trung tâm
    ax.set_title('Cấu hình chân robot cho vị trí đầu vào')
    ax.set_xlabel('X-axis (m)')
    ax.set_ylabel('Y-axis (m)')

    # Điều chỉnh giới hạn trục sao cho toàn bộ cấu hình chân robot có thể hiển thị đầy đủ
    ax.set_xlim(min(x_00, x1, x2, x3, x4) - 0.1, max(x_00, x1, x2, x3, x4) + 0.1)
    ax.set_ylim(min(y_00, y1, y2, y3, y4) - 0.1, max(y_00, y1, y2, y3, y4) + 0.1)

    ax.legend()
    ax.set_aspect('equal', adjustable='box')  # Đảm bảo tỷ lệ trục là 1:1
    ax.grid(True)

    # Hiển thị đồ thị
    plt.show()

# Nhập tọa độ x, y từ bàn phím
try:
    input_x = float(input("Nhập giá trị tọa độ x: "))
    input_y = float(input("Nhập giá trị tọa độ y: "))

    # Xoay điểm đầu vào một góc 30 độ
    rotated_x, rotated_y = rotate_point(input_x, input_y, 30)

    print(f"Tọa độ sau khi xoay 30 độ: x = {rotated_x:.3f}, y = {rotated_y:.3f}")

    # Vẽ cấu hình cho tọa độ đã xoay và in ra q1, q2
    draw_input_position(rotated_x, rotated_y)
except ValueError:
    print("Vui lòng nhập giá trị số hợp lệ cho tọa độ x và y.")
