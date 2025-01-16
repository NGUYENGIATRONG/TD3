
import pybullet as p
import pybullet_data
import numpy as np
import time


def load_simulation():
    # Kết nối với PyBullet (trong chế độ GUI để có thể xem)
    physicsClient = p.connect(p.GUI)

    # Đặt đường dẫn dữ liệu PyBullet
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Reset lại mô phỏng
    p.resetSimulation()

    # Đặt các thông số vật lý
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)  # Đặt bước thời gian cho mô phỏng

    # Tải mô hình mặt phẳng (plane.urdf)
    plane_id = p.loadURDF("plane.urdf")

    # Độ dốc của miếng cản
    incline_deg = 15  # Độ dốc 30 độ
    # incline_deg = 30  # Độ dốc 30 độ
    wedge_halfheight_offset = 0.01
    incline_ori = 0  # Góc xoay

    # # Tính chiều cao nửa của miếng cản dựa trên độ nghiêng
    wedge_halfheight = wedge_halfheight_offset + 1.5 * np.tan(np.radians(incline_deg)) / 2.0
    wedgePos = [0, 0, wedge_halfheight]  # Vị trí của miếng cản
    wedgeOrientation = p.getQuaternionFromEuler([0, 0, 0])  # Hướng của miếng cản

    # Đường dẫn mô hình URDF của miếng cản
    wedge_model_path = "/home/giatrong/PycharmProjects/pythonProject/simulation/map30xuongdoc/urdf/map30xuongdoc.urdf"

    # Tải mô hình miếng cản
    wedge_id = p.loadURDF(wedge_model_path, wedgePos, wedgeOrientation)

    # Cài đặt ma sát cho miếng cản (giả sử có hàm set_wedge_friction)
    p.changeDynamics(wedge_id, -1, lateralFriction=1.6)

    return physicsClient, plane_id, wedge_id


if __name__ == "__main__":
    # Tải mô hình mặt phẳng và miếng cản
    physicsClient, plane_id, wedge_id = load_simulation()

    # Chạy mô phỏng
    while True:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)  # Điều chỉnh tốc độ mô phỏng