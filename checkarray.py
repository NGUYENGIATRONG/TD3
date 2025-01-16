import simulation.spot_pybullet_env as spot
import argparse
from fabulous.color import blue, green, red, bold
import numpy as np
import time
from test_spot_policy import motor_angles_list

# Góc động cơ cố định mà bạn chọn (thay đổi các giá trị này theo nhu cầu)
fixed_motor_angles = [np.radians( 22.37531541 - 20), np.radians( 40 - ( 180 - 140.4276661)),np.radians(22.37531541 - 20),np.radians(40 - ( 180 - 140.4276661) ),np.radians(22.37531541 -20),np.radians(40 - ( 180 - 140.4276661)),np.radians(22.37531541 -20),np.radians(40 - ( 180 - 140.4276661))]  # Giá trị góc theo radian (có thể thay đổi)

#hip knee
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Các tham số chương trình
    parser.add_argument('--FrictionCoeff', help='hệ số ma sát chân được đặt', type=float, default=1.6)
    parser.add_argument('--WedgeIncline', help='độ nghiêng của mặt phẳng nghiêng', type=int, default=15)
    parser.add_argument('--MotorStrength', help='cường độ động cơ tối đa được áp dụng', type=float, default=7.0)
    args = parser.parse_args()



    # Khởi tạo môi trường mô phỏng
    env = spot.SpotEnv(render=True,
                       on_rack=True,
                       wedge=False,
                       gait='trot',
                       default_pos=(-1, 0, 0.2))


    state = env.reset()

    # Robot giữ nguyên góc động cơ cố định
    t_r = 0

    while 1:
        # Sử dụng góc động cơ cố định cho mỗi bước
        state, r, done, info = env.step(fixed_motor_angles)
        t_r += r
        motor_angles_list=env.get_motor_angles()
        print(np.degrees(motor_angles_list))
        # In thông tin góc động cơ tại mỗi bước (nếu cần)

