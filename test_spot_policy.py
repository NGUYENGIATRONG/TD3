# Import spotenv
import simulation.spot_pybullet_env as spot
import argparse
from fabulous.color import blue, green, red, bold
import numpy as np

step_length =[0.08,0.08,0.08,0.08]
# step_height =[0.08,0.08,0.08,0.08]
mang1 = np.full(5000,0)
mang2 = np.full(5000,0)
mang3 = np.full(5000,0)
mang4 = np.full(5000,0)
mang5 = np.full(5000,0)
mang6 = np.full(5000,0)
mang7 = np.full(5000,0)
mang8 = np.full(5000,0)
motor_angles_list = [mang1,mang2,mang3,mang4,mang5,mang6,mang7,mang8]
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Các tham số của chương trình
    parser.add_argument('--PolicyDir', help='directory of the policy to be tested', type=str, default='23.04.1.j')
    parser.add_argument('--FrictionCoeff', help='foot friction value to be set', type=float, default=1.6)
    parser.add_argument('--WedgeIncline', help='wedge incline degree of the wedge', type=int, default=15)
    parser.add_argument('--WedgeOrientation', help='wedge orientation degree of the wedge', type=float, default=0)
    parser.add_argument('--MotorStrength', help='maximum motor Strength to be applied', type=float, default=7.0)
    parser.add_argument('--RandomTest', help='flag to sample test values randomly ', type=bool, default=False)
    parser.add_argument('--seed', help='seed for the random sampling', type=float, default=100)
    parser.add_argument('--EpisodeLength', help='number of gait steps of a episode', type=int, default=30000)
    parser.add_argument('--PerturbForce', help='perturbation force to applied perpendicular to the heading direction of the robot', type=float, default=0.0)
    parser.add_argument('--Downhill', help='should robot walk downhill?', type=bool, default=False)
    parser.add_argument('--Stairs', help='test on staircase', type=bool, default=False)
    parser.add_argument('--AddImuNoise', help='flag to add noise in IMU readings', type=bool, default=False)
    parser.add_argument('--Test', help='Test without data', type=bool, default=False)

    args = parser.parse_args()
    policy = np.load("experiments/" + args.PolicyDir + "/iterations/zeros12x11.npy")

    WedgePresent = True
    if args.WedgeIncline == 0 or args.Stairs:
        WedgePresent = False
    elif args.WedgeIncline < 0:
        args.WedgeIncline = -1 * args.WedgeIncline
        args.Downhill = True

    # Khởi tạo môi trường mô phỏng
    env = spot.SpotEnv(render=True,
                       wedge=True,
                       stairs=args.Stairs,
                       downhill=args.Downhill,
                       seed_value=args.seed,
                       on_rack=True,
                       gait='trot',
                       imu_noise=args.AddImuNoise,
                       test=args.Test,
                       default_pos=(-0.0, -0.03, 0.2))

    if args.RandomTest:
        env.set_randomization(default=False)
    else:
        env.incline_deg = args.WedgeIncline
        env.incline_ori = np.radians(args.WedgeOrientation)
        env.set_foot_friction(args.FrictionCoeff)
        env.clips = args.MotorStrength
        env.perturb_steps = 300
        env.y_f = args.PerturbForce

    state = env.reset()

    if args.Test:
        print(bold(blue("\nTest without data\n")))

    print(
        bold(blue("\nTest Parameters:\n")),
        green('\nWedge Inclination:'), red(env.incline_deg),
        green('\nWedge Orientation:'), red(np.degrees(env.incline_ori)),
        green('\nCoeff. of friction:'), red(env.friction),
        green('\nMotor saturation torque:'), red(env.clips)
    )

    # Simulation starts
    t_r = 0
    # Thêm biến lưu lực ngẫu nhiên và bước đếm
    step_counter = 0
    random_force = [0, 0]  # Lực ngẫu nhiên ban đầu là 0
    force_interval = 1  # Mỗi 30000 bước thì thay đổi lực ngẫu nhiên

    for i_step in range(args.EpisodeLength):
        # Lấy danh sách góc động cơ tại bước `i_step`
        # motor_angles = [motor_angles_list[i][i_step] for i in
        #                 range(8)]  # Tạo danh sách góc cho từng động cơ tại bước `i_step`
        # motor_angles = env.get_motor_angles()
        # Thực hiện bước mô phỏng với góc động cơ đã có sẵn
        action = policy.dot(state)
        state, r, done, info = env.step(action)
        t_r += r
        # env.draw_trajectory_link_3(interval=0.1, line_color=[1, 0, 0], line_width=1, lifeTime=0)
 # In thông tin về góc động cơ tại mỗi bước (nếu cần thiết)
        # print(bold(blue(f"\nMotor Angles at Step {i_step}:")), motor_angles)
        # env.apply_ext_force(0,100,link_index=3,visulaize=True,life_time=5)
        step_counter += 1
        # env.pybullet_client.resetDebugVisualizerCamera(0.95, 90, -80, env.get_base_pos_and_orientation()[0])
        env.pybullet_client.resetDebugVisualizerCamera(0.95, 0, -0, env.get_base_pos_and_orientation()[0])

        # Điều kiện kết thúc nếu cần thiết
        # if done:
        #     break

    print("Total reward: " + str(t_r) + ' -> ' + str(args.PolicyDir))
# cấu hình token

