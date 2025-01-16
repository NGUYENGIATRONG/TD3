import simulation.spot_pybullet_env as spot
import argparse
from fabulous.color import blue, green, red, bold
import numpy as np
from collections import deque

step_length = [0.08, 0.08, 0.08, 0.08]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Các tham số của chương trình
    parser.add_argument('--PolicyDir', help='directory of the policy to be tested', type=str, default='23.04.1.j')
    parser.add_argument('--FrictionCoeff', help='foot friction value to be set', type=float, default=7.0)
    parser.add_argument('--WedgeIncline', help='wedge incline degree of the wedge', type=int, default=15)
    parser.add_argument('--WedgeOrientation', help='wedge orientation degree of the wedge', type=float, default=0)
    parser.add_argument('--MotorStrength', help='maximum motor Strength to be applied', type=float, default=7.0)
    parser.add_argument('--RandomTest', help='flag to sample test values randomly ', type=bool, default=False)
    parser.add_argument('--seed', help='seed for the random sampling', type=float, default=100)
    parser.add_argument('--EpisodeLength', help='number of gait steps of an episode', type=int, default=5000)
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
                       on_rack=False,
                       gait='trot',
                       imu_noise=args.AddImuNoise,
                       test=args.Test,
                       default_pos=(-1.9, -0.08, 0.2))

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

    # Biến lưu góc roll, pitch, yaw
    roll_angles = []
    pitch_angles = []
    yaw_angles = []

    # Chạy mô phỏng
    for step in range(args.EpisodeLength):
        state, r, _, angle = env.step(step_length)
        env.apply_ext_force(0, 0, visulaize=True)
        env.pybullet_client.resetDebugVisualizerCamera(0.95, 0, -0, env.get_base_pos_and_orientation()[0])

        # Lấy thông tin góc từ robot
        pos, ori = env.get_base_pos_and_orientation()
        roll, pitch, yaw = env._pybullet_client.getEulerFromQuaternion(ori)

        # Lưu góc vào danh sách
        roll_angles.append(np.degrees(roll))
        pitch_angles.append(np.degrees(pitch))
        yaw_angles.append(np.degrees(yaw))

    # Lưu các góc roll, pitch, yaw vào file .npy
    np.save("roll_angles.npy", np.array(roll_angles))
    np.save("pitch_angles.npy", np.array(pitch_angles))
    np.save("yaw_angles.npy", np.array(yaw_angles))

    print("Simulation completed!")
    print("Roll angles saved to roll_angles.npy")
    print("Pitch angles saved to pitch_angles.npy")
    print("Yaw angles saved to yaw_angles.npy")
