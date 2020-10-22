from rlkit.samplers.rollout_functions import multitask_rollout_with_relabeler
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import uuid
from rlkit.envs.point_reacher_env import PointReacherEnv
from scripts.visualization_utils import load_pkl
filename = str(uuid.uuid4())

def simulate_policy(args):
    # various plotting and rollout options
    num_samples = 10
    render = True
    # load saved pkl data
    print(args.file)
    pkl_data = load_pkl(args, discretized=discretized)
    policy, env, relabeler, json_data = pkl_data

    if isinstance(env, PointReacherEnv):
        render = False
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    for i in range(num_samples):
        path = multitask_rollout_with_relabeler(
            env,
            policy,
            relabeler,
            max_path_length=args.H,
            render=args.render or render,
            render_kwargs=dict(mode='human'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    simulate_policy(args)
