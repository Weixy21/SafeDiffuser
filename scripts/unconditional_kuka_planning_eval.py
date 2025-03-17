import os
import numpy as np
import torch
import pdb
import pybullet as p
import os.path as osp

import gym
import d4rl

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch import Trainer
from denoising_diffusion_pytorch.datasets.tamp import KukaDataset
from denoising_diffusion_pytorch.mixer_old import MixerUnet
from denoising_diffusion_pytorch.mixer import MixerUnet as MixerUnetNew
from denoising_diffusion_pytorch.temporal_attention import TemporalUnet
from denoising_diffusion_pytorch.utils.rendering import KukaRenderer
import diffusion.utils as utils
import environments
from imageio import get_writer
import torch.nn as nn

from diffusion.models.mlp import TimeConditionedMLP
from diffusion.models import Config

from denoising_diffusion_pytorch.utils.pybullet_utils import get_bodies, sample_placement, pairwise_collision, \
    RED, GREEN, BLUE, BLACK, WHITE, BROWN, TAN, GREY, RGBA, connect, get_movable_joints, set_joint_position, set_pose, add_fixed_constraint, remove_fixed_constraint, set_velocity, get_joint_positions, get_pose, enable_gravity, \
    get_com_pose, create_sphere, get_all_links, set_joint_positions, get_joint_limits, get_joint_state, get_link_state, dump_link

from gym_stacking.env import StackEnv
from tqdm import tqdm


def get_env_state(robot, cubes, attachments):
    joints = get_movable_joints(robot)
    joint_pos = get_joint_positions(robot, joints)

    for cube in cubes:
        pos, rot = get_pose(cube)
        pos, rot = np.array(pos), np.array(rot)

        if cube in attachments:
            attach = np.ones(1)
        else:
            attach = np.zeros(1)

        joint_pos = np.concatenate([joint_pos, pos, rot, attach], axis=0)

    return joint_pos


def execute(samples, diffusion, env, idx=0):
    postprocess_samples = []
    robot = env.robot
    joints = get_movable_joints(robot)
    gains = np.ones(len(joints))

    cubes = env.cubes
    link = 8

    lk = get_all_links(robot)
    m_lk = get_movable_joints(robot)
    
    joint_limits = []
    for j in m_lk:
        # print(j)
        lb, ub = get_joint_limits(robot, j)
        joint_limits.append(np.array([lb, ub]))
    joint_limits = np.array(joint_limits)

    joints_val = samples[:,:7]
    b1 = joints_val - joint_limits[:,0]
    b2 = joint_limits[:,1] - joints_val
    # print("lower limit sat:", b1.min())
    # print("upper limit sat:", b2.min())

    # limits = np.array([[-2.96705973,  2.96705973],
    #    [-2.0943951 ,  2.0943951 ],
    #    [-2.96705973,  2.96705973],
    #    [-2.0943951 ,  2.0943951 ],
    #    [-2.96705973,  2.96705973],
    #    [-2.0943951 ,  2.0943951 ],
    #    [-3.05432619,  3.05432619]])
    # link 0 pos: (-0.1, 0.0, 0.07)
    # link 1 pos: (0.0, -0.03, 0.27749999999999997)
    # link 2 pos: (-0.00030000000000868655, 0.041999999999711046, 0.41900000000020565)
    # link 3 pos: (2.068234906255073e-14, 0.029999999998998647, 0.6945)
    # link 4 pos: (6.545955175194354e-14, -0.03400000000067328, 0.8470000000001665)
    # link 5 pos: (-9.999999989286561e-05, -0.0209999999993537, 1.0405000000002058)
    # link 6 pos: (1.272582842055947e-13, 0.00040000000201545136, 1.180599999999998)
    # link 7 pos: (1.481887826451083e-13, 2.4091388606262143e-12, 1.2810000000000001)
    # link 8 pos: (1.5335936032291432e-13, 2.409138860626216e-12, 1.306)
    # link 9 pos: (1.533651974974231e-13, 2.409138860626216e-12, 1.306)

    ###############################################add obstacles
    # pos, att = get_com_pose(2, -1)
    # id1 = create_sphere(0.1, 0, color=RGBA(1, 0, 0, 1))
    # id2 = create_sphere(0.1, 0, color=RGBA(0, 1, 0, 1))
    # id3 = create_sphere(0.1, 0, color=RGBA(0, 0, 1, 1))
    # set_pose(id1, ((-0.1, 0.0, 0.07), att))
    # set_pose(id2, ((0.0, -0.03, 0.27749999999999997), att))
    # set_pose(id3, ((-0.00030000000000868655, 0.041999999999711046, 0.41900000000020565), att))

    near = 0.001
    far = 4.0
    projectionMatrix = p.computeProjectionMatrixFOV(60., 1.0, near, far)
    location = np.array([1.5, 0, 2.0])
    end = np.array([0.0, 0.0, 1])
    viewMatrix = p.computeViewMatrix(location, end, [0, 0, 1])

    first = True
    saved_id = []
    ims = []
    diff_step, horizon, obs_dim = diffusion.shape
    tt = 20
    list1 = [*range(0, diff_step-40, 20)]
    list2 = [*range(diff_step-40, diff_step, 2)]
    list0 = list(np.concatenate((list1, list2), axis = 0))
    for s in list0:#range(0, diff_step, tt):
        sample_s = diffusion[s]
        # print('diffusion_step: ', s, '/', diff_step)
        j = 0
        for k in range(1, horizon, 2):
            set_joint_positions(robot, m_lk, sample_s[k, :7])
            pos, att = get_com_pose(robot, lk[-1])
            if first:
                id = create_sphere(0.02, 0, color=RGBA(0.961, 0.122, 0.129, 0.5))
                p.setCollisionFilterGroupMask(id, -1, 0,0)
                saved_id.append(id)
            else:
                id = saved_id[j]
                j = j + 1
            set_pose(id, (pos, att))
        first = False
        set_joint_positions(robot, m_lk, samples[0,:7])
        _, _, im, _, seg = p.getCameraImage(width=1024, height=1024, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix) #256, 256
        im = np.array(im)
        im = im.reshape((1024, 1024, 4))
        ims.append(im)

    # for k in range(1, horizon, 5):
    #     set_joint_positions(robot, m_lk, samples[k, :7])
    #     pos, att = get_com_pose(robot, lk[-1])
    #     # id = create_sphere(0.03, 0, color=RGBA(0.941, 0.22, 0.078, 0.239))
    #     id = saved_id[k]
    #     # pdb.set_trace()
    #     p.setCollisionFilterGroupMask(id, -1, 0,0)
    #     # p.setCollisionFilterPair(robot, id, lk[-1], -1, 0)
    #     set_pose(id, (pos, att))

    set_joint_positions(robot, m_lk, samples[0,:7])
    
    # pdb.set_trace()
    # near = 0.001
    # far = 4.0
    # projectionMatrix = p.computeProjectionMatrixFOV(60., 1.0, near, far)

    # # location = np.array([0.1, 0.1, 2.0])
    # # end = np.array([0.0, 0.0, 1.0])
    # location = np.array([1.5, 1.5, 2.0])
    # end = np.array([0.0, 0.0, 1])
    # viewMatrix = p.computeViewMatrix(location, end, [0, 0, 1])

    attachments = set()

    states = [get_env_state(robot, cubes, attachments)]
    rewards = 0
    

    for sample in samples[1:]:
        p.setJointMotorControlArray(bodyIndex=robot, jointIndices=joints, controlMode=p.POSITION_CONTROL,
                targetPositions=sample[:7], positionGains=gains)

        attachments = set()
        # Add constraints of objects
        for j in range(4):
            contact = sample[14+j*8]

            if contact > 0.5:
                add_fixed_constraint(cubes[j], robot, link)
                attachments.add(cubes[j])
                env.attachments[j] = 1
            else:
                remove_fixed_constraint(cubes[j], robot, link)
                set_velocity(cubes[j], linear=[0, 0, 0], angular=[0, 0, 0, 0])
                env.attachments[j] = 0


        for i in range(10):
            p.stepSimulation()

        states.append(get_env_state(robot, cubes, attachments))

        _, _, im, _, seg = p.getCameraImage(width=1024, height=1024, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix) #256, 256
        im = np.array(im)
        im = im.reshape((1024, 1024, 4))

        state = env.get_state()
        # print(state)
        reward = env.compute_reward()

        rewards = rewards + reward
        ims.append(im)
        # writer.append_data(im)

    attachments = {}
    env.attachments[:] = 0
    env.get_state()
    reward = env.compute_reward()
    rewards = rewards + reward
    state = get_env_state(robot, cubes, attachments)

    # writer.close()

    return state, states, ims, rewards


def eval_episode(model, env, dataset, idx=0):
    state = env.reset()
    states = [state]

    idxs = [(0, 3), (1, 0), (2, 1)]
    cond_idxs = [map_tuple[idx] for idx in idxs]
    stack_idxs = [idx[0] for idx in idxs]
    place_idxs = [idx[1] for idx in idxs]

    samples_full_list = []
    obs_dim = dataset.obs_dim

    samples = torch.Tensor(state)
    samples = (samples - dataset.mins) / (dataset.maxs - dataset.mins + 1e-8)
    samples = samples[None, None, None].cuda()
    samples = (samples - 0.5) * 2

    conditions = [
           (0, obs_dim, samples),
    ]

    rewards = 0
    frames = []

    total_samples = []

    for i in range(1):  #3
        # samples = samples_orig = trainer.ema_model.guided_conditional_sample(model, 1, conditions, cond_idxs[i], stack_idxs[i], place_idxs[i])
        trainer.ema_model.mins = torch.tensor(dataset.mins[:7])
        trainer.ema_model.maxs = torch.tensor(dataset.maxs[:7])
        samples, diffusion, b_min = samples_orig, diffusion_orig, b_min_orig = trainer.ema_model.conditional_sample(1, conditions)

        samples = torch.clamp(samples, -1, 1)
        samples_unscale = (samples + 1) * 0.5
        samples = dataset.unnormalize(samples_unscale)
        samples = to_np(samples.squeeze(0).squeeze(0))
        
        diffusion = torch.clamp(diffusion, -1, 1)
        diffusion_unscale = (diffusion + 1)*0.5
        diffusion = dataset.unnormalize(diffusion_unscale)
        diffusion = to_np(diffusion)

        samples, samples_list, frames_new, reward = execute(samples, diffusion, env, idx=i)
        frames.extend(frames_new)
        total_samples.extend(samples_list)

        samples_full_list.extend(samples_list)

        samples = (samples - dataset.mins) / (dataset.maxs - dataset.mins + 1e-8)
        samples = torch.Tensor(samples[None, None, None]).to(samples_orig.device)
        samples = (samples - 0.5) * 2


        conditions = [
               (0, obs_dim, samples),
        ]

        samples_list.append(samples)

        rewards = rewards + reward

    ####################################################################save videos
    if not osp.exists("uncond_samples_CBF_cf/"):
        os.makedirs("uncond_samples_CBF_cf/")

    writer = get_writer("uncond_samples_CBF_cf/uncond_video_writer{}.mp4".format(idx))

    for frame in frames:
        writer.append_data(frame)

    np.save("uncond_samples_CBF_cf/uncond_sample_{}.npy".format(idx), np.array(total_samples))


    return rewards, b_min


class PosGuide(nn.Module):
    def __init__(self, cube, cube_other):
        super().__init__()
        self.cube = cube
        self.cube_other = cube_other

    def forward(self, x, t):
        cube_one = x[..., 64:, 7+self.cube*8: 7+self.cube*8]
        cube_two = x[..., 64:, 7+self.cube_other*8:7+self.cube_other*8]

        pred = -100 * torch.pow(cube_one - cube_two, 2).sum(dim=-1)
        return pred



def to_np(x):
    return x.detach().cpu().numpy()

def pad_obs(obs, val=0):
    state = np.concatenate([np.ones(1)*val, obs])
    return state

def set_obs(env, obs):
    state = pad_obs(obs)
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])

#### dataset
H = 128
dataset = KukaDataset(H)
# import pdb; pdb.set_trace()
env_name = "multiple_cube_kuka_temporal_convnew_real2_128"
H = 128
T = 1000

diffusion_path = f'logs/{env_name}/'
diffusion_epoch = 650

dataset = KukaDataset(H)
weighted = 5.0
trial = 0

savepath = f'logs/{env_name}/plans_weighted{weighted}_{H}_{T}/{trial}'
utils.mkdir(savepath)

## dimensions
obs_dim = dataset.obs_dim
act_dim = 0

#### model
# model = MixerUnet(
#     dim = 32,
#     image_size = (H, obs_dim),
#     dim_mults = (1, 2, 4, 8),
#     channels = 2,
#     out_dim = 1,
# ).cuda()

# model = MixerUnetNew(
#     H,
#     obs_dim * 2,
#     0,
#     dim = 32,
#     dim_mults = (1, 2, 4, 8),
# #     out_dim = 1,
# ).cuda()

model = TemporalUnet(
    horizon = H,
    transition_dim = obs_dim,
    cond_dim = H,
    dim = 128,
    dim_mults = (1, 2, 4, 8),
).cuda()


diffusion = GaussianDiffusion(
    model,
    channels = 2,
    image_size = (H, obs_dim),
    timesteps = T,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

#### load reward and value functions
# reward_model, *_ = utils.load_model(reward_path, reward_epoch)
# value_model, *_ = utils.load_model(value_path, value_epoch)
# value_guide = guides.ValueGuide(reward_model, value_model, discount)
env = StackEnv(conditional=False)

trainer = Trainer(
    diffusion,
    dataset,
    env,
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                     # turn on mixed precision training with apex
    results_folder = diffusion_path,
)


print(f'Loading: {diffusion_epoch}')
trainer.load(diffusion_epoch)
render_kwargs = {
    'trackbodyid': 2,
    'distance': 10,
    'lookat': [10, 2, 0.5],
    'elevation': 0
}

x = dataset[0][0].view(1, 1, H, obs_dim).cuda()
conditions = [
       (0, obs_dim, x[:, :, :1]),
]
trainer.ema_model.eval()
hidden_dims = [128, 128, 128]


config = Config(
    model_class=TimeConditionedMLP,
    time_dim=128,
    input_dim=obs_dim,
    hidden_dims=hidden_dims,
    output_dim=12,
    savepath="",
)

device = torch.device('cuda')
model = config.make()
model.to(device)


ckpt_path = "logs/kuka_cube_stack_classifier_new3/value_0.99/state_80.pt"
ckpt = torch.load(ckpt_path)

model.load_state_dict(ckpt)


samples_list = []
frames = []

# models = [PosGuide(1, 3), PosGuide(1, 4), PosGuide(1, 2)]

counter = 0
map_tuple = {}
for i in range(4):
    for j in range(4):
        if i == j:
            continue

        map_tuple[(i, j)] = counter
        counter = counter + 1


# Red = block 0
# Green = block 1
# Blue = block 2
# Yellow block 3


rewards =  []
b_batch = []
time_batch = []
import time

for i in tqdm(range(20)):  #100
    start = time.time()
    reward, b_min = eval_episode(model, env, dataset, idx=i)
    end = time.time()
    # assert False
    rewards.append(reward)
    b_batch.append(b_min.cpu().numpy())
    time_batch.append(end-start)
    print("rewards mean: ", np.mean(rewards))
    print("rewards std: ", np.std(rewards) / len(rewards) ** 0.5)
    print("safety: ", np.min(b_batch))
    print("computation time: ", np.mean(time_batch))

exit()
import pdb
pdb.set_trace()

samples_full_list = np.array(samples_full_list)
np.save("execution.npy", samples_full_list)

# writer = get_writer("full_execution.mp4")

for frame in frames:
    writer.append_data(frame)

writer.close()
import pdb
pdb.set_trace()
assert False

# samples_next = trainer.ema_model.guided_conditional_sample(model, 1, conditions)
# samples_next = trainer.ema_model.conditional_sample(1, conditions)
samples = torch.cat(samples_list, dim=-2)


# samples = trainer.ema_model.conditional_sample(1, conditions)
samples = torch.clamp(samples, -1, 1)
samples_unscale = (samples + 1) * 0.5
samples = dataset.unnormalize(samples_unscale)



# x = x = (x + 1) * 0.5
# x = dataset.unnormalize(x)

samples = to_np(samples.squeeze(0).squeeze(0))
postprocess(samples, renderer)


savepath = "execute_sim_11.mp4"

savepath = savepath.replace('.png', '.mp4')
writer = get_writer(savepath)

for img in imgs:
    writer.append_data(img)

writer.close()
