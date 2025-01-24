import json
import numpy as np
from os.path import join
import pdb
import os

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import torch

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-515
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wei/.mujoco/mujoco200/bin
#python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def smooth(diffusion):
    steps, horizon = diffusion.shape[0], diffusion.shape[1]
    diffusion_copy = diffusion.copy()
    for i in range(steps - 20, steps, 1):
        for j in range(5, horizon, 1):
            diffusion_copy[i,j,0:2] = np.mean(diffusion[i, j-5:j, 0:2], axis=0)
    
    return diffusion_copy

#---------------------------------- main loop ----------------------------------#
safe1_batch, safe2_batch = [], []
score_batch = []
comp_time = []
elbo_batch = []
success = 0
import time
for iter in range(1):   # num of testing runs
    print("step: ", iter, "/100")

    observation = env.reset()    #array([ 0.94875744,  8.93648809, -0.01347715,  0.06358764])
    observation = np.array([ 0.94875744,  2.93648809, -0.01347715,  0.06358764])   # fix the initial position and final destination for comparison (not needed for general testing)
    env.set_state(observation[0:2], observation[2:4]) ############################################################ same as the last line

    if args.conditional:
        print('Resetting target')
        env.set_target()

    ## set conditioning xy position to be the goal
    target = env._target
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0
    for t in range(env.max_episode_steps):

        state = env.state_vector().copy()

        ## can replan if desired, but the open-loop plans are good enough for maze2d
        ## that we really only need to plan once
        if t == 0:

            cond[0] = observation
            start = time.time()
            action, samples, diffusion_paths, safe1, safe2, elbo = policy(cond, batch_size=args.batch_size)
            end = time.time()
            comp_time.append(end-start)
            elbo_batch.append(elbo)
            
    #############################       single test
            # cond[0] = observation
            # action, samples, diffusion_paths, safe1, safe2 = policy(cond, batch_size=args.batch_size)  #policy.normalizer.normalizers['observations'].mins
            actions = samples.actions[0]
            sequence = samples.observations[0]
            diffusion_paths = diffusion_paths[0]

            
            ##################################################save videos/images
            fullpath = join(args.savepath, f'{iter}.png')
            renderer.composite(fullpath, samples.observations, ncol=1)
            #########################################s################# 8/3/2023
            # diffusion_sm = smooth(diffusion_paths)    # smooth the generated traj.
            diffusion_sm = diffusion_paths            # do not smooth the generated traj.
            renderer.render_diffusion(join(args.savepath, f'diffusion.mp4'), diffusion_sm)

            # makedirs(join(args.savepath, 'trap'))
            # fullpath = join(args.savepath, f'trap/{iter}.png')
            # renderer.composite(fullpath, samples.observations, ncol=1)

            diff_step = diffusion_sm.shape[0]  
            makedirs(join(args.savepath, 'png'))
            for kk in range(diff_step):
                imgpath = join(args.savepath, f'png/{kk}.png')
                renderer.composite(imgpath, diffusion_sm[kk:kk+1], ncol=1)
            ##################################################end saving videos/images

        # ####
        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
            

        ## can use actions or define a simple controller based on state predictions
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        
        # else:
        #     actions = actions[1:]
        #     if len(actions) > 1:
        #         action = actions[0]
        #     else:
        #         # action = np.zeros(2)
        #         action = -state[2:]
        #         pdb.set_trace()



        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        score = env.get_normalized_score(total_reward)

        ###############################################################################################
        # print(
        #     f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        #     f'{action}'
        # )

        # if 'maze2d' in args.dataset:
        #     xy = next_observation[:2]
        #     goal = env.unwrapped._target
        #     print(
        #         f'maze | pos: {xy} | goal: {goal}'
        #     )

        ## update rollout observations
        rollout.append(next_observation.copy())

        # logger.log(score=score, step=t)

        ###############################################################################################
        # if t % args.vis_freq == 0 or terminal:
        #     fullpath = join(args.savepath, f'{t}.png')

        #     if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)


        #     # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

        #     ## save rollout thus far
        #     renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)   ## debug

        #     # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        #     # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

        if terminal:
            break

        observation = next_observation

    if reward > 0.95:
        success = success + 1

    # score = 0

    safe1_batch.append(torch.cat([safe1[-1].unsqueeze(0).unsqueeze(0), torch.tensor(score).unsqueeze(0).unsqueeze(0).to(safe1.device)], dim = 1))
    safe2_batch.append(torch.cat([safe2[-1].unsqueeze(0).unsqueeze(0), torch.tensor(score).unsqueeze(0).unsqueeze(0).to(safe2.device)], dim = 1))
    score_batch.append(score)
    # logger.finish(t, env.max_episode_steps, score=score, value=0)

elbo_batch = np.array(elbo_batch)
print("elbo mean: ", np.mean(elbo_batch))
print("elbo std: ", np.std(elbo_batch))

score_batch = np.array(score_batch)
safe1_batch = torch.cat(safe1_batch, dim=0)
safe2_batch = torch.cat(safe2_batch, dim=0)
comp_time = np.array(comp_time)
print("safe1: ", torch.min(safe1_batch[:,0]).cpu().numpy())
print("safe2: ", torch.min(safe2_batch[:,0]).cpu().numpy())
print("score mean: ", np.mean(score_batch))
print("score std: ", np.std(score_batch))
print("computation time: ", np.mean(comp_time))
print("success rate: ", success)


exit()

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 4), facecolor='white')
ax1 = fig.add_subplot(121, frameon=False)
ax2 = fig.add_subplot(122, frameon=False)
plt.show(block=False)

ax1.cla()
ax1.set_title('Trajectories')
ax1.set_xlabel('score')
ax1.set_ylabel('min. S-spec')
ax1.plot(safe1_batch.cpu().numpy()[:,1], safe1_batch.cpu().numpy()[:,0], 'r*', label = 'ground truth')

ax2.cla()
ax2.set_title('Trajectories')
ax2.set_xlabel('score')
ax2.set_ylabel('min. C-spec')
ax2.plot(safe2_batch.cpu().numpy()[:,1], safe2_batch.cpu().numpy()[:,0], 'r*', label = 'ground truth')

imgpath = join(args.savepath, f'stat.png')

plt.savefig(imgpath)

import pdb; pdb.set_trace()

exit()






## save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
