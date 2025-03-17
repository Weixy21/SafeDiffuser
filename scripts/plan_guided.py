import pdb
import os
# os.system('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wei/.mujoco/mujoco200/bin')
# os.system('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-515')
# python scripts/plan_guided.py --dataset walker2d-medium-expert-v2 --logbase logs
import diffuser.sampling as sampling
import diffuser.utils as utils

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')

args.batch_size = 1
#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)
value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

# import pdb; pdb.set_trace()
# data = {'norm_obs': dataset.fields.normed_observations, 'norm_act': dataset.fields.normed_actions, 'len': dataset.fields.path_lengths}
# import pickle
# output = open('./scripts/data_walker2d_medium.pkl', 'wb') 
# pickle.dump(data, output)
# output.close()


## initialize value guide
value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)  

#ln -s /usr/lib/nvidia /usr/lib/nvidia-515 # to address the issue: ERROR: Shadow framebuffer is not complete, error 0x8cd7 

## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)

logger = logger_config()
policy = policy_config()

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

env = dataset.env
comp_time = []
safety = []
scores = []
import time
for kk in range(10):

    observation = env.reset()

    ## observations for rendering
    rollout = [observation.copy()]
    total_reward = 0
    for t in range(args.max_episode_length):
        
        # if t % 10 == 0: print(args.savepath, flush=True)

        ## save state for rendering only
        state = env.state_vector().copy()

        ## format current observation for conditioning
        conditions = {0: observation}

        # utils.colab.run_diffusion(diffusion, dataset, observation, n_samples=1, device=args.device, horizon=320, guide=guide, sample_fn=sampling.n_step_guided_p_sample
        #     )
        start = time.time()
        action, samples, diffusion, b_min = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)
        end = time.time()
        if t == 0:
            safety.append(b_min.cpu().numpy())
            comp_time.append(end-start)
        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(action)

        ## print reward and score
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        print(
            f'step: {kk}/10 | t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
            f'values: {samples.values} | scale: {args.scale}',
            flush=True,
        )

        ## update rollout observations
        rollout.append(next_observation.copy())

        ## render every `args.vis_freq` steps
        # logger.log(t, samples, state, rollout, diffusion)

        if terminal:
            break

        observation = next_observation
    scores.append(score)

## write results to json file at `args.savepath`
# logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)
import numpy as np
comp_time = np.array(comp_time)
safety = np.array(safety)
scores = np.array(scores)

print("safety: ", np.min(safety))
print("score mean: ", np.mean(scores))
print("score std: ", np.std(scores))
print("computation time: ", np.mean(comp_time))