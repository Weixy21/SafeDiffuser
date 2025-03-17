import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gym
import mujoco_py as mjc
import warnings
import pdb

from .arrays import to_np
from .video import save_video, save_videos

from diffuser.datasets.d4rl import load_environment

#-----------------------------------------------------------------------------#
#------------------------------- helper structs ------------------------------#
#-----------------------------------------------------------------------------#

def env_map(env_name):
    '''
        map D4RL dataset names to custom fully-observed
        variants for rendering
    '''
    if 'halfcheetah' in env_name:
        return 'HalfCheetahFullObs-v2'
    elif 'hopper' in env_name:
        return 'HopperFullObs-v2'
    elif 'walker2d' in env_name:
        return 'Walker2dFullObs-v2'
    else:
        return env_name

#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#

def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

#-----------------------------------------------------------------------------#
#---------------------------------- renderers --------------------------------#
#-----------------------------------------------------------------------------#

class MuJoCoRenderer:
    '''
        default mujoco renderer
    '''

    def __init__(self, env):
        if type(env) is str:
            env = env_map(env)
            self.env = gym.make(env)
        else:
            self.env = env
        ## - 1 because the envs in renderer are fully-observed
        self.observation_dim = np.prod(self.env.observation_space.shape) - 1
        self.action_dim = np.prod(self.env.action_space.shape)
        try:
            self.viewer = mjc.MjRenderContextOffscreen(self.env.sim)
            # pdb.set_trace()
            # self.viewer2 = mjc.MjViewer(self.env.sim)
            # self.viewer2 = mjc.MjRenderContext(self.env.sim)   #debug
        except:
            print('[ utils/rendering ] Warning: could not initialize offscreen renderer')
            self.viewer = None
        self.saved_frames = 0
        self.saved_masks = []

    def pad_observation(self, observation):
        state = np.concatenate([
            np.zeros(1),
            observation,
        ])
        return state

    def pad_observations(self, observations):
        qpos_dim = self.env.sim.data.qpos.size
        ## xpos is hidden
        xvel_dim = qpos_dim - 1
        xvel = observations[:, xvel_dim]
 
        xpos = np.cumsum(xvel) * self.env.dt
        states = np.concatenate([
            xpos[:,None],
            observations,
        ], axis=-1)
        return states

    def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None, conditions=None):

        if type(dim) == int:
            dim = (dim, dim)

        if self.viewer is None:
            return np.zeros((*dim, 3), np.uint8)

        if render_kwargs is None:
            xpos = observation[0] if not partial else 0
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 3,
                'lookat': [xpos, -0.5, 1],
                'elevation': -20
            }

        for key, val in render_kwargs.items():
            if key == 'lookat':
                self.viewer.cam.lookat[:] = val[:]
                # self.viewer2.cam.lookat[:] = val[:]  #debug
            else:
                setattr(self.viewer.cam, key, val)
                # setattr(self.viewer2.cam, key, val)  #debug

        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation

        qpos_dim = self.env.sim.data.qpos.size
        if not qvel or state.shape[-1] == qpos_dim:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])

        set_state(self.env, state)

        self.viewer.render(*dim)
        # self.viewer2.render()
        # self.viewer2.render(*dim)  #debug
        data = self.viewer.read_pixels(*dim, depth=False) 
        # data2 = self.viewer2.read_pixels(*dim, depth=False, segmentation = True)  #debug
        # data2 = self.viewer2._read_pixels_as_in_window(resolution=dim)
        # img = self.env.sim.render(*dim)
        # pdb.set_trace()
        data = data[::-1, :, :]
        return data

    def _renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def renders(self, samples, partial=False, saveframes=None, readframes=None, index = 600, **kwargs):
        if partial:
            samples = self.pad_observations(samples)
            partial = False

        if readframes is not None and readframes:
            sample_images = self.saved_frames
        else:
            sample_images = self._renders(samples, partial=partial, **kwargs)
        if saveframes is not None and saveframes:
            self.saved_frames = sample_images

        composite = np.ones_like(sample_images[0]) * 255
        import torch
        # import pdb; pdb.set_trace()
        composite[81:83,:,:] = torch.tensor([255,0,0]).expand(2,256,3)  #172, 82:84, walker2d roof    ####1024 11/5/2024
        # composite[67:69,:,:] = torch.tensor([255,0,0]).expand(2,1024,3)  #172, 82:84, hopper roof 
        # composite[120:122,:,:] = torch.tensor([255,0,0]).expand(2,1024,3)  #172, 82:84, halfcheetah roof
        i = 0
        # for img in sample_images[:index]:
        for img in sample_images[range(0, index, 20)]:
            # pdb.set_trace()  #imageio.imsave('logs/walker2d-medium-expert-v2/plans/H600_T20_d0.99/0/test.png', img)
            if readframes is not None and readframes:
                mask = self.saved_masks[i]
                i = i+1
            else:
                mask = get_image_mask(img)
            composite[mask] = img[mask]
            if saveframes is not None and saveframes:
                self.saved_masks.append(mask)

        return composite

    def composite(self, savepath, paths, dim=(1024, 256), **kwargs):

        render_kwargs = {
            'trackbodyid': 2,
            'distance': 7, #10
            'lookat': [7, 2, 0.8], #[5, 2, 0.5]
            'elevation': 0
        }
        images = []
        for path in paths:
            ## [ H x obs_dim ]
            path = atmost_2d(path)
            img = self.renders(to_np(path), dim=dim, partial=True, qvel=True, render_kwargs=render_kwargs, **kwargs) 
            images.append(img)
        images = np.concatenate(images, axis=0)

        if savepath is not None:
            imageio.imsave(savepath, images)
            print(f'Saved {len(paths)} samples to: {savepath}')

        return images

    def render_rollout(self, savepath, states, **video_kwargs):
        if type(states) is list: states = np.array(states)
        images = self._renders(states, partial=True)
        save_video(savepath, images, **video_kwargs)

    def render_plan(self, savepath, actions, observations_pred, state, fps=30):
        ## [ batch_size x horizon x observation_dim ]
        observations_real = rollouts_from_state(self.env, state, actions)

        ## there will be one more state in `observations_real`
        ## than in `observations_pred` because the last action
        ## does not have an associated next_state in the sampled trajectory
        observations_real = observations_real[:,:-1]

        images_pred = np.stack([
            self._renders(obs_pred, partial=True)
            for obs_pred in observations_pred
        ])

        images_real = np.stack([
            self._renders(obs_real, partial=False)
            for obs_real in observations_real
        ])

        ## [ batch_size x horizon x H x W x C ]
        images = np.concatenate([images_pred, images_real], axis=-2)
        save_videos(savepath, *images)

    def render_diffusion(self, savepath, diffusion_path, **video_kwargs):
        '''
            diffusion_path : [ n_diffusion_steps x batch_size x 1 x horizon x joined_dim ]
        '''
        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [10, 2, 0.5],
            'elevation': 0,
        }

        diffusion_path = to_np(diffusion_path)

        n_diffusion_steps, batch_size, _, horizon, joined_dim = diffusion_path.shape

        frames = []
        for t in reversed(range(n_diffusion_steps)):
            print(f'[ utils/renderer ] Diffusion: {t} / {n_diffusion_steps}')

            ## [ batch_size x horizon x observation_dim ]
            states_l = diffusion_path[t].reshape(batch_size, horizon, joined_dim)[:, :, :self.observation_dim]

            frame = []
            for states in states_l:
                img = self.composite(None, states, dim=(1024, 256), partial=True, qvel=True, render_kwargs=render_kwargs)
                frame.append(img)
            frame = np.concatenate(frame, axis=0)

            frames.append(frame)

        save_video(savepath, frames, **video_kwargs)
    
    def render_diffusion_samp(self, savepath, diffusion_path, **video_kwargs):
        '''
            diffusion_path : [batch_size x n_diffusion_steps x horizon x joined_dim ]
        '''
        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [10, 2, 0.5],
            'elevation': 0,
        }

        diffusion_path = to_np(diffusion_path)

        batch_size, n_diffusion_steps, horizon, joined_dim = diffusion_path.shape
        print(np.max(diffusion_path[0,-1,:,0]))
   
        # png_diff = diffusion_path[:,:, range(0, horizon, 10), :]    #debug
        frames = []
        for t in range(n_diffusion_steps):
            print(f'[ utils/renderer ] Diffusion: {t} / {n_diffusion_steps}')
            
            ## [ batch_size x horizon x observation_dim ]
            states_l = diffusion_path[:,t,:,:].reshape(batch_size, horizon, joined_dim)[:, :, :self.observation_dim]  

            frame = []
            img = self.composite(os.path.join(savepath, f'{t}.png'), states_l, dim=(1024, 256))
            frame.append(img)
            frame = np.concatenate(frame, axis=0)

            frames.append(frame)

        # save_video(savepath, frames, **video_kwargs)
    
    def render_diffusion_samp_c(self, savepath, diffusion_path, **video_kwargs):
        '''
            diffusion_path : [batch_size x n_diffusion_steps x horizon x joined_dim ]
        '''
        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [10, 2, 0.5],
            'elevation': 0,
        }

        diffusion_path = to_np(diffusion_path)

        batch_size, n_diffusion_steps, horizon, joined_dim = diffusion_path.shape
        print(np.max(diffusion_path[0,-1,:,0]))
        # pdb.set_trace()
        frames = []
        for t in range(n_diffusion_steps):
            print(f'[ utils/renderer ] Diffusion: {t} / {n_diffusion_steps}')
            
            ## [ batch_size x horizon x observation_dim ]
            states_l = diffusion_path[:,t,:,:].reshape(batch_size, horizon, joined_dim)[:, :, :self.observation_dim]

            frame = []
            if t==n_diffusion_steps-1:
                img = self.composite(None, states_l, dim=(1024, 256), saveframes = True, readframes = False)
            else:
                img = self.composite(None, states_l, dim=(1024, 256), saveframes = False, readframes = False)
            frame.append(img)
            frame = np.concatenate(frame, axis=0)
            for ii in range(20):  # compensate x10
                frames.append(frame)
        
        for i in reversed(range(1, horizon, 1)):
            print(f'[ utils/renderer ] Planning: {i} / {horizon}')
            
            ## [ batch_size x horizon x observation_dim ]
            states_l = diffusion_path[:,-1,:,:].reshape(batch_size, horizon, joined_dim)[:, :i, :self.observation_dim]

            frame = []
            img = self.composite(None, states_l, dim=(1024, 256), readframes = True, index = i)
            frame.append(img)
            frame = np.concatenate(frame, axis=0)

            frames.append(frame)

        for i in range(1, horizon, 1):
            print(f'[ utils/renderer ] Planning: {i} / {horizon}')
            
            ## [ batch_size x horizon x observation_dim ]
            states_l = diffusion_path[:,-1,:,:].reshape(batch_size, horizon, joined_dim)[:, :i, :self.observation_dim]

            frame = []
            img = self.composite(None, states_l, dim=(1024, 256), readframes = True, index = i)
            frame.append(img)
            frame = np.concatenate(frame, axis=0)

            frames.append(frame)
        


        save_video(savepath, frames, **video_kwargs)

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

#-----------------------------------------------------------------------------#
#---------------------------------- rollouts ---------------------------------#
#-----------------------------------------------------------------------------#

def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f'[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, '
            f'but got state of size {state.size}')
        state = state[:qpos_dim + qvel_dim]

    env.set_state(state[:qpos_dim], state[qpos_dim:])

def rollouts_from_state(env, state, actions_l):
    rollouts = np.stack([
        rollout_from_state(env, state, actions)
        for actions in actions_l
    ])
    return rollouts

def rollout_from_state(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions)+1):
        ## if terminated early, pad with zeros
        observations.append( np.zeros(obs.size) )
    return np.stack(observations)
