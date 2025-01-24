from collections import namedtuple
# import numpy as np
import torch
import einops
import pdb

import diffuser.utils as utils
# from diffusion.datasets.preprocessing import get_policy_preprocess_fn

Trajectories = namedtuple('Trajectories', 'actions observations')
# GuidedTrajectories = namedtuple('GuidedTrajectories', 'actions observations value')

class Policy:

    def __init__(self, diffusion_model, normalizer):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def __call__(self, conditions, debug=False, batch_size=1):


        conditions = self._format_conditions(conditions, batch_size)

        ## batchify and move to tensor [ batch_size x observation_dim ]
        # observation_np = observation_np[None].repeat(batch_size, axis=0)
        # observation = utils.to_torch(observation_np, device=self.device)

        ## run reverse diffusion process
        self.diffusion_model.norm_mins = self.normalizer.normalizers['observations'].mins
        self.diffusion_model.norm_maxs = self.normalizer.normalizers['observations'].maxs
        sample, diffusion = self.diffusion_model(conditions)

        #if get elbo/NLL####################################################for elbo/NLL
        # import pickle
        # with open('./diffuser.pkl', 'rb') as f:  # load a data from diffuser as a baseline
        #     data = pickle.load(f)
        # diffuser_state = torch.tensor(data['gt']).float().to(self.device) 
        # f.close()
        # diff_num = diffusion.shape[1]
        # elbo = []
        # sum_elbo = 0
        # for step in range(0, diff_num-2, 1):
        #     timesteps = torch.full((batch_size,), diff_num-2 - step, device=self.device, dtype=torch.long)
        #     elboi = self.diffusion_model._vb_terms_bpd(diffuser_state, conditions, diffusion[:,step,:,:], timesteps)
        #     sum_elbo = sum_elbo + elboi
        #     elbo.append(elboi)
        # sum_elbo = sum_elbo.detach().cpu().numpy()[0]/255  #ave
        #else#####################################################################
        sum_elbo = 0
        #end#########################################################################


        # # print('ELBO:', elbo)
        # # print('ELBO ave:', sum_elbo/255)
        #######################################################################
        
        sample = utils.to_np(sample)
        diffusion = utils.to_np(diffusion)

        
        #####################################################for elbo, # save a data from diffuser as a baseline
        # data = {'gt': diffusion[:,-1,:,:]}
        # import pickle
        # output = open('./diffuser.pkl', 'wb') 
        # pickle.dump(data, output)
        # output.close()
        #######################################################################

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        # actions = np.tanh(actions)

        ## extract first action
        action = actions[0, 0]

        # if debug:
        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        normed_diffusion = diffusion[:,:,:,self.action_dim:]
        diffusions = self.normalizer.unnormalize(normed_diffusion, 'observations')

        # if deltas.shape[-1] < observation.shape[-1]:
        #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
        #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
        #     deltas = np.concatenate([deltas, padding], axis=-1)

        # ## [ batch_size x horizon x observation_dim ]
        # next_observations = observation_np + deltas.cumsum(axis=1)
        # ## [ batch_size x (horizon + 1) x observation_dim ]
        # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)

        trajectories = Trajectories(actions, observations)
        return action, trajectories, diffusions, self.diffusion_model.safe1, self.diffusion_model.safe2, sum_elbo
        # else:
        #     return action
