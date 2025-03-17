from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb
from torch.autograd import Variable
from qpth.qp import QPFunction, QPSolvers

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


Sample = namedtuple('Sample', 'trajectories values chains')


@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.mean = 0  # for normalization
        self.std = 0
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs): 
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x_t = x.clone()
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)

            ##########################################walker2d
            # x, b_min = self.GD(x_t, x)  # 13, 23
            # x, b_min = self.Shield(x_t, x)  # 11
            # x, b_min = self.invariance(x_t, x)  # 12
            # x, b_min = self.invariance_cf(x_t, x)   #closed form
            # x, b_min = self.invariance_cpx(x_t, x)  # 22:
            # x, b_min = self.invariance_cpx_cf(x_t, x) #closed form

            ##########################################hopper
            # x, b_min = self.GD_hopper(x_t, x) # 13, 23
            # x, b_min = self.Shield_hopper(x_t, x)  # 11
            # x, b_min = self.invariance_hopper(x_t, x)  # 12
            # x, b_min = self.invariance_hopper_cf(x_t, x)   #closed form
            # x, b_min = self.invariance_hopper_cpx(x_t, x)  # 22
            x, b_min = self.invariance_hopper_cpx_cf(x_t, x)  #closed form

            ##########################################cheetah
            # x, b_min = self.invariance_cheetah(x_t, x)
            
            x = apply_conditioning(x, cond, self.action_dim)

            ############################ diffuser only
            # height = 1.4   #1.3    walker2d
            # height = (height - self.mean[0]) / self.std[0]
            # b = height - x[:,6:7]  - 0.1*x[:,15:16]
            # b_min = torch.min(b)

            # height = 1.6   #1.5      hopper
            # height = (height - self.mean[0]) / self.std[0]
            # b = height - x[:,3:4] - 0.1*x[:,9:10]
            # b_min = torch.min(b)

            # progress.update({'t': i, 'vmin': values.min().item(), 'vmax': b_min.item()}) 
            # progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()
        # pdb.set_trace()  #unx = x[0,:,6:].cpu().numpy()*self.std + self.mean

        x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain), b_min

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs): 
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, return_chain = True, **sample_kwargs)    # debug

###################################################################walker2d  
    @torch.no_grad()   #only for sampling
    def invariance(self, x, xp1):

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.3
        height = (height - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        b = height - x[:,6:7] # - 0.1*x[:,15:16]   # - 0.01  # for robustness
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,6:7])
        #Lgbu2 = -0.1*torch.ones_like(x[:,6:7])
  
        G = torch.cat([-Lgbu1], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b
        
   
        q = -torch.cat([ref[:,6:7]], dim = 1).to(G.device)  #, ref[:,15:16]
        Q = Variable(torch.eye(1))
        Q = Q.unsqueeze(0).expand(nBatch, 1, 1).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,6:7] = x[:,6:7] + out[:,0:1]
        # rt[:,15:16] = x[:,15:16] + out[:,1:2]
        # print(out[0:4,0:1])
        rt = rt.unsqueeze(0)
        return rt, torch.min(b)  # + 0.01  # for robustness
    
    @torch.no_grad()   #only for sampling
    def invariance_cf(self, x, xp1):

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.3
        height = (height - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        b = height - x[:,6:7] # - 0.1*x[:,15:16]    # - 0.01  # for robustness
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,6:7])
        #Lgbu2 = -0.1*torch.ones_like(x[:,6:7])
  
        G0 = torch.cat([-Lgbu1], dim = 1)
        k = 1
        h0 = Lfb + k*b

        Lgbu1 = 1*torch.ones_like(x[:,6:7])
        G1 = torch.cat([-Lgbu1], dim = 1)
        h1 = Lfb + k*(x[:,6:7] + 10)

        q = -torch.cat([ref[:,6:7]], dim = 1).to(G0.device)  #, ref[:,15:16]

        y1_bar = 1*G0  # H or Q = identity matrix
        y2_bar = 1*G1
        u_bar = -1*q
        p1_bar = h0 - torch.sum(G0*u_bar,dim = 1).unsqueeze(1)
        p2_bar = h1 - torch.sum(G1*u_bar,dim = 1).unsqueeze(1)

        G = torch.cat([torch.sum(y1_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y1_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0)], dim = 0)
        #G = 1*[y1_bar*y1_bar', y1_bar*y2_bar'; y2_bar*y1_bar', y2_bar*y2_bar']
        w_p1_bar = torch.clamp(p1_bar, max=0)
        w_p2_bar = torch.clamp(p2_bar, max=0)

        # G 0-(1,1), 1-(1,2), 2-(2,1), 3-(2,2)
        lambda1 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, torch.zeros_like(p1_bar), torch.where(G[1]*w_p1_bar < G[0]*p2_bar, w_p1_bar/G[0], torch.clamp(G[3]*p1_bar - G[2]*p2_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))
        
        lambda2 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, w_p2_bar/G[3], torch.where(G[1]*w_p1_bar < G[0]*p2_bar, torch.zeros_like(p1_bar), torch.clamp(G[0]*p2_bar - G[1]*p1_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))

        out = lambda1*y1_bar + lambda2*y2_bar + u_bar
        rt = xp1.clone()      
        rt[:,6:7] = x[:,6:7] + out[:,0:1]
        # print(out[0:4,0:1])
        rt = rt.unsqueeze(0)

        return rt, torch.min(b)  # + 0.01  # for robustness
    
    
    @torch.no_grad()   #only for sampling
    def invariance_cpx(self, x, xp1):

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.4
        height = (height - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        b = height - x[:,6:7] - 0.1*x[:,15:16]   # - 0.01  # for robustness
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,6:7])
        Lgbu2 = -0.1*torch.ones_like(x[:,6:7])
  
        G = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b
        
   
        q = -torch.cat([ref[:,6:7], ref[:,15:16]], dim = 1).to(G.device)  #
        Q = Variable(torch.eye(2))
        Q = Q.unsqueeze(0).expand(nBatch, 2, 2).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,6:7] = x[:,6:7] + out[:,0:1]
        rt[:,15:16] = x[:,15:16] + out[:,1:2]
        rt = rt.unsqueeze(0)
        return rt, torch.min(b)  # + 0.01  # for robustness
    
    @torch.no_grad()   #only for sampling
    def invariance_cpx_cf(self, x, xp1):

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.4
        height = (height - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        b = height - x[:,6:7] - 0.1*x[:,15:16] # - 0.01  # for robustness
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,6:7])
        Lgbu2 = -0.1*torch.ones_like(x[:,6:7])
  
        G0 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        k = 1
        h0 = Lfb + k*b
        
        Lgbu1 = 1*torch.ones_like(x[:,6:7])
        Lgbu2 = 0.1*torch.ones_like(x[:,6:7])
        G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        h1 = Lfb + k*(x[:,6:7] + 0.1*x[:,15:16] + 10)
   
        q = -torch.cat([ref[:,6:7], ref[:,15:16]], dim = 1).to(G0.device)  #

        y1_bar = 1*G0  # H or Q = identity matrix
        y2_bar = 1*G1
        u_bar = -1*q
        p1_bar = h0 - torch.sum(G0*u_bar,dim = 1).unsqueeze(1)
        p2_bar = h1 - torch.sum(G1*u_bar,dim = 1).unsqueeze(1)

        G = torch.cat([torch.sum(y1_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y1_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0)], dim = 0)
        #G = 1*[y1_bar*y1_bar', y1_bar*y2_bar'; y2_bar*y1_bar', y2_bar*y2_bar']
        w_p1_bar = torch.clamp(p1_bar, max=0)
        w_p2_bar = torch.clamp(p2_bar, max=0)

        # G 0-(1,1), 1-(1,2), 2-(2,1), 3-(2,2)
        lambda1 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, torch.zeros_like(p1_bar), torch.where(G[1]*w_p1_bar < G[0]*p2_bar, w_p1_bar/G[0], torch.clamp(G[3]*p1_bar - G[2]*p2_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))
        
        lambda2 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, w_p2_bar/G[3], torch.where(G[1]*w_p1_bar < G[0]*p2_bar, torch.zeros_like(p1_bar), torch.clamp(G[0]*p2_bar - G[1]*p1_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))

        out = lambda1*y1_bar + lambda2*y2_bar + u_bar

        rt = xp1.clone()      
        rt[:,6:7] = x[:,6:7] + out[:,0:1]
        rt[:,15:16] = x[:,15:16] + out[:,1:2]
        rt = rt.unsqueeze(0)
        return rt, torch.min(b) # + 0.01  # for robustness

###################################################################hopper  
    @torch.no_grad()   #only for sampling
    def invariance_hopper(self, x, xp1):

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.5
        height = (height - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        b = height - x[:,3:4] # - 0.1*x[:,9:10]   # - 0.01  # for robustness
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,3:4])
        #Lgbu2 = -0.1*torch.ones_like(x[:,3:4])
  
        G = torch.cat([-Lgbu1], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b
        
   
        q = -torch.cat([ref[:,3:4]], dim = 1).to(G.device)  #, ref[:,15:16]
        Q = Variable(torch.eye(1))
        Q = Q.unsqueeze(0).expand(nBatch, 1, 1).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,3:4] = x[:,3:4] + out[:,0:1]
        # rt[:,15:16] = x[:,15:16] + out[:,1:2]
        rt = rt.unsqueeze(0)
        return rt, torch.min(b)  # + 0.01  # for robustness
    
    @torch.no_grad()   #only for sampling
    def invariance_hopper_cf(self, x, xp1):

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.5
        height = (height - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        b = height - x[:,3:4] # - 0.1*x[:,9:10]  # - 0.01  # for robustness
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,3:4])
        #Lgbu2 = -0.1*torch.ones_like(x[:,3:4])
  
        G0 = torch.cat([-Lgbu1], dim = 1)
        k = 1
        h0 = Lfb + k*b

        Lgbu1 = 1*torch.ones_like(x[:,3:4])
        G1 = torch.cat([-Lgbu1], dim = 1)
        h1 = Lfb + k*(x[:,3:4] + 10)
        
   
        q = -torch.cat([ref[:,3:4]], dim = 1).to(G0.device)  #, ref[:,15:16]

        y1_bar = 1*G0  # H or Q = identity matrix
        y2_bar = 1*G1
        u_bar = -1*q
        p1_bar = h0 - torch.sum(G0*u_bar,dim = 1).unsqueeze(1)
        p2_bar = h1 - torch.sum(G1*u_bar,dim = 1).unsqueeze(1)

        G = torch.cat([torch.sum(y1_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y1_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0)], dim = 0)
        #G = 1*[y1_bar*y1_bar', y1_bar*y2_bar'; y2_bar*y1_bar', y2_bar*y2_bar']
        w_p1_bar = torch.clamp(p1_bar, max=0)
        w_p2_bar = torch.clamp(p2_bar, max=0)

        # G 0-(1,1), 1-(1,2), 2-(2,1), 3-(2,2)
        lambda1 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, torch.zeros_like(p1_bar), torch.where(G[1]*w_p1_bar < G[0]*p2_bar, w_p1_bar/G[0], torch.clamp(G[3]*p1_bar - G[2]*p2_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))
        
        lambda2 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, w_p2_bar/G[3], torch.where(G[1]*w_p1_bar < G[0]*p2_bar, torch.zeros_like(p1_bar), torch.clamp(G[0]*p2_bar - G[1]*p1_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))

        out = lambda1*y1_bar + lambda2*y2_bar + u_bar
        rt = xp1.clone()      
        rt[:,3:4] = x[:,3:4] + out[:,0:1]
        # print(out[0:4,0:1])
        rt = rt.unsqueeze(0)

        return rt, torch.min(b)  # + 0.01  # for robustness
    
    @torch.no_grad()   #only for sampling
    def invariance_hopper_cpx(self, x, xp1):

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.6
        height = (height - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        b = height - x[:,3:4] - 0.1*x[:,9:10]   # - 0.01  # for robustness
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,3:4])
        Lgbu2 = -0.1*torch.ones_like(x[:,3:4])
  
        G = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b
        
   
        q = -torch.cat([ref[:,3:4], ref[:,9:10]], dim = 1).to(G.device)  #
        Q = Variable(torch.eye(2))
        Q = Q.unsqueeze(0).expand(nBatch, 2, 2).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,3:4] = x[:,3:4] + out[:,0:1]
        rt[:,9:10] = x[:,9:10] + out[:,1:2]
        rt = rt.unsqueeze(0)
        return rt, torch.min(b)  # + 0.01  # for robustness
    
    @torch.no_grad()   #only for sampling
    def invariance_hopper_cpx_cf(self, x, xp1):

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.6
        height = (height - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        b = height - x[:,3:4] - 0.1*x[:,9:10]   # - 0.01  # for robustness
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,3:4])
        Lgbu2 = -0.1*torch.ones_like(x[:,3:4])
  
        G0 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        k = 1
        h0 = Lfb + k*b

        Lgbu1 = 1*torch.ones_like(x[:,3:4])
        Lgbu2 = 0.1*torch.ones_like(x[:,3:4])
        G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        h1 = Lfb + k*(x[:,3:4] + 0.1*x[:,9:10] + 10)    
   
        q = -torch.cat([ref[:,3:4], ref[:,9:10]], dim = 1).to(G0.device)  #

        y1_bar = 1*G0  # H or Q = identity matrix
        y2_bar = 1*G1
        u_bar = -1*q
        p1_bar = h0 - torch.sum(G0*u_bar,dim = 1).unsqueeze(1)
        p2_bar = h1 - torch.sum(G1*u_bar,dim = 1).unsqueeze(1)

        G = torch.cat([torch.sum(y1_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y1_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0)], dim = 0)
        #G = 1*[y1_bar*y1_bar', y1_bar*y2_bar'; y2_bar*y1_bar', y2_bar*y2_bar']
        w_p1_bar = torch.clamp(p1_bar, max=0)
        w_p2_bar = torch.clamp(p2_bar, max=0)

        # G 0-(1,1), 1-(1,2), 2-(2,1), 3-(2,2)
        lambda1 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, torch.zeros_like(p1_bar), torch.where(G[1]*w_p1_bar < G[0]*p2_bar, w_p1_bar/G[0], torch.clamp(G[3]*p1_bar - G[2]*p2_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))
        
        lambda2 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, w_p2_bar/G[3], torch.where(G[1]*w_p1_bar < G[0]*p2_bar, torch.zeros_like(p1_bar), torch.clamp(G[0]*p2_bar - G[1]*p1_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))

        out = lambda1*y1_bar + lambda2*y2_bar + u_bar

        rt = xp1.clone()      
        rt[:,3:4] = x[:,3:4] + out[:,0:1]
        rt[:,9:10] = x[:,9:10] + out[:,1:2]
        rt = rt.unsqueeze(0)
        return rt, torch.min(b)   # + 0.01  # for robustness

###################################################################cheetah     
    @torch.no_grad()   #only for sampling
    def invariance_cheetah(self, x, xp1):

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        radius = 0.4
        radius = (radius - self.mean[0]) / self.std[0]
        cx = 4
        cy = -0.2
        cx = (cx - self.mean[14]) / self.std[14]
        cy = (cy - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        xpos = torch.cumsum(x[:,14:15], dim=0) * 0.05

        b = (xpos - cx)**2 + (x[:,6:7] - cy)**2 - radius**2 
        Lfb = 0 
        Lgbu1 = 2*(x[:,6:7] - cy)
  
        G = torch.cat([-Lgbu1], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b
        
   
        q = -torch.cat([ref[:,6:7]], dim = 1).to(G.device) 
        Q = Variable(torch.eye(1))
        Q = Q.unsqueeze(0).expand(nBatch, 1, 1).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,6:7] = x[:,6:7] + out[:,0:1]

        rt = rt.unsqueeze(0)
        return rt, torch.min(b)
    
    @torch.no_grad()   #only for sampling
    def invariance_cheetah_cpx(self, x, xp1):

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.4
        height = (height - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        b = height - x[:,6:7] - 0.1*x[:,15:16] 
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,6:7])
        Lgbu2 = -0.1*torch.ones_like(x[:,6:7])
  
        G = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b
        
   
        q = -torch.cat([ref[:,6:7], ref[:,15:16]], dim = 1).to(G.device)  #
        Q = Variable(torch.eye(2))
        Q = Q.unsqueeze(0).expand(nBatch, 2, 2).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,6:7] = x[:,6:7] + out[:,0:1]
        rt[:,15:16] = x[:,15:16] + out[:,1:2]
        rt = rt.unsqueeze(0)
        return rt, torch.min(b)

####################################################################shield    
    @torch.no_grad()   #Walker2d
    def Shield(self, x0, xp10):

        x = x0.clone()
        xp1 = xp10.clone()

        xp1 = xp1.squeeze(0)

        nBatch = xp1.shape[0]

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.3  
        height = (height - self.mean[0]) / self.std[0]

        ############################################ceiling
        b = height - xp1[:,6:7] # - 0.1*x[:,15:16] 

        for k in range(nBatch):
            if b[k, 0] < 0: 
                xp1[k,6] = height

        b = height - xp1[:,6:7]

        xp1 = xp1.unsqueeze(0)
        return xp1, torch.min(b[:,0])
    
    @torch.no_grad()   #Hopper
    def Shield_hopper(self, x0, xp10):

        x = x0.clone()
        xp1 = xp10.clone()

        xp1 = xp1.squeeze(0)

        nBatch = xp1.shape[0]

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.5 
        height = (height - self.mean[0]) / self.std[0]

        ############################################ceiling
        b = height - xp1[:,3:4] 

        for k in range(nBatch):
            if b[k, 0] < 0: 
                xp1[k,3] = height

        b = height - xp1[:,3:4]

        xp1 = xp1.unsqueeze(0)
        return xp1, torch.min(b[:,0])

###################################################################GD     
    @torch.no_grad()   #walker2d
    def GD(self, x0, xp10):

        x = x0.clone()
        xp1 = xp10.clone()

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.4  #1.3
        height = (height - self.mean[0]) / self.std[0]

        ############################################ceiling
        b = height - xp1[:,6:7]  - 0.1*x[:,15:16] 


        for k in range(nBatch):
            if b[k, 0] < 0:  # 0
                # u = -0.1
                u = -0.05
                xp1[k,6] = x[k,6] + u

                u2 = -0.05*10
                xp1[k,15] = x[k, 15] + u2

        xp1 = xp1.unsqueeze(0)
        return xp1, torch.min(b[:,0])
    
    @torch.no_grad()   #Hopper
    def GD_hopper(self, x0, xp10):

        x = x0.clone()
        xp1 = xp10.clone()

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.6  # 1.5
        height = (height - self.mean[0]) / self.std[0]

        ############################################ceiling
        b = height - x[:,3:4] - 0.1*x[:,9:10]  


        for k in range(nBatch):
            if b[k, 0] < 0:  # 0
                # u = -0.1
                u = -0.05
                xp1[k,3] = x[k,3] + u

                u2 = -0.05*10
                xp1[k,9] = x[k, 9] + u2

        xp1 = xp1.unsqueeze(0)
        return xp1, torch.min(b[:,0])


    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)


class ValueDiffusion(GaussianDiffusion):

    def p_losses(self, x_start, cond, target, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        pred = self.model(x_noisy, cond, t)

        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x, cond, t):
        return self.model(x, cond, t)

