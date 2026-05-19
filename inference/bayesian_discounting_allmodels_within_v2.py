#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 2026
@author: benwagner

WITHIN-SUBJECTS PIPELINE (CAFE & FUTURE)
Models: M1, M3, M5, M_LOG, M_FRAC
Features: Baseline+Shift, Save/Load Architecture, WAIC, PPCs, Violin Plots.
"""

# %% [1] Imports and Master Setup
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import pyro
from pyro.optim import Adam
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from tqdm import tqdm

################################################################################################
#  CONTROL PANEL script for both within datasets. For haloperidol dataset see other script #####
################################################################################################


# Options: "M1", "M3", "M5", "M_LOG", "M_FRAC", or "ALL"
SELECTED_MODEL = "ALL"        
# Options: "FUTURE" or "CAFE"
SELECTED_DATASET = "CAFE" 

# --- THE NEW SAVE/LOAD TOGGLE ---
# True = Run SVI and save new samples to disk.
# False = Skip SVI, load saved samples, and just run WAIC/PPC/Plots.
TRAIN_MODELS = True 
# ==========================================

BETA_BOUND = 10.0  #max inverse temperature

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")


# %% [2] Data Loading & Preprocessing (Within-Subjects)
print(f"Initializing Dataset: {SELECTED_DATASET}")
if SELECTED_DATASET == "CAFE":
    with open('array_cafe_gamble.pkl', 'rb') as f:
        data = pickle.load(f)
    data = torch.tensor(data).to(device) 
    data = data.permute(1, 0, 2, 3)      
    data[:,:,:,5] = 20.0 

elif SELECTED_DATASET == "FUTURE":
    with open('array_future.pkl', 'rb') as f:
        data = pickle.load(f)
    data = torch.tensor(data).to(device) 
    data = data.permute(1, 0, 2, 3)      
    data[:,:,:,5] = 20.0 

else:
    raise ValueError("Invalid SELECTED_DATASET for Within-Subjects Pipeline.")
    
print(f"Data shape successfully set to: {data.shape}")


# %% [3] Model Definitions (Baseline + Shift Architecture)

# ----- M1: Square Root (2 Params) -----
def model_m1(data):
    num_params = 2
    num_agents, num_trials = data.shape[0], data.shape[2] 
    
    m_base = pyro.param('m_base', torch.zeros(num_params, device=device))
    s_base = pyro.param('s_base', torch.ones(num_params, device=device), constraint=dist.constraints.positive)
    m_shift = pyro.param('m_shift', torch.zeros(num_params, device=device))
    s_shift = pyro.param('s_shift', torch.ones(num_params, device=device), constraint=dist.constraints.positive)
    
    with pyro.plate('ag_idx', num_agents):
        base_dist = dist.Normal(torch.zeros(num_params, device=device), torch.ones(num_params, device=device)).to_event(1)
        locs_base = pyro.sample('locs_base', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(m_base, s_base)]))
        shift = pyro.sample('shift', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(m_shift, s_shift)]))

    locs_flat = torch.stack([locs_base, locs_base + shift], dim=1).reshape(num_agents * 2, num_params)
    data_flat = data.reshape(num_agents * 2, num_trials, data.shape[-1])

    with pyro.plate('data', num_agents * 2 * num_trials):
        a_param = torch.exp(locs_flat[:,0]).unsqueeze(-1).expand(-1, num_trials)
        beta = torch.sigmoid(locs_flat[:,1]).unsqueeze(-1).expand(-1, num_trials)*BETA_BOUND
        sigma_combine = a_param * torch.sqrt(data_flat[:,:,2])
        e_mean = (data_flat[:,:,3])/(1 + sigma_combine**2) + 1e-8
        
        log1 = torch.log(e_mean) * beta
        log2 = torch.log(data_flat[:,:,5]) * beta
        logits = torch.stack([log1, log2], dim=0)
        p = torch.softmax(logits, dim=0)[0]
    pyro.sample("obs", dist.Bernoulli(probs=p).to_event(2), obs=data_flat[:,:,4])

def guide_m1(data):
    num_params, num_agents = 2, data.shape[0]
    m_locs_base = pyro.param('m_locs_base', torch.zeros(num_agents, num_params, device=data.device))
    st_locs_base = pyro.param('st_locs_base', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    m_shift = pyro.param('m_shift_locs', torch.zeros(num_agents, num_params, device=data.device))
    st_shift = pyro.param('st_shift_locs', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    
    with pyro.plate('ag_idx', num_agents):
        locs_base = pyro.sample("locs_base", dist.MultivariateNormal(m_locs_base, scale_tril=st_locs_base))
        shift = pyro.sample("shift", dist.MultivariateNormal(m_shift, scale_tril=st_shift))
    return {'locs_base': locs_base, 'shift': shift}

#M3: Sigmoid (4 Params inkl beta) 
def model_m3(data):
    num_params = 4
    num_agents, num_trials = data.shape[0], data.shape[2]
    
    a_b = pyro.param('a_b', torch.ones(num_params, device=device), constraint=dist.constraints.positive)
    lam_b = pyro.param('lam_b', torch.ones(num_params, device=device), constraint=dist.constraints.positive)
    tau_b = pyro.sample('tau_b', dist.Gamma(a_b, a_b/lam_b).to_event(1)) 
    sig_b = pyro.deterministic('sig_b', 1/torch.sqrt(tau_b)) 
    m_b = pyro.param('m_b', torch.zeros(num_params, device=device))
    s_b = pyro.param('s_b', torch.ones(num_params, device=device), constraint=dist.constraints.positive)
    mu_b = pyro.sample('mu_b', dist.Normal(m_b, s_b*sig_b).to_event(1)) 
    
    m_s = pyro.param('m_s', torch.zeros(num_params, device=device))
    s_s = pyro.param('s_s', torch.ones(num_params, device=device), constraint=dist.constraints.positive)
    
    with pyro.plate('ag_idx', num_agents):
        base_dist = dist.Normal(torch.zeros(num_params, device=device), torch.ones(num_params, device=device)).to_event(1)
        locs_base = pyro.sample('locs_base', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(mu_b, sig_b)]))
        shift = pyro.sample('shift', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(m_s, s_s)]))

    locs_flat = torch.stack([locs_base, locs_base + shift], dim=1).reshape(num_agents * 2, num_params)
    data_flat = data.reshape(num_agents * 2, num_trials, data.shape[-1])

    with pyro.plate('data', num_agents * 2 * num_trials):
        sigma_rate = torch.exp(locs_flat[:,0]).unsqueeze(-1).expand(-1, num_trials)
        a_param = torch.exp(locs_flat[:,1]).unsqueeze(-1).expand(-1, num_trials)
        b_param = torch.exp(locs_flat[:,2]).unsqueeze(-1).expand(-1, num_trials)
        beta = torch.sigmoid(locs_flat[:,3]).unsqueeze(-1).expand(-1, num_trials)*BETA_BOUND
        sigma_combine = sigma_rate/(1+b_param*torch.exp(-a_param*data_flat[:,:,2]))
        e_mean = (data_flat[:,:,3])/(1 + sigma_combine**2) + 1e-8
        
        log1 = torch.log(e_mean) * beta
        log2 = torch.log(data_flat[:,:,5]) * beta
        logits = torch.stack([log1, log2], dim=0)
        p = torch.softmax(logits, dim=0)[0]
    pyro.sample("obs", dist.Bernoulli(probs=p).to_event(2), obs=data_flat[:,:,4])

def guide_m3(data):
    num_params, num_agents = 4, data.shape[0]
    trns = torch.distributions.biject_to(dist.constraints.positive)
    
    m_hyp_base = pyro.param('m_hyp_base', torch.zeros(2*num_params, device=data.device))
    st_hyp_base = pyro.param('scale_tril_hyp_base', torch.eye(2*num_params, device=data.device), constraint=dist.constraints.lower_cholesky)
    hyp_base = pyro.sample('hyp_base', dist.MultivariateNormal(m_hyp_base, scale_tril=st_hyp_base), infer={'is_auxiliary': True})
    
    unc_mu, unc_tau = hyp_base[..., :num_params], hyp_base[..., num_params:]
    c_tau = trns(unc_tau)
    
    ld_tau = -trns.inv.log_abs_det_jacobian(c_tau, unc_tau)
    ld_tau = dist.util.sum_rightmost(ld_tau, ld_tau.dim() - c_tau.dim() + 1)
    
    mu_b = pyro.sample("mu_b", dist.Delta(unc_mu, event_dim=1))
    tau_b = pyro.sample("tau_b", dist.Delta(c_tau, log_density=ld_tau, event_dim=1))

    m_locs_base = pyro.param('m_locs_base', torch.zeros(num_agents, num_params, device=data.device))
    st_locs_base = pyro.param('st_locs_base', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    m_shift = pyro.param('m_shift_locs', torch.zeros(num_agents, num_params, device=data.device))
    st_shift = pyro.param('st_shift_locs', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    
    with pyro.plate('ag_idx', num_agents):
        locs_base = pyro.sample("locs_base", dist.MultivariateNormal(m_locs_base, scale_tril=st_locs_base))
        shift = pyro.sample("shift", dist.MultivariateNormal(m_shift, scale_tril=st_shift))
        
    return {'tau_b': tau_b, 'mu_b': mu_b, 'locs_base': locs_base, 'shift': shift}

#M5: standard Hyperbolic (2 Params k + beta)
def model_m5(data):
    num_params = 2
    num_agents, num_trials = data.shape[0], data.shape[2] 
    
    m_base = pyro.param('m_base', torch.zeros(num_params, device=device))
    s_base = pyro.param('s_base', torch.ones(num_params, device=device), constraint=dist.constraints.positive)
    m_shift = pyro.param('m_shift', torch.zeros(num_params, device=device))
    s_shift = pyro.param('s_shift', torch.ones(num_params, device=device), constraint=dist.constraints.positive)
    
    with pyro.plate('ag_idx', num_agents):
        base_dist = dist.Normal(torch.zeros(num_params, device=device), torch.ones(num_params, device=device)).to_event(1)
        locs_base = pyro.sample('locs_base', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(m_base, s_base)]))
        shift = pyro.sample('shift', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(m_shift, s_shift)]))

    locs_flat = torch.stack([locs_base, locs_base + shift], dim=1).reshape(num_agents * 2, num_params)
    data_flat = data.reshape(num_agents * 2, num_trials, data.shape[-1])

    with pyro.plate('data', num_agents * 2 * num_trials):
        k_param = torch.exp(locs_flat[:,0]).unsqueeze(-1).expand(-1, num_trials)
        beta = torch.sigmoid(locs_flat[:,1]).unsqueeze(-1).expand(-1, num_trials)*BETA_BOUND
        e_mean = (data_flat[:,:,3]) / (1 + k_param * data_flat[:,:,2]) + 1e-8
        
        log1 = torch.log(e_mean) * beta
        log2 = torch.log(data_flat[:,:,5]) * beta
        logits = torch.stack([log1, log2], dim=0)
        p = torch.softmax(logits, dim=0)[0]
    pyro.sample("obs", dist.Bernoulli(probs=p).to_event(2), obs=data_flat[:,:,4])

def guide_m5(data):
    num_params, num_agents = 2, data.shape[0]
    m_locs_base = pyro.param('m_locs_base', torch.zeros(num_agents, num_params, device=data.device))
    st_locs_base = pyro.param('st_locs_base', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    m_shift = pyro.param('m_shift_locs', torch.zeros(num_agents, num_params, device=data.device))
    st_shift = pyro.param('st_shift_locs', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    
    with pyro.plate('ag_idx', num_agents):
        locs_base = pyro.sample("locs_base", dist.MultivariateNormal(m_locs_base, scale_tril=st_locs_base))
        shift = pyro.sample("shift", dist.MultivariateNormal(m_shift, scale_tril=st_shift))
    return {'locs_base': locs_base, 'shift': shift}

# alternative M_LOG: Logarithmic increase (3 Params)
def model_m_log(data):
    num_params = 3
    num_agents, num_trials = data.shape[0], data.shape[2] 
    
    m_base = pyro.param('m_base', torch.zeros(num_params, device=device))
    s_base = pyro.param('s_base', torch.ones(num_params, device=device), constraint=dist.constraints.positive)
    m_shift = pyro.param('m_shift', torch.zeros(num_params, device=device))
    s_shift = pyro.param('s_shift', torch.ones(num_params, device=device), constraint=dist.constraints.positive)
    
    with pyro.plate('ag_idx', num_agents):
        base_dist = dist.Normal(torch.zeros(num_params, device=device), torch.ones(num_params, device=device)).to_event(1)
        locs_base = pyro.sample('locs_base', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(m_base, s_base)]))
        shift = pyro.sample('shift', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(m_shift, s_shift)]))

    locs_flat = torch.stack([locs_base, locs_base + shift], dim=1).reshape(num_agents * 2, num_params)
    data_flat = data.reshape(num_agents * 2, num_trials, data.shape[-1])

    with pyro.plate('data', num_agents * 2 * num_trials):
        a_param = torch.exp(locs_flat[:,0]).unsqueeze(-1).expand(-1, num_trials)
        b_param = torch.exp(locs_flat[:,1]).unsqueeze(-1).expand(-1, num_trials)
        beta = torch.sigmoid(locs_flat[:,2]).unsqueeze(-1).expand(-1, num_trials)*BETA_BOUND
        sigma_combine = a_param * torch.log(1 + b_param * data_flat[:,:,2])
        e_mean = (data_flat[:,:,3]) / (1 + sigma_combine**2) + 1e-8
        
        log1 = torch.log(e_mean) * beta
        log2 = torch.log(data_flat[:,:,5]) * beta
        logits = torch.stack([log1, log2], dim=0)
        p = torch.softmax(logits, dim=0)[0]
    pyro.sample("obs", dist.Bernoulli(probs=p).to_event(2), obs=data_flat[:,:,4])

def guide_m_log(data):
    num_params, num_agents = 3, data.shape[0]
    m_locs_base = pyro.param('m_locs_base', torch.zeros(num_agents, num_params, device=data.device))
    st_locs_base = pyro.param('st_locs_base', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    m_shift = pyro.param('m_shift_locs', torch.zeros(num_agents, num_params, device=data.device))
    st_shift = pyro.param('st_shift_locs', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    
    with pyro.plate('ag_idx', num_agents):
        locs_base = pyro.sample("locs_base", dist.MultivariateNormal(m_locs_base, scale_tril=st_locs_base))
        shift = pyro.sample("shift", dist.MultivariateNormal(m_shift, scale_tril=st_shift))
    return {'locs_base': locs_base, 'shift': shift}

#M_FRAC: Fractional Power (3 Params) delay scaled by s (inspired from differences in time scaling)
def model_m_frac(data):
    num_params = 3
    num_agents, num_trials = data.shape[0], data.shape[2] 
    
    m_base = pyro.param('m_base', torch.zeros(num_params, device=device))
    s_base = pyro.param('s_base', torch.ones(num_params, device=device), constraint=dist.constraints.positive)
    m_shift = pyro.param('m_shift', torch.zeros(num_params, device=device))
    s_shift = pyro.param('s_shift', torch.ones(num_params, device=device), constraint=dist.constraints.positive)
    
    with pyro.plate('ag_idx', num_agents):
        base_dist = dist.Normal(torch.zeros(num_params, device=device), torch.ones(num_params, device=device)).to_event(1)
        locs_base = pyro.sample('locs_base', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(m_base, s_base)]))
        shift = pyro.sample('shift', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(m_shift, s_shift)]))

    locs_flat = torch.stack([locs_base, locs_base + shift], dim=1).reshape(num_agents * 2, num_params)
    data_flat = data.reshape(num_agents * 2, num_trials, data.shape[-1])

    with pyro.plate('data', num_agents * 2 * num_trials):
        a_param = torch.exp(locs_flat[:,0]).unsqueeze(-1).expand(-1, num_trials)
        s_param = torch.sigmoid(locs_flat[:,1]).unsqueeze(-1).expand(-1, num_trials) * 2.0 
        beta = torch.sigmoid(locs_flat[:,2]).unsqueeze(-1).expand(-1, num_trials)*BETA_BOUND
        sigma_combine = a_param * torch.pow(data_flat[:,:,2] + 1e-8, s_param)
        e_mean = (data_flat[:,:,3]) / (1 + sigma_combine**2) + 1e-8
        
        log1 = torch.log(e_mean) * beta
        log2 = torch.log(data_flat[:,:,5]) * beta
        logits = torch.stack([log1, log2], dim=0)
        p = torch.softmax(logits, dim=0)[0]
    pyro.sample("obs", dist.Bernoulli(probs=p).to_event(2), obs=data_flat[:,:,4])

def guide_m_frac(data):
    num_params, num_agents = 3, data.shape[0]
    m_locs_base = pyro.param('m_locs_base', torch.zeros(num_agents, num_params, device=data.device))
    st_locs_base = pyro.param('st_locs_base', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    m_shift = pyro.param('m_shift_locs', torch.zeros(num_agents, num_params, device=data.device))
    st_shift = pyro.param('st_shift_locs', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    
    with pyro.plate('ag_idx', num_agents):
        locs_base = pyro.sample("locs_base", dist.MultivariateNormal(m_locs_base, scale_tril=st_locs_base))
        shift = pyro.sample("shift", dist.MultivariateNormal(m_shift, scale_tril=st_shift))
    return {'locs_base': locs_base, 'shift': shift}


# %% [4] Architecture: Train, Evaluate, and Plot
def run_model_pipeline(model_name, data):
    pyro.clear_param_store()
    
    models = {"M1": (model_m1, guide_m1), "M3": (model_m3, guide_m3), 
              "M5": (model_m5, guide_m5), "M_LOG": (model_m_log, guide_m_log), 
              "M_FRAC": (model_m_frac, guide_m_frac)}
              
    active_model, active_guide = models[model_name]
    sample_file = f"{model_name}_{SELECTED_DATASET}_samples.pt"
    
    # ---------------------------------------------------------
    #TRAINING OR LOADING
    # ---------------------------------------------------------
    if TRAIN_MODELS:
        print(f"\n{'='*40}\nTRAINING SVI FOR {model_name}\n{'='*40}")
        n_steps = 2 if ('CI' in os.environ) else 5000
        optimizer = Adam({"lr": 0.01})
        svi = SVI(active_model, active_guide, optimizer, loss=Trace_ELBO())

        loss = []
        pbar = tqdm(range(n_steps), position=0)
        for step in pbar:
            loss.append(torch.tensor(svi.step(data)))
            pbar.set_description(f"{model_name} Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
            if torch.isnan(loss[-1]):
                print("Encountered NaN loss. Breaking loop.")
                break

        plt.figure(figsize=(10, 4))
        plt.plot(loss)
        plt.title(f"{model_name} ELBO minimization (Within-Subjects)")
        plt.show()

        print("Extracting 1000 Posteriors and Saving to Disk...")
        samples = [active_guide(data) for _ in range(1000)]
        torch.save(samples, sample_file)
        
    else:
        print(f"\n{'='*40}\nLOADING SAVED MODEL FOR {model_name}\n{'='*40}")
        try:
            samples = torch.load(sample_file)
            print("Successfully loaded saved posterior samples.")
        except FileNotFoundError:
            print(f"Error: {sample_file} not found. Please set TRAIN_MODELS = True first.")
            return torch.tensor([0.])

    # ---------------------------------------------------------
    # WAIC CALCULATION
    # ---------------------------------------------------------
    num_agents, num_trials = data.shape[0], data.shape[2]
    
    if model_name in ["M1", "M5"]: num_params = 2
    elif model_name in ["M_LOG", "M_FRAC"]: num_params = 3
    elif model_name == "M3": num_params = 4

    locs = []
    for s in samples:
        lb = s['locs_base'].detach()
        sh = s['shift'].detach()
        locs_flat = torch.stack([lb, lb + sh], dim=1).reshape(num_agents * 2, num_params).cpu()
        locs.append(locs_flat)
        
    data_flat = data.reshape(num_agents * 2, num_trials, data.shape[-1]).cpu()
    probs = []
    
    for i in locs:
        if model_name == "M1":
            a_param = torch.exp(i[:,0]).unsqueeze(-1).expand(-1, num_trials)
            beta = torch.sigmoid(i[:,1]).unsqueeze(-1).expand(-1, num_trials) * BETA_BOUND
            sigma_combine = a_param * torch.sqrt(data_flat[:,:,2])
            e_mean = (data_flat[:,:,3]) / (1 + sigma_combine**2) + 1e-8
            
        elif model_name == "M3":
            sigma_rate = torch.exp(i[:,0]).unsqueeze(-1).expand(-1, num_trials) #sigma_max
            a_param = torch.exp(i[:,1]).unsqueeze(-1).expand(-1, num_trials)
            b_param = torch.exp(i[:,2]).unsqueeze(-1).expand(-1, num_trials)
            beta = torch.sigmoid(i[:,3]).unsqueeze(-1).expand(-1, num_trials) * BETA_BOUND  
            sigma_combine = sigma_rate / (1 + b_param * torch.exp(-a_param * data_flat[:,:,2]))
            e_mean = (data_flat[:,:,3]) / (1 + sigma_combine**2) + 1e-8
            
        elif model_name == "M5":
            k_param = torch.exp(i[:,0]).unsqueeze(-1).expand(-1, num_trials)
            beta = torch.sigmoid(i[:,1]).unsqueeze(-1).expand(-1, num_trials) * BETA_BOUND
            e_mean = (data_flat[:,:,3]) / (1 + k_param * data_flat[:,:,2]) + 1e-8
            
        elif model_name == "M_LOG":
            a_param = torch.exp(i[:,0]).unsqueeze(-1).expand(-1, num_trials)
            b_param = torch.exp(i[:,1]).unsqueeze(-1).expand(-1, num_trials)
            beta = torch.sigmoid(i[:,2]).unsqueeze(-1).expand(-1, num_trials) * BETA_BOUND
            sigma_combine = a_param * torch.log(1 + b_param * data_flat[:,:,2])
            e_mean = (data_flat[:,:,3]) / (1 + sigma_combine**2) + 1e-8
            
        elif model_name == "M_FRAC":
            a_param = torch.exp(i[:,0]).unsqueeze(-1).expand(-1, num_trials)
            s_param = torch.sigmoid(i[:,1]).unsqueeze(-1).expand(-1, num_trials) * 2.0 
            beta = torch.sigmoid(i[:,2]).unsqueeze(-1).expand(-1, num_trials) * BETA_BOUND
            sigma_combine = a_param * torch.pow(data_flat[:,:,2] + 1e-8, s_param)
            e_mean = (data_flat[:,:,3]) / (1 + sigma_combine**2) + 1e-8

        log1 = torch.log(e_mean) * beta
        log2 = torch.log(data_flat[:,:,5]) * beta
        logits = torch.stack([log1, log2], dim=0)
        probs.append(torch.softmax(logits, dim=0)[0])

    likeli = []
    for i in probs:
        temp = i * data_flat[:,:,4] + (1 - data_flat[:,:,4]) * (1 - i)
        likeli.append(torch.clamp(temp, min=1e-8, max=1.0))

    final = torch.stack(likeli).permute(1, 2, 0)
    log_likes = torch.log(final)
    lppd = torch.logsumexp(log_likes, dim=-1) - torch.log(torch.tensor(1000.))
    lppd_sum = lppd.sum(dim=1)
    p_waic = log_likes.var(dim=-1, unbiased=True).sum(dim=1)
    subj_waic = -2 * (lppd_sum - p_waic)
    total_waic = subj_waic.sum()
    se_waic = torch.sqrt(subj_waic.shape[0] * subj_waic.var(unbiased=True))

    print(f"--> {model_name} Total WAIC: {total_waic.item():.2f} (SE: {se_waic.item():.2f})")
    
    ##############################
    #POSTERIOR PREDICTIVE CHECKS##
    ##############################
    valid_mask = (data_flat[:, :, 5] > 0).float() 
    valid_trials_per_subj = valid_mask.sum(dim=1)
    mean_probs = torch.stack(probs).mean(dim=0)

    predicted_choices = (mean_probs > 0.5).float()
    actual_choices = data_flat[:, :, 4]

    correct_matches = (predicted_choices == actual_choices).float() * valid_mask
    correct_per_agent = correct_matches.sum(dim=1)
    agent_accuracies = (correct_per_agent / (valid_trials_per_subj + 1e-8)) * 100

    overall_accuracy = (correct_per_agent.sum().item() / valid_trials_per_subj.sum().item()) * 100
    print(f"--> {model_name} Hit Rate: {overall_accuracy:.2f}%")
    
    # ---------------------------------------------------------
    # plot: HYPERPARAMETER POSTERIOR DISTRIBUTIONS
    # ---------------------------------------------------------
    print("Generating Hyperparameter Posterior Distributions...")
    
    # 1. Stack the 1000 posterior samples
    stacked_base = torch.stack([s['locs_base'].detach().cpu() for s in samples]) # Shape: [1000, 30, params]
    stacked_shift = torch.stack([s['shift'].detach().cpu() for s in samples])    # Shape: [1000, 30, params]
    
    # 2. Average across the agents to get the Group-Level Mean for each of the 1000 samples
    group_base_raw = stacked_base.mean(dim=1) 
    group_cond2_raw = (stacked_base + stacked_shift).mean(dim=1)

    # 3. Transform parameters from log/logit space back into NATURAL space
    if model_name == "M1":
        param_names = ["a (Scaling)", "beta (Inv Temp)"]
        group_base_nat = torch.stack([torch.exp(group_base_raw[:,0]), torch.sigmoid(group_base_raw[:,1]) * BETA_BOUND], dim=1)
        group_cond2_nat = torch.stack([torch.exp(group_cond2_raw[:,0]), torch.sigmoid(group_cond2_raw[:,1]) * BETA_BOUND], dim=1)
        
    elif model_name == "M3":
        param_names = ["sigma_rate (Plateau)", "a (Slope)", "b (Intercept)", "beta (Inv Temp)"]
        group_base_nat = torch.stack([torch.exp(group_base_raw[:,0]), torch.exp(group_base_raw[:,1]), torch.exp(group_base_raw[:,2]), torch.sigmoid(group_base_raw[:,3]) * BETA_BOUND], dim=1)
        group_cond2_nat = torch.stack([torch.exp(group_cond2_raw[:,0]), torch.exp(group_cond2_raw[:,1]), torch.exp(group_cond2_raw[:,2]), torch.sigmoid(group_cond2_raw[:,3]) * BETA_BOUND], dim=1)
        
    elif model_name == "M5":
        param_names = ["k (Discount Rate)", "beta (Inv Temp)"]
        group_base_nat = torch.stack([torch.exp(group_base_raw[:,0]), torch.sigmoid(group_base_raw[:,1]) * BETA_BOUND], dim=1)
        group_cond2_nat = torch.stack([torch.exp(group_cond2_raw[:,0]), torch.sigmoid(group_cond2_raw[:,1]) * BETA_BOUND], dim=1)
        
    elif model_name == "M_LOG":
        param_names = ["a (Amplitude)", "b (Compression)", "beta (Inv Temp)"]
        group_base_nat = torch.stack([torch.exp(group_base_raw[:,0]), torch.exp(group_base_raw[:,1]), torch.sigmoid(group_base_raw[:,2]) * BETA_BOUND], dim=1)
        group_cond2_nat = torch.stack([torch.exp(group_cond2_raw[:,0]), torch.exp(group_cond2_raw[:,1]), torch.sigmoid(group_cond2_raw[:,2]) * BETA_BOUND], dim=1)
        
    elif model_name == "M_FRAC":
        param_names = ["a (Amplitude)", "s (Exponent)", "beta (Inv Temp)"]
        group_base_nat = torch.stack([torch.exp(group_base_raw[:,0]), torch.sigmoid(group_base_raw[:,1]) * 2.0, torch.sigmoid(group_base_raw[:,2]) * BETA_BOUND], dim=1)
        group_cond2_nat = torch.stack([torch.exp(group_cond2_raw[:,0]), torch.sigmoid(group_cond2_raw[:,1]) * 2.0, torch.sigmoid(group_cond2_raw[:,2]) * BETA_BOUND], dim=1)

    #Calculate the Absolute Shift in Natural Space
    group_shift_nat = group_cond2_nat - group_base_nat

    #enerate Density Plots (KDE) for the Hyperparameters
    sns.set_theme(style="whitegrid")
    #squeeze=False ensures 'axes' is always a 2D array, even if there is only 1 parameter row
    fig, axes = plt.subplots(len(param_names), 2, figsize=(10, 3.5 * len(param_names)), squeeze=False)
    
    for p_idx, p_name in enumerate(param_names):
        
        #Left Column: Overlay Baseline vs Condition 2
        sns.kdeplot(group_base_nat[:, p_idx].numpy(), ax=axes[p_idx, 0], fill=True, label="Cond 1 (Base)", color="#4c72b0")
        sns.kdeplot(group_cond2_nat[:, p_idx].numpy(), ax=axes[p_idx, 0], fill=True, label="Cond 2", color="#dd8452")
        axes[p_idx, 0].set_title(f"Group Mean: {p_name}", fontweight='bold')
        axes[p_idx, 0].legend()
        axes[p_idx, 0].set_ylabel("Density")
        
        #Right Column: The True Shift Distribution (Condition 2 - Baseline)
        shift_data = group_shift_nat[:, p_idx].numpy()
        sns.kdeplot(shift_data, ax=axes[p_idx, 1], fill=True, color="#55a868")
        
        #Extract 95% Credible Intervals to print directly on the plot
        mean_shift = shift_data.mean()
        ci_lower = np.percentile(shift_data, 2.5)
        ci_upper = np.percentile(shift_data, 97.5)
        
        axes[p_idx, 1].axvline(0, color='red', linestyle='--', alpha=0.7) # Mark zero shift for significance checking
        axes[p_idx, 1].axvline(ci_lower, color='black', linestyle=':', alpha=0.5)
        axes[p_idx, 1].axvline(ci_upper, color='black', linestyle=':', alpha=0.5)
        
        axes[p_idx, 1].set_title(f"Group Shift ($\Delta$): {p_name}\nMean: {mean_shift:.3f} [95% CI: {ci_lower:.3f}, {ci_upper:.3f}]", fontsize=11)
        axes[p_idx, 1].set_ylabel("")
        
    plt.suptitle(f"{model_name} Hyperparameter Posteriors ({SELECTED_DATASET})", fontsize=16, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    
    return subj_waic


# %%Execution Loop
if SELECTED_MODEL == "ALL":
    models_to_run = ["M1", "M3", "M5", "M_LOG", "M_FRAC"]
else:
    models_to_run = [SELECTED_MODEL]
    
results = {}

for m in models_to_run:
    results[m] = run_model_pipeline(m, data)

if len(results) > 1:
    print(f"\n{'='*40}\nFINAL MODEL RANKING\n{'='*40}")
    totals = {name: waic.sum().item() for name, waic in results.items()}
    ranked = sorted(totals.items(), key=lambda item: item[1])
    best_model_name = ranked[0][0]
    best_waic_tensor = results[best_model_name]
    
    print(f"1. {best_model_name} (Best Fit) - WAIC: {ranked[0][1]:.2f}")
    for i in range(1, len(ranked)):
        comp_name = ranked[i][0]
        comp_waic_tensor = results[comp_name]
        delta_waic = comp_waic_tensor - best_waic_tensor
        total_delta = delta_waic.sum().item()
        se_delta = torch.sqrt(delta_waic.shape[0] * delta_waic.var(unbiased=True)).item()
        print(f"{i+1}. {comp_name} - WAIC: {ranked[i][1]:.2f} | Δ vs Best: +{total_delta:.2f} (SE: {se_delta:.2f})")

    print(f"\n{'='*40}\nGENERATING WAIC COMPARISON PLOT\n{'='*40}")
    model_names, delta_waics, se_deltas_2x = [], [], []

    for name, _ in ranked:
        comp_waic_tensor = results[name]
        delta_tensor = comp_waic_tensor - best_waic_tensor
        d_waic = delta_tensor.sum().item()
        se_d = torch.sqrt(delta_tensor.shape[0] * delta_tensor.var(unbiased=True)).item()
        
        model_names.append(name)
        delta_waics.append(d_waic)
        se_deltas_2x.append(se_d * 2)

    plt.figure(figsize=(8, 4))
    sns.set_theme(style="whitegrid") 
    plt.errorbar(delta_waics, model_names, xerr=se_deltas_2x, fmt='o', color='black', ecolor='gray', elinewidth=2, capsize=5, markersize=8)
    plt.axvline(0, color='red', linestyle='--', alpha=0.7, label=f'Baseline ({best_model_name})')
    plt.xlabel(r'$\Delta$ WAIC (Relative to Best Model)')
    plt.title(f'Model Comparison ({SELECTED_DATASET} Dataset - Within-Subjects)\nError bars represent $2 \times SE_{{\Delta}}$')
    plt.gca().invert_yaxis() 
    plt.legend()
    plt.tight_layout()
    plt.show()