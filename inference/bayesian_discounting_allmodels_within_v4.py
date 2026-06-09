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

#
#M5 standard Hyperbolic (2 Params k + beta) - CLASSICAL SOFTMAX
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
        
        # Classical Linear Softmax
        v_ll = (data_flat[:,:,3]) / (1 + k_param * data_flat[:,:,2])
        v_ss = data_flat[:,:,5]
        
        logit_ll = v_ll * beta
        logit_ss = v_ss * beta
        logits = torch.stack([logit_ll, logit_ss], dim=0)
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
            
            # Classical Linear Softmax
            v_ll = (data_flat[:,:,3]) / (1 + k_param * data_flat[:,:,2])
            v_ss = data_flat[:,:,5]
            
            logit_ll = v_ll * beta
            logit_ss = v_ss * beta
            logits = torch.stack([logit_ll, logit_ss], dim=0)
            probs.append(torch.softmax(logits, dim=0)[0])
            continue # Skip the log1/log2 lines below for M5!
            
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
    
    #return both samples and waic
    return subj_waic, samples



# %%Execution Loop
if SELECTED_MODEL == "ALL":
    models_to_run = ["M1", "M3", "M5", "M_LOG", "M_FRAC"]
else:
    models_to_run = [SELECTED_MODEL]
    
results = {}
all_samples = {} # <--- New dictionary to hold samples

for m in models_to_run:
    # Catch both outputs
    waic, samps = run_model_pipeline(m, data)
    results[m] = waic
    all_samples[m] = samps  # <--- Store samples in RAM

#%%
#WAIC PLOT
# %% [5] FINAL MODEL RANKING & WAIC COMPARISON PLOT
import torch
import matplotlib.pyplot as plt

if len(results) > 1:
    print(f"\n{'='*40}\nFINAL MODEL RANKING\n{'='*40}")
    
    # 1. Calculate and sort the WAIC totals
    totals = {name: waic.sum().item() for name, waic in results.items()}
    ranked = sorted(totals.items(), key=lambda item: item[1])
    best_model_name = ranked[0][0]
    best_waic_tensor = results[best_model_name]
    
    # 2. Print the text rankings to the console
    print(f"1. {best_model_name} (Best Fit) - WAIC: {ranked[0][1]:.2f}")
    for i in range(1, len(ranked)):
        comp_name = ranked[i][0]
        comp_waic_tensor = results[comp_name]
        delta_waic = comp_waic_tensor - best_waic_tensor
        total_delta = delta_waic.sum().item()
        se_delta = torch.sqrt(delta_waic.shape[0] * delta_waic.var(unbiased=True)).item()
        print(f"{i+1}. {comp_name} - WAIC: {ranked[i][1]:.2f} | Δ vs Best: +{total_delta:.2f} (SE: {se_delta:.2f})")

    # 3. Prepare data for the plot
    model_names, delta_waics, se_deltas_2x = [], [], []
    for name, _ in ranked:
        comp_waic_tensor = results[name]
        delta_tensor = comp_waic_tensor - best_waic_tensor
        d_waic = delta_tensor.sum().item()
        se_d = torch.sqrt(delta_tensor.shape[0] * delta_tensor.var(unbiased=True)).item()
        
        model_names.append(name)
        delta_waics.append(d_waic)
        se_deltas_2x.append(se_d * 2)

    # ==========================================
    # APPLY 'bentheme' GLOBALLY
    # ==========================================
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 16,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'figure.facecolor': 'none', 
        'axes.facecolor': 'none',   
        'savefig.facecolor': 'none',
        'savefig.edgecolor': 'none',
        'legend.fontsize': 16,
        'axes.grid': False
    })

    print(f"\n{'='*40}\nGENERATING WAIC COMPARISON PLOT\n{'='*40}")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot with thicker lines and larger markers
    ax.errorbar(delta_waics, model_names, xerr=se_deltas_2x, fmt='o', color='black', ecolor='black', elinewidth=2.5, capsize=6, capthick=2.5, markersize=10)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Baseline ({best_model_name})')
    
    ax.set_xlabel(r'$\Delta$ WAIC (Relative to Best Model)')
    ax.set_title(f'Model Comparison ({SELECTED_DATASET} Dataset)\n ')
    ax.invert_yaxis() 
    
    # Apply structural bentheme (thick spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(width=1.5, length=7, color='black')
    
    ax.legend(loc='upper right', frameon=False)
    
    plt.tight_layout()
    plt.show()

else:
    print("\n[NOTE] You need to run 'ALL' models to generate a WAIC comparison plot!")
    
#%%
# %%Hyperparameter Posteriors (Independent Cell and rtheme style
# %% [7] Plot Hyperparameter Posteriors (Independent Cell and rtheme style)
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BETA_BOUND = 10.0  

def calc_hdi(samples, credible_mass=0.95):
    sorted_samples = np.sort(samples)
    interval_idx_inc = int(np.floor(credible_mass * len(sorted_samples)))
    n_intervals = len(sorted_samples) - interval_idx_inc
    interval_width = sorted_samples[interval_idx_inc:] - sorted_samples[:n_intervals]
    min_idx = np.argmin(interval_width)
    return sorted_samples[min_idx], sorted_samples[min_idx + interval_idx_inc]

def transform_to_nat_base_only(raw_tensor, model_name):
    # Transforms log/logit space back into NATURAL space for plotting
    if model_name == "M1":
        param_names = ["a (Scaling)", r"$\beta$"]
        nat_tensor = torch.stack([torch.exp(raw_tensor[:,0]), torch.sigmoid(raw_tensor[:,1]) * BETA_BOUND], dim=1)
        
    elif model_name == "M3":
        param_names = ["max sigma_rate", "a (Slope)", "b (Intercept)", r"$\beta$"]
        nat_tensor = torch.stack([torch.exp(raw_tensor[:,0]), torch.exp(raw_tensor[:,1]), torch.exp(raw_tensor[:,2]), torch.sigmoid(raw_tensor[:,3]) * BETA_BOUND], dim=1)
        
    elif model_name == "M5":
        param_names = ["k (Discount Rate)", r"Softmax $\beta$"]
        nat_tensor = torch.stack([torch.exp(raw_tensor[:,0]), torch.sigmoid(raw_tensor[:,1]) * BETA_BOUND], dim=1)
        
    elif model_name == "M_LOG":
        param_names = ["a (Amplitude)", "b (Compression)", r"$\beta$"]
        nat_tensor = torch.stack([torch.exp(raw_tensor[:,0]), torch.exp(raw_tensor[:,1]), torch.sigmoid(raw_tensor[:,2]) * BETA_BOUND], dim=1)
        
    elif model_name == "M_FRAC":
        param_names = ["a (Amplitude)", "s (Exponent)", r"$\beta$"]
        nat_tensor = torch.stack([torch.exp(raw_tensor[:,0]), torch.sigmoid(raw_tensor[:,1]) * 2.0, torch.sigmoid(raw_tensor[:,2]) * BETA_BOUND], dim=1)
        
    return param_names, nat_tensor

def plot_hyperparameters(model_name, samples, dataset_name):
    print(f"Generating plot for {model_name}...")
    
    if dataset_name == "CAFE":
        label_base = "Café Env"
        label_cond2 = "Gambling Env"
    elif dataset_name == "FUTURE":
        label_base = "Without Future Cues"
        label_cond2 = "With Future Cues"
    else:
        label_base = "Cond 1 (Base)"
        label_cond2 = "Cond 2"
    
    # 1. Stack the posterior samples
    stacked_base = torch.stack([s['locs_base'].detach().cpu() for s in samples])
    stacked_shift = torch.stack([s['shift'].detach().cpu() for s in samples])
    
    # 2. Average across the agents to get the group means (Latent Space)
    group_base_raw = stacked_base.mean(dim=1) 
    group_cond2_raw = (stacked_base + stacked_shift).mean(dim=1)
    
    # This IS the pure magnitude effect (Latent Space)
    shift_raw = stacked_shift.mean(dim=1)

    # 3. Transform ONLY the Base and Cond2 into natural space for the Left Column overlay
    param_names, group_base_nat = transform_to_nat_base_only(group_base_raw, model_name)
    _, group_cond2_nat = transform_to_nat_base_only(group_cond2_raw, model_name)

    plt.rcParams.update({
        'font.size': 16, 'axes.titlesize': 16, 'axes.titleweight': 'bold',
        'axes.labelsize': 16, 'text.color': 'black', 'axes.labelcolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black', 'figure.facecolor': 'none', 
        'axes.facecolor': 'none', 'savefig.facecolor': 'none',
        'savefig.edgecolor': 'none', 'legend.fontsize': 16, 'axes.grid': False
    })
    
    def apply_bentheme(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.tick_params(width=1.5, length=7, color='black')

    fig, axes = plt.subplots(len(param_names), 2, figsize=(12, 4.5 * len(param_names)), squeeze=False)
    
    for p_idx, p_name in enumerate(param_names):
        # Left Column: Overlay Natural Space Distributions
        sns.kdeplot(group_base_nat[:, p_idx].numpy(), ax=axes[p_idx, 0], fill=True, label=label_base, color="#4c72b0", linewidth=2.5, alpha=0.5)
        sns.kdeplot(group_cond2_nat[:, p_idx].numpy(), ax=axes[p_idx, 0], fill=True, label=label_cond2, color="#dd8452", linewidth=2.5, alpha=0.5)
        
        axes[p_idx, 0].set_title(f"Group Mean: {p_name}")
        axes[p_idx, 0].set_ylabel("Density")
        
        # Right Column: The Raw Latent Shift
        shift_data = shift_raw[:, p_idx].numpy()
        sns.kdeplot(shift_data, ax=axes[p_idx, 1], fill=True, color="#55a868", linewidth=2.5, alpha=0.5)
        
        mean_shift = shift_data.mean()
        ci_lower, ci_upper = calc_hdi(shift_data, 0.95)
            
        axes[p_idx, 1].axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8) 
        axes[p_idx, 1].axvline(ci_lower, color='black', linestyle=':', linewidth=2, alpha=0.8)
        axes[p_idx, 1].axvline(ci_upper, color='black', linestyle=':', linewidth=2, alpha=0.8)
        
        axes[p_idx, 1].set_title(f"Group Shift (Latent $\Delta$): {p_name}\nMean: {mean_shift:.3f} [95% HDI: {ci_lower:.3f}, {ci_upper:.3f}]")
        axes[p_idx, 1].set_ylabel("")
        
        # ONLY add legend to the very last row (Softmax beta) to prevent repetition
        if p_idx == len(param_names) - 1:
            axes[p_idx, 0].legend(loc='lower center', bbox_to_anchor=(0.5, -0.6), ncol=2, frameon=False)
            
        apply_bentheme(axes[p_idx, 0])
        apply_bentheme(axes[p_idx, 1])
        
    plt.suptitle(f"{model_name} Hyperparameter Posteriors ({dataset_name})", fontsize=20, y=1.05, fontweight='bold')
    plt.tight_layout()
    plt.show()

# --- EXECUTE PLOTTING ---
for m in models_to_run:
    if all_samples[m] is not None:
        plot_hyperparameters(m, all_samples[m], SELECTED_DATASET)