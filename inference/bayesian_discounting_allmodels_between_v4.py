#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 2026
@author: benwagner

MIXED-DESIGN PIPELINE (Haloperidol CSV)
Models: M1, M3, M5, M_LOG, M_FRAC
Features: 2x2 Mixed Design (Drug x Magnitude), Conditional Group-Level Shrinkage, Masked WAIC.
"""

# %% [1] Imports and Master Setup
import os
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

# ==========================================
# MASTER CONTROL PANEL
# ==========================================
# Options: "M1", "M3", "M5", "M_LOG", "M_FRAC", or "ALL"
SELECTED_MODEL = "ALL"        

# True = Run SVI and save new samples to disk.
# False = Skip SVI, load saved samples, and just run WAIC/PPC/Plots.
TRAIN_MODELS = True
# ==========================================

BETA_BOUND = 20.0  

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")


# %% [2] Data Loading & Preprocessing (Mixed-Design)
print("Loading Haloperidol CSV for Mixed Design...")
df = pd.read_csv('Haloperidol_Discounting_Dataset_Softmax.csv')

# FIX 1: Sort by DRUG first, then loopID, then Trial
df = df.sort_values(by=['Drug', 'loopID', 'Trial'])

subjects = df['loopID'].unique()
num_subjects = len(subjects)

# Find the maximum possible trials any subject has in a single condition
max_trials = df.groupby(['loopID', 'Condition']).size().max()

# Data Shape: [Subjects, Within-Conds (Base/Shift), Trials, Features]
feature_matrix = np.zeros((num_subjects, 2, max_trials, 6))
drug_ids = torch.zeros(num_subjects, dtype=torch.long).to(device)

unique_drugs = df['Drug'].unique()
# Assuming Drug 1 = Placebo (0), Drug 2 = Haloperidol (1)
drug_map = {unique_drugs[0]: 0, unique_drugs[1]: 1} 
print(f"Drug Mapping Applied: {drug_map}")

for i, subj in enumerate(subjects):
    subj_data = df[df['loopID'] == subj]
    
    # ----------------------------------------------------
    # Baseline (SS=20) is Condition 2 -> Stored at index 0
    # ----------------------------------------------------
    base_data = subj_data[subj_data['Condition'] == 2]
    actual_base = len(base_data)
    if actual_base > 0:
        feature_matrix[i, 0, :actual_base, 2] = base_data['Delay'].values
        feature_matrix[i, 0, :actual_base, 3] = base_data['Value_LL'].values
        feature_matrix[i, 0, :actual_base, 4] = base_data['Choice'].values
        feature_matrix[i, 0, :actual_base, 5] = base_data['Value_SS'].values
    
    # ----------------------------------------------------
    # Shift (SS=100) is Condition 1 -> Stored at index 1
    # ----------------------------------------------------
    shift_data = subj_data[subj_data['Condition'] == 1]
    actual_shift = len(shift_data)
    if actual_shift > 0:
        feature_matrix[i, 1, :actual_shift, 2] = shift_data['Delay'].values
        feature_matrix[i, 1, :actual_shift, 3] = shift_data['Value_LL'].values
        feature_matrix[i, 1, :actual_shift, 4] = shift_data['Choice'].values
        feature_matrix[i, 1, :actual_shift, 5] = shift_data['Value_SS'].values
    
    # Route the Between-Subjects variable
    drug_val = subj_data['Drug'].iloc[0]
    drug_ids[i] = drug_map[drug_val]
    
data = torch.tensor(feature_matrix, dtype=torch.float32).to(device)
print(f"Data shape successfully set to: {data.shape}")
print(f"Group Split Check: {(drug_ids == 0).sum().item()} Placebo agents vs {(drug_ids == 1).sum().item()} Haloperidol agents")


# %% [3] Model Definitions (Mixed Architecture)

# ----- M1: Square Root (2 Params) -----
def model_m1(data, drug_ids):
    num_params = 2
    num_agents, num_trials = data.shape[0], data.shape[2] 
    
    # Hierarchical priors depend on the Drug group (Size 2)
    m_base = pyro.param('m_base', torch.zeros(2, num_params, device=device))
    s_base = pyro.param('s_base', torch.ones(2, num_params, device=device), constraint=dist.constraints.positive)
    m_shift = pyro.param('m_shift', torch.zeros(2, num_params, device=device))
    s_shift = pyro.param('s_shift', torch.ones(2, num_params, device=device), constraint=dist.constraints.positive)
    
    with pyro.plate('ag_idx', num_agents):
        # Route agent to the correct Drug group prior
        ab_m, ab_s = m_base[drug_ids], s_base[drug_ids]
        as_m, as_s = m_shift[drug_ids], s_shift[drug_ids]
        
        base_dist = dist.Normal(torch.zeros(num_params, device=device), torch.ones(num_params, device=device)).to_event(1)
        locs_base = pyro.sample('locs_base', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(ab_m, ab_s)]))
        shift = pyro.sample('shift', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(as_m, as_s)]))

    locs_flat = torch.stack([locs_base, locs_base + shift], dim=1).reshape(num_agents * 2, num_params)
    data_flat = data.reshape(num_agents * 2, num_trials, data.shape[-1])

    with pyro.plate('data', num_agents * 2 * num_trials):
        a_param = torch.exp(locs_flat[:,0]).unsqueeze(-1).expand(-1, num_trials)
        beta = torch.sigmoid(locs_flat[:,1]).unsqueeze(-1).expand(-1, num_trials)*BETA_BOUND
        
        sigma_combine = a_param * torch.sqrt(data_flat[:,:,2])
        e_mean = (data_flat[:,:,3])/(1 + sigma_combine**2) + 1e-8
        
        log1 = torch.log(e_mean) * beta
        log2 = torch.log(data_flat[:,:,5] + 1e-8) * beta # Bulletproof Log
        logits = torch.stack([log1, log2], dim=0)
        p = torch.softmax(logits, dim=0)[0]
        p = torch.clamp(p, min=1e-5, max=1.0-1e-5)
        
    valid_mask = (data_flat[:, :, 5] > 0)
    masked_dist = dist.Bernoulli(probs=p).mask(valid_mask).to_event(2)
    pyro.sample("obs", masked_dist, obs=data_flat[:,:,4])

def guide_m1(data, drug_ids):
    num_params, num_agents = 2, data.shape[0]
    m_locs_base = pyro.param('m_locs_base', torch.zeros(num_agents, num_params, device=data.device))
    st_locs_base = pyro.param('st_locs_base', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    m_shift = pyro.param('m_shift_locs', torch.zeros(num_agents, num_params, device=data.device))
    st_shift = pyro.param('st_shift_locs', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    
    with pyro.plate('ag_idx', num_agents):
        locs_base = pyro.sample("locs_base", dist.MultivariateNormal(m_locs_base, scale_tril=st_locs_base))
        shift = pyro.sample("shift", dist.MultivariateNormal(m_shift, scale_tril=st_shift))
    return {'locs_base': locs_base, 'shift': shift}

# ----- M3: Sigmoid (4 Params) -----
def model_m3(data, drug_ids):
    num_params = 4
    num_agents, num_trials = data.shape[0], data.shape[2]
    
    a_b = pyro.param('a_b', torch.ones(2, num_params, device=device), constraint=dist.constraints.positive)
    lam_b = pyro.param('lam_b', torch.ones(2, num_params, device=device), constraint=dist.constraints.positive)
    tau_b = pyro.sample('tau_b', dist.Gamma(a_b, a_b/lam_b).to_event(2)) 
    sig_b = pyro.deterministic('sig_b', 1/torch.sqrt(tau_b)) 
    
    m_b = pyro.param('m_b', torch.zeros(2, num_params, device=device))
    s_b = pyro.param('s_b', torch.ones(2, num_params, device=device), constraint=dist.constraints.positive)
    mu_b = pyro.sample('mu_b', dist.Normal(m_b, s_b*sig_b).to_event(2)) 
    
    m_s = pyro.param('m_s', torch.zeros(2, num_params, device=device))
    s_s = pyro.param('s_s', torch.ones(2, num_params, device=device), constraint=dist.constraints.positive)
    
    with pyro.plate('ag_idx', num_agents):
        ab_mu, ab_sig = mu_b[drug_ids], sig_b[drug_ids]
        as_m, as_s = m_s[drug_ids], s_s[drug_ids]
        
        base_dist = dist.Normal(torch.zeros(num_params, device=device), torch.ones(num_params, device=device)).to_event(1)
        locs_base = pyro.sample('locs_base', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(ab_mu, ab_sig)]))
        shift = pyro.sample('shift', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(as_m, as_s)]))

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
        log2 = torch.log(data_flat[:,:,5] + 1e-8) * beta
        logits = torch.stack([log1, log2], dim=0)
        p = torch.softmax(logits, dim=0)[0]
        p = torch.clamp(p, min=1e-5, max=1.0-1e-5)
        
    valid_mask = (data_flat[:, :, 5] > 0)
    masked_dist = dist.Bernoulli(probs=p).mask(valid_mask).to_event(2)
    pyro.sample("obs", masked_dist, obs=data_flat[:,:,4])

def guide_m3(data, drug_ids):
    num_params, num_agents = 4, data.shape[0]
    trns = torch.distributions.biject_to(dist.constraints.positive)
    
    m_hyp_base = pyro.param('m_hyp_base', torch.zeros(2, 2*num_params, device=data.device))
    st_hyp_base = pyro.param('scale_tril_hyp_base', torch.eye(2*num_params, device=data.device).repeat(2, 1, 1), constraint=dist.constraints.lower_cholesky)
    hyp_base = pyro.sample('hyp_base', dist.MultivariateNormal(m_hyp_base, scale_tril=st_hyp_base).to_event(1), infer={'is_auxiliary': True})
    
    unc_mu, unc_tau = hyp_base[..., :num_params], hyp_base[..., num_params:]
    c_tau = trns(unc_tau)
    ld_tau = -trns.inv.log_abs_det_jacobian(c_tau, unc_tau).sum()
    
    mu_b = pyro.sample("mu_b", dist.Delta(unc_mu, event_dim=2))
    tau_b = pyro.sample("tau_b", dist.Delta(c_tau, log_density=ld_tau, event_dim=2))

    m_locs_base = pyro.param('m_locs_base', torch.zeros(num_agents, num_params, device=data.device))
    st_locs_base = pyro.param('st_locs_base', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    m_shift = pyro.param('m_shift_locs', torch.zeros(num_agents, num_params, device=data.device))
    st_shift = pyro.param('st_shift_locs', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    
    with pyro.plate('ag_idx', num_agents):
        locs_base = pyro.sample("locs_base", dist.MultivariateNormal(m_locs_base, scale_tril=st_locs_base))
        shift = pyro.sample("shift", dist.MultivariateNormal(m_shift, scale_tril=st_shift))
        
    return {'tau_b': tau_b, 'mu_b': mu_b, 'locs_base': locs_base, 'shift': shift}

# ----- M5: Hyperbolic (2 Params) -----
def model_m5(data, drug_ids):
    num_params = 2
    num_agents, num_trials = data.shape[0], data.shape[2] 
    
    m_base = pyro.param('m_base', torch.zeros(2, num_params, device=device))
    s_base = pyro.param('s_base', torch.ones(2, num_params, device=device), constraint=dist.constraints.positive)
    m_shift = pyro.param('m_shift', torch.zeros(2, num_params, device=device))
    s_shift = pyro.param('s_shift', torch.ones(2, num_params, device=device), constraint=dist.constraints.positive)
    
    with pyro.plate('ag_idx', num_agents):
        ab_m, ab_s = m_base[drug_ids], s_base[drug_ids]
        as_m, as_s = m_shift[drug_ids], s_shift[drug_ids]
        
        base_dist = dist.Normal(torch.zeros(num_params, device=device), torch.ones(num_params, device=device)).to_event(1)
        locs_base = pyro.sample('locs_base', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(ab_m, ab_s)]))
        shift = pyro.sample('shift', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(as_m, as_s)]))

    locs_flat = torch.stack([locs_base, locs_base + shift], dim=1).reshape(num_agents * 2, num_params)
    data_flat = data.reshape(num_agents * 2, num_trials, data.shape[-1])

    with pyro.plate('data', num_agents * 2 * num_trials):
        k_param = torch.exp(locs_flat[:,0]).unsqueeze(-1).expand(-1, num_trials)
        beta = torch.sigmoid(locs_flat[:,1]).unsqueeze(-1).expand(-1, num_trials)*BETA_BOUND
        
        # 1. Calculate raw subjective values
        v_ll = (data_flat[:,:,3]) / (1 + k_param * data_flat[:,:,2])
        v_ss = data_flat[:,:,5]
        
        # 2. Apply Classical Linear Softmax
        logit_ll = v_ll * beta
        logit_ss = v_ss * beta
        
        logits = torch.stack([logit_ll, logit_ss], dim=0)
        p = torch.softmax(logits, dim=0)[0]
        p = torch.clamp(p, min=1e-5, max=1.0-1e-5)
        
    valid_mask = (data_flat[:, :, 5] > 0)
    masked_dist = dist.Bernoulli(probs=p).mask(valid_mask).to_event(2)
    pyro.sample("obs", masked_dist, obs=data_flat[:,:,4])

def guide_m5(data, drug_ids):
    num_params, num_agents = 2, data.shape[0]
    m_locs_base = pyro.param('m_locs_base', torch.zeros(num_agents, num_params, device=data.device))
    st_locs_base = pyro.param('st_locs_base', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    m_shift = pyro.param('m_shift_locs', torch.zeros(num_agents, num_params, device=data.device))
    st_shift = pyro.param('st_shift_locs', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    
    with pyro.plate('ag_idx', num_agents):
        locs_base = pyro.sample("locs_base", dist.MultivariateNormal(m_locs_base, scale_tril=st_locs_base))
        shift = pyro.sample("shift", dist.MultivariateNormal(m_shift, scale_tril=st_shift))
    return {'locs_base': locs_base, 'shift': shift}

# ----- M_LOG: Logarithmic (3 Params) -----
def model_m_log(data, drug_ids):
    num_params = 3
    num_agents, num_trials = data.shape[0], data.shape[2] 
    
    m_base = pyro.param('m_base', torch.zeros(2, num_params, device=device))
    s_base = pyro.param('s_base', torch.ones(2, num_params, device=device), constraint=dist.constraints.positive)
    m_shift = pyro.param('m_shift', torch.zeros(2, num_params, device=device))
    s_shift = pyro.param('s_shift', torch.ones(2, num_params, device=device), constraint=dist.constraints.positive)
    
    with pyro.plate('ag_idx', num_agents):
        ab_m, ab_s = m_base[drug_ids], s_base[drug_ids]
        as_m, as_s = m_shift[drug_ids], s_shift[drug_ids]
        
        base_dist = dist.Normal(torch.zeros(num_params, device=device), torch.ones(num_params, device=device)).to_event(1)
        locs_base = pyro.sample('locs_base', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(ab_m, ab_s)]))
        shift = pyro.sample('shift', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(as_m, as_s)]))

    locs_flat = torch.stack([locs_base, locs_base + shift], dim=1).reshape(num_agents * 2, num_params)
    data_flat = data.reshape(num_agents * 2, num_trials, data.shape[-1])

    with pyro.plate('data', num_agents * 2 * num_trials):
        a_param = torch.exp(locs_flat[:,0]).unsqueeze(-1).expand(-1, num_trials)
        b_param = torch.exp(locs_flat[:,1]).unsqueeze(-1).expand(-1, num_trials)
        beta = torch.sigmoid(locs_flat[:,2]).unsqueeze(-1).expand(-1, num_trials)*BETA_BOUND
        
        sigma_combine = a_param * torch.log(1 + b_param * data_flat[:,:,2])
        e_mean = (data_flat[:,:,3]) / (1 + sigma_combine**2) + 1e-8
        
        log1 = torch.log(e_mean) * beta
        log2 = torch.log(data_flat[:,:,5] + 1e-8) * beta
        logits = torch.stack([log1, log2], dim=0)
        p = torch.softmax(logits, dim=0)[0]
        p = torch.clamp(p, min=1e-5, max=1.0-1e-5)
        
    valid_mask = (data_flat[:, :, 5] > 0)
    masked_dist = dist.Bernoulli(probs=p).mask(valid_mask).to_event(2)
    pyro.sample("obs", masked_dist, obs=data_flat[:,:,4])

def guide_m_log(data, drug_ids):
    num_params, num_agents = 3, data.shape[0]
    m_locs_base = pyro.param('m_locs_base', torch.zeros(num_agents, num_params, device=data.device))
    st_locs_base = pyro.param('st_locs_base', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    m_shift = pyro.param('m_shift_locs', torch.zeros(num_agents, num_params, device=data.device))
    st_shift = pyro.param('st_shift_locs', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    
    with pyro.plate('ag_idx', num_agents):
        locs_base = pyro.sample("locs_base", dist.MultivariateNormal(m_locs_base, scale_tril=st_locs_base))
        shift = pyro.sample("shift", dist.MultivariateNormal(m_shift, scale_tril=st_shift))
    return {'locs_base': locs_base, 'shift': shift}

# ----- M_FRAC: Fractional Power (3 Params) -----
def model_m_frac(data, drug_ids):
    num_params = 3
    num_agents, num_trials = data.shape[0], data.shape[2] 
    
    m_base = pyro.param('m_base', torch.zeros(2, num_params, device=device))
    s_base = pyro.param('s_base', torch.ones(2, num_params, device=device), constraint=dist.constraints.positive)
    m_shift = pyro.param('m_shift', torch.zeros(2, num_params, device=device))
    s_shift = pyro.param('s_shift', torch.ones(2, num_params, device=device), constraint=dist.constraints.positive)
    
    with pyro.plate('ag_idx', num_agents):
        ab_m, ab_s = m_base[drug_ids], s_base[drug_ids]
        as_m, as_s = m_shift[drug_ids], s_shift[drug_ids]
        
        base_dist = dist.Normal(torch.zeros(num_params, device=device), torch.ones(num_params, device=device)).to_event(1)
        locs_base = pyro.sample('locs_base', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(ab_m, ab_s)]))
        shift = pyro.sample('shift', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(as_m, as_s)]))

    locs_flat = torch.stack([locs_base, locs_base + shift], dim=1).reshape(num_agents * 2, num_params)
    data_flat = data.reshape(num_agents * 2, num_trials, data.shape[-1])

    with pyro.plate('data', num_agents * 2 * num_trials):
        a_param = torch.exp(locs_flat[:,0]).unsqueeze(-1).expand(-1, num_trials)
        s_param = torch.sigmoid(locs_flat[:,1]).unsqueeze(-1).expand(-1, num_trials) * 2.0 
        beta = torch.sigmoid(locs_flat[:,2]).unsqueeze(-1).expand(-1, num_trials)*BETA_BOUND 
        
        sigma_combine = a_param * torch.pow(data_flat[:,:,2] + 1e-8, s_param)
        e_mean = (data_flat[:,:,3]) / (1 + sigma_combine**2) + 1e-8
        
        log1 = torch.log(e_mean) * beta
        log2 = torch.log(data_flat[:,:,5] + 1e-8) * beta
        logits = torch.stack([log1, log2], dim=0)
        p = torch.softmax(logits, dim=0)[0]
        p = torch.clamp(p, min=1e-5, max=1.0-1e-5)
        
    valid_mask = (data_flat[:, :, 5] > 0)
    masked_dist = dist.Bernoulli(probs=p).mask(valid_mask).to_event(2)
    pyro.sample("obs", masked_dist, obs=data_flat[:,:,4])

def guide_m_frac(data, drug_ids):
    num_params, num_agents = 3, data.shape[0]
    m_locs_base = pyro.param('m_locs_base', torch.zeros(num_agents, num_params, device=data.device))
    st_locs_base = pyro.param('st_locs_base', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    m_shift = pyro.param('m_shift_locs', torch.zeros(num_agents, num_params, device=data.device))
    st_shift = pyro.param('st_shift_locs', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    
    with pyro.plate('ag_idx', num_agents):
        locs_base = pyro.sample("locs_base", dist.MultivariateNormal(m_locs_base, scale_tril=st_locs_base))
        shift = pyro.sample("shift", dist.MultivariateNormal(m_shift, scale_tril=st_shift))
    return {'locs_base': locs_base, 'shift': shift}


# %% [4] Engine for Training and WAIC
def run_model_pipeline(model_name, data, drug_ids):
    print(f"\n{'='*40}\nSTARTING PIPELINE FOR {model_name}\n{'='*40}")
    pyro.clear_param_store()
    
    models = {"M1": (model_m1, guide_m1), "M3": (model_m3, guide_m3), 
              "M5": (model_m5, guide_m5), "M_LOG": (model_m_log, guide_m_log), 
              "M_FRAC": (model_m_frac, guide_m_frac)}
              
    active_model, active_guide = models[model_name]
    sample_file = f"{model_name}_MIXED_samples.pt"
        
    # ---------------------------------------------------------
    # TRAINING OR LOADING
    # ---------------------------------------------------------
    if TRAIN_MODELS:
        n_steps = 2 if ('CI' in os.environ) else 5000
        optimizer = Adam({"lr": 0.01})
        svi = SVI(active_model, active_guide, optimizer, loss=Trace_ELBO())

        loss = []
        pbar = tqdm(range(n_steps), position=0)
        for step in pbar:
            loss.append(torch.tensor(svi.step(data, drug_ids)))
            pbar.set_description(f"{model_name} Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
            if torch.isnan(loss[-1]):
                print("Encountered NaN loss. Breaking loop.")
                break

        plt.figure(figsize=(10, 4))
        plt.plot(loss)
        plt.title(f"{model_name} ELBO minimization (Mixed Design)")
        plt.show()

        print("Extracting 1000 Posteriors and Saving to Disk...")
        samples = [active_guide(data, drug_ids) for _ in range(1000)]
        torch.save(samples, sample_file)
        
    else:
        print(f"LOADING SAVED MODEL FOR {model_name}...")
        try:
            samples = torch.load(sample_file, weights_only=False)
            print("Successfully loaded saved posterior samples.")
        except FileNotFoundError:
            print(f"Error: {sample_file} not found. Please set TRAIN_MODELS = True.")
            return torch.tensor([0.]), None

    # ---------------------------------------------------------
    # WAIC CALCULATION (Properly Masked)
    # ---------------------------------------------------------
    print("Calculating Masked WAIC...")
    num_agents = data.shape[0]
    num_trials = data.shape[2]
    
    locs = []
    for s in samples:
        lb = s['locs_base'].detach()
        sh = s['shift'].detach()
        if model_name in ["M1", "M5"]: num_params = 2
        elif model_name in ["M_LOG", "M_FRAC"]: num_params = 3
        elif model_name == "M3": num_params = 4
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
            sigma_rate = torch.exp(i[:,0]).unsqueeze(-1).expand(-1, num_trials) 
            a_param = torch.exp(i[:,1]).unsqueeze(-1).expand(-1, num_trials)
            b_param = torch.exp(i[:,2]).unsqueeze(-1).expand(-1, num_trials)
            beta = torch.sigmoid(i[:,3]).unsqueeze(-1).expand(-1, num_trials) * BETA_BOUND  
            sigma_combine = sigma_rate / (1 + b_param * torch.exp(-a_param * data_flat[:,:,2]))
            e_mean = (data_flat[:,:,3]) / (1 + sigma_combine**2) + 1e-8
            
        elif model_name == "M5":
            k_param = torch.exp(i[:,0]).unsqueeze(-1).expand(-1, num_trials)
            beta = torch.sigmoid(i[:,1]).unsqueeze(-1).expand(-1, num_trials) * BETA_BOUND
            
            # 1. Calculate raw subjective values
            v_ll = (data_flat[:,:,3]) / (1 + k_param * data_flat[:,:,2])
            v_ss = data_flat[:,:,5]
            
            # 2. Apply Classical Linear Softmax (No Logs!)
            logit_ll = v_ll * beta
            logit_ss = v_ss * beta
            
            logits = torch.stack([logit_ll, logit_ss], dim=0)
            p = torch.softmax(logits, dim=0)[0]
            p = torch.clamp(p, min=1e-5, max=1.0-1e-5)
            probs.append(p)
            
            continue 
            
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
        log2 = torch.log(data_flat[:,:,5] + 1e-8) * beta
        logits = torch.stack([log1, log2], dim=0)
        p = torch.softmax(logits, dim=0)[0]
        p = torch.clamp(p, min=1e-5, max=1.0-1e-5)
        probs.append(p)

    likeli = []
    for i in probs:
        temp = i * data_flat[:,:,4] + (1 - data_flat[:,:,4]) * (1 - i)
        temp = torch.clamp(temp, min=1e-8, max=1.0) 
        likeli.append(temp.cpu())

    final = torch.stack(likeli).permute(1, 2, 0)
    log_likes = torch.log(final)
    
    valid_mask_cpu = (data_flat[:, :, 5] > 0).float()
    
    lppd = torch.logsumexp(log_likes, dim=-1) - torch.log(torch.tensor(1000.))
    lppd_sum = (lppd * valid_mask_cpu).sum(dim=1)
    
    p_waic = log_likes.var(dim=-1, unbiased=True)
    p_waic_sum = (p_waic * valid_mask_cpu).sum(dim=1)
    
    subj_waic = -2 * (lppd_sum - p_waic_sum)
    total_waic = subj_waic.sum()
    se_waic = torch.sqrt(subj_waic.shape[0] * subj_waic.var(unbiased=True))

    print(f"--> {model_name} Total WAIC: {total_waic.item():.2f} (SE: {se_waic.item():.2f})")
    
    # ---------------------------------------------------------
    # POSTERIOR PREDICTIVE CHECKS (DETERMINISTIC ACCURACY)
    # ---------------------------------------------------------
    print("Calculating Deterministic Predictive Accuracy...")
    
    valid_trials_per_subj = valid_mask_cpu.sum(dim=1)
    mean_probs = torch.stack(probs).mean(dim=0)

    predicted_choices = (mean_probs > 0.5).float()
    actual_choices = data_flat[:, :, 4]

    correct_matches = (predicted_choices == actual_choices).float() * valid_mask_cpu
    correct_per_agent = correct_matches.sum(dim=1)
    agent_accuracies = (correct_per_agent / (valid_trials_per_subj + 1e-8)) * 100

    mean_acc = agent_accuracies.mean().item()
    sd_acc = agent_accuracies.std().item()
    min_acc = agent_accuracies.min().item()
    max_acc = agent_accuracies.max().item()
    range_acc = max_acc - min_acc

    total_correct_all = correct_per_agent.sum().item()
    total_trials_all = valid_trials_per_subj.sum().item()
    overall_accuracy = (total_correct_all / total_trials_all) * 100

    print(f"--> {model_name} Overall Hit Rate (Pooled): {overall_accuracy:.2f}%")
    print(f"    Participant Accuracies: Mean = {mean_acc:.2f}%, SD = {sd_acc:.2f}%, Min = {min_acc:.2f}%, Max = {max_acc:.2f}% (Range = {range_acc:.2f}%)")
    
    return subj_waic, samples


# %% [5] Execution Loop
if SELECTED_MODEL == "ALL":
    models_to_run = ["M1", "M3", "M5", "M_LOG", "M_FRAC"]
else:
    models_to_run = [SELECTED_MODEL]
    
results = {}
all_samples = {}

for m in models_to_run:
    waic, samps = run_model_pipeline(m, data, drug_ids)
    results[m] = waic
    all_samples[m] = samps


# %% [6] FINAL MODEL RANKING & WAIC COMPARISON PLOT
import torch
import matplotlib.pyplot as plt

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

    model_names, delta_waics, se_deltas_2x = [], [], []
    for name, _ in ranked:
        comp_waic_tensor = results[name]
        delta_tensor = comp_waic_tensor - best_waic_tensor
        d_waic = delta_tensor.sum().item()
        se_d = torch.sqrt(delta_tensor.shape[0] * delta_tensor.var(unbiased=True)).item()
        
        model_names.append(name)
        delta_waics.append(d_waic)
        se_deltas_2x.append(se_d * 2)

    plt.rcParams.update({
        'font.size': 16, 'axes.titlesize': 16, 'axes.titleweight': 'bold',
        'axes.labelsize': 16, 'text.color': 'black', 'axes.labelcolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black', 'figure.facecolor': 'none', 
        'axes.facecolor': 'none', 'savefig.facecolor': 'none',
        'savefig.edgecolor': 'none', 'legend.fontsize': 16, 'axes.grid': False
    })

    print(f"\n{'='*40}\nGENERATING WAIC COMPARISON PLOT\n{'='*40}")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.errorbar(delta_waics, model_names, xerr=se_deltas_2x, fmt='o', color='black', ecolor='black', elinewidth=2.5, capsize=6, capthick=2.5, markersize=10)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Baseline ({best_model_name})')
    
    ax.set_xlabel(r'$\Delta$ WAIC (Relative to Best Model)')
    ax.set_title('Model Comparison \n')
    ax.invert_yaxis() 
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(width=1.5, length=7, color='black')
    
    ax.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    plt.show()


# %% [7] Plot Hyperparameter Posteriors (Independent Mixed Design Cell)
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BETA_BOUND = 20.0  

def calc_hdi(samples, credible_mass=0.95):
    sorted_samples = np.sort(samples)
    interval_idx_inc = int(np.floor(credible_mass * len(sorted_samples)))
    n_intervals = len(sorted_samples) - interval_idx_inc
    interval_width = sorted_samples[interval_idx_inc:] - sorted_samples[:n_intervals]
    min_idx = np.argmin(interval_width)
    return sorted_samples[min_idx], sorted_samples[min_idx + interval_idx_inc]

def transform_to_nat(base_raw, cond2_raw, model_name):
    # Transforms log/logit space back into NATURAL space for plotting
    if model_name == "M1":
        param_names = ["a (Scaling)", r"$\beta$"]
        base_nat = torch.stack([torch.exp(base_raw[:,0]), torch.sigmoid(base_raw[:,1]) * BETA_BOUND], dim=1)
        cond2_nat = torch.stack([torch.exp(cond2_raw[:,0]), torch.sigmoid(cond2_raw[:,1]) * BETA_BOUND], dim=1)
        
    elif model_name == "M3":
        param_names = ["sigma_rate (Plateau)", "a (Slope)", "b (Intercept)", r"$\beta$"]
        base_nat = torch.stack([torch.exp(base_raw[:,0]), torch.exp(base_raw[:,1]), torch.exp(base_raw[:,2]), torch.sigmoid(base_raw[:,3]) * BETA_BOUND], dim=1)
        cond2_nat = torch.stack([torch.exp(cond2_raw[:,0]), torch.exp(cond2_raw[:,1]), torch.exp(cond2_raw[:,2]), torch.sigmoid(cond2_raw[:,3]) * BETA_BOUND], dim=1)
        
    elif model_name == "M5":
        param_names = ["k (Discount Rate)", r"Softmax $\beta$"]
        base_nat = torch.stack([torch.exp(base_raw[:,0]), torch.sigmoid(base_raw[:,1]) * BETA_BOUND], dim=1)
        cond2_nat = torch.stack([torch.exp(cond2_raw[:,0]), torch.sigmoid(cond2_raw[:,1]) * BETA_BOUND], dim=1)
        
    elif model_name == "M_LOG":
        param_names = ["a (Amplitude)", "b (Compression)", r"$\beta$"]
        base_nat = torch.stack([torch.exp(base_raw[:,0]), torch.exp(base_raw[:,1]), torch.sigmoid(base_raw[:,2]) * BETA_BOUND], dim=1)
        cond2_nat = torch.stack([torch.exp(cond2_raw[:,0]), torch.exp(cond2_raw[:,1]), torch.sigmoid(cond2_raw[:,2]) * BETA_BOUND], dim=1)
        
    elif model_name == "M_FRAC":
        param_names = ["a (Amplitude)", "s (Exponent)", r"$\beta$"]
        base_nat = torch.stack([torch.exp(base_raw[:,0]), torch.sigmoid(base_raw[:,1]) * 2.0, torch.sigmoid(base_raw[:,2]) * BETA_BOUND], dim=1)
        cond2_nat = torch.stack([torch.exp(cond2_raw[:,0]), torch.sigmoid(cond2_raw[:,1]) * 2.0, torch.sigmoid(cond2_raw[:,2]) * BETA_BOUND], dim=1)
        
    return param_names, base_nat, cond2_nat

def plot_mixed_design_posteriors(model_name, samples, c_ids):
    print(f"\nGenerating Mixed Design plot for {model_name}...")
    
    label_base = "Placebo"
    label_cond2 = "Haloperidol"
    
    # 1. Stack the posterior samples 
    stacked_base = torch.stack([s['locs_base'].detach().cpu() for s in samples])
    stacked_shift = torch.stack([s['shift'].detach().cpu() for s in samples])
    
    # 2. Filter agents by drug condition and average to get group means
    base_raw_pbo = stacked_base[:, c_ids == 0, :].mean(dim=1) 
    base_raw_hal = stacked_base[:, c_ids == 1, :].mean(dim=1)
    
    # These ARE the shifts (the raw latent changes)
    shift_raw_pbo = stacked_shift[:, c_ids == 0, :].mean(dim=1)
    shift_raw_hal = stacked_shift[:, c_ids == 1, :].mean(dim=1)

    # 3. Transform ONLY the Baseline into Natural Space for the left column
    param_names, base_nat_pbo = transform_to_nat_base_only(base_raw_pbo, model_name)
    _, base_nat_hal = transform_to_nat_base_only(base_raw_hal, model_name)
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
        # LEFT COLUMN: Drug Effect on BASELINE (SS=20)
        sns.kdeplot(base_nat_pbo[:, p_idx].numpy(), ax=axes[p_idx, 0], fill=True, label=label_base, color="#4c72b0", linewidth=2.5, alpha=0.5)
        sns.kdeplot(base_nat_hal[:, p_idx].numpy(), ax=axes[p_idx, 0], fill=True, label=label_cond2, color="#dd8452", linewidth=2.5, alpha=0.5)
        
        axes[p_idx, 0].set_title(f"Baseline (SS=20): {p_name}")
        axes[p_idx, 0].set_ylabel("Density")
        
        # RIGHT COLUMN: Raw Latent Shift
        pbo_shift_data = shift_raw_pbo[:, p_idx].numpy()
        hal_shift_data = shift_raw_hal[:, p_idx].numpy()
        
        sns.kdeplot(pbo_shift_data, ax=axes[p_idx, 1], fill=True, label="Placebo Shift", color="#55a868", linewidth=2.5, alpha=0.5)
        sns.kdeplot(hal_shift_data, ax=axes[p_idx, 1], fill=True, label="Haloperidol Shift", color="#c44e52", linewidth=2.5, alpha=0.5)
        axes[p_idx, 1].axvline(0, color='black', linestyle=':', linewidth=2, alpha=0.8) 
        axes[p_idx, 1].set_title(f"Magnitude Shift (Latent $\Delta$): {p_name}")
        axes[p_idx, 1].set_ylabel("")
        
        # ONLY add legends to the very last row (the Softmax Beta row)
        if p_idx == len(param_names) - 1:
            axes[p_idx, 0].legend(loc='lower center', bbox_to_anchor=(0.5, -0.6), ncol=2, frameon=False)
            axes[p_idx, 1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.6), ncol=2, frameon=False)
        
        apply_bentheme(axes[p_idx, 0])
        apply_bentheme(axes[p_idx, 1])
        
    plt.suptitle(f"{model_name} Hyperparameter Posteriors (Mixed Design)", fontsize=20, y=1.05, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Helper function to avoid the cond2 transform dependency
def transform_to_nat_base_only(base_raw, model_name):
    if model_name == "M1":
        names = ["a (Scaling)", r"Softmax $\beta$"]
        nat = torch.stack([torch.exp(base_raw[:,0]), torch.sigmoid(base_raw[:,1]) * BETA_BOUND], dim=1)
    elif model_name == "M3":
        names = ["sigma_rate (Plateau)", "a (Slope)", "b (Intercept)", r"Softmax $\beta$"]
        nat = torch.stack([torch.exp(base_raw[:,0]), torch.exp(base_raw[:,1]), torch.exp(base_raw[:,2]), torch.sigmoid(base_raw[:,3]) * BETA_BOUND], dim=1)
    elif model_name == "M5":
        names = ["k (Discount Rate)", r"Softmax $\beta$"]
        nat = torch.stack([torch.exp(base_raw[:,0]), torch.sigmoid(base_raw[:,1]) * BETA_BOUND], dim=1)
    elif model_name == "M_LOG":
        names = ["a (Amplitude)", "b (Compression)", r"Softmax $\beta$"]
        nat = torch.stack([torch.exp(base_raw[:,0]), torch.exp(base_raw[:,1]), torch.sigmoid(base_raw[:,2]) * BETA_BOUND], dim=1)
    elif model_name == "M_FRAC":
        names = ["a (Amplitude)", "s (Exponent)", r"Softmax $\beta$"]
        nat = torch.stack([torch.exp(base_raw[:,0]), torch.sigmoid(base_raw[:,1]) * 2.0, torch.sigmoid(base_raw[:,2]) * BETA_BOUND], dim=1)
    return names, nat

# --- EXECUTE PLOTTING ---
d_ids_cpu = drug_ids.cpu() 
for m in models_to_run:
    if all_samples[m] is not None:
        plot_mixed_design_posteriors(m, all_samples[m], d_ids_cpu)