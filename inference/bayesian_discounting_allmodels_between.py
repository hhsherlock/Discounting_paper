#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 2026
@author: benwagner

BETWEEN-SUBJECTS PIPELINE (Haloperidol CSV)
Models: M1, M3, M5, M_LOG, M_FRAC
Features: Conditional Group-Level Shrinkage, WAIC Plotting, Deterministic PPCs, Dynamic SS.
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

# ==========================================
# MASTER CONTROL PANEL BETWEEN
# ==========================================
# Options: "M1", "M3", "M5", "M_LOG", "M_FRAC", or "ALL"
SELECTED_MODEL = "ALL"        
# ==========================================

BETA_BOUND = 10.0  #max inverse temperature

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")


# %%Data Loading & Preprocessing (Between-Subjects)
print("Loading Haloperidol CSV for Between-Subjects Design...")
df = pd.read_csv('Haloperidol_Discounting_Dataset_Softmax.csv')

#sort by Condition FIRST, then Subject, then Trial
df = df.sort_values(by=['Condition', 'realID', 'Trial'])

subjects = df['realID'].unique()
num_subjects = len(subjects)

trials_per_subj = df.groupby('realID').size()
num_trials = trials_per_subj.max()

feature_matrix = np.zeros((num_subjects, num_trials, 6))
cond_ids = torch.zeros(num_subjects, dtype=torch.long).to(device)

for i, subj in enumerate(subjects):
    subj_data = df[df['realID'] == subj]
    actual_trials = len(subj_data)
    
    feature_matrix[i, :actual_trials, 2] = subj_data['Delay'].values
    feature_matrix[i, :actual_trials, 3] = subj_data['Value_LL'].values
    feature_matrix[i, :actual_trials, 4] = subj_data['Choice'].values
    # Map dynamic SS Value to index 5
    feature_matrix[i, :actual_trials, 5] = subj_data['Value_SS'].values
    
    # Map Condition 1 -> 0, Condition 2 -> 1 for matrix routing
    cond_val = subj_data['Condition'].iloc[0]
    cond_ids[i] = 0 if cond_val == 1 else 1
    
data = torch.tensor(feature_matrix, dtype=torch.float32).to(device)
print(f"Data shape successfully set to: {data.shape}")


# %% ##Model Definitions (M1, M3, M5, M_LOG, M_FRAC)

# M1: Square Root (2 Params) -----
def model_m1(data, cond_ids):
    num_params = 2
    num_agents, num_trials = data.shape[0], data.shape[1]
    m = pyro.param('m', torch.zeros(2, num_params, device=device))
    s = pyro.param('s', torch.ones(2, num_params, device=device), constraint=dist.constraints.positive)
    
    with pyro.plate('ag_idx', num_agents):
        agent_m, agent_s = m[cond_ids], s[cond_ids]
        base_dist = dist.Normal(torch.zeros(num_params, device=device), torch.ones(num_params, device=device)).to_event(1)
        locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(agent_m, agent_s)]))

    with pyro.plate('data', num_agents*num_trials):
        a_param = torch.exp(locs[:,0]).unsqueeze(-1).expand(-1, num_trials)
        beta = torch.sigmoid(locs[:,1]).unsqueeze(-1).expand(-1, num_trials)*BETA_BOUND  #max inverse temperature
        
        sigma_combine = a_param * torch.sqrt(data[:,:,2])
        e_mean = (data[:,:,3])/(1 + sigma_combine**2) + 1e-8
        
        log1 = torch.log(e_mean) * beta
        log2 = torch.log(data[:,:,5]) * beta
        logits = torch.stack([log1, log2], dim=0)
        p = torch.softmax(logits, dim=0)[0]
    pyro.sample("obs", dist.Bernoulli(probs=p).to_event(2), obs=data[:,:,4])

def guide_m1(data, cond_ids):
    num_params, num_agents = 2, data.shape[0]
    m_locs = pyro.param('m_locs', torch.zeros(num_agents, num_params, device=data.device))
    st_locs = pyro.param('scale_tril_locs', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    with pyro.plate('ag_idx', num_agents):
        locs = pyro.sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))
    return {'locs': locs}

#
#M3: Sigmoid (4 Params)
def model_m3(data, cond_ids):
    num_params = 4
    num_agents, num_trials = data.shape[0], data.shape[1]
    a = pyro.param('a', torch.ones(2, num_params, device=device), constraint=dist.constraints.positive)
    lam = pyro.param('lam', torch.ones(2, num_params, device=device), constraint=dist.constraints.positive)
    
    # FIX to_event(2) to consume both the Condition and Parameter dimensions
    tau = pyro.sample('tau', dist.Gamma(a, a/lam).to_event(2)) 
    sig = pyro.deterministic('sig', 1/torch.sqrt(tau)) 
    
    m = pyro.param('m', torch.zeros(2, num_params, device=device))
    s = pyro.param('s', torch.ones(2, num_params, device=device), constraint=dist.constraints.positive)
    
    #changed to to_event(2) to match tau 
    mu = pyro.sample('mu', dist.Normal(m, s*sig).to_event(2)) 

    with pyro.plate('ag_idx', num_agents):
        agent_mu, agent_sig = mu[cond_ids], sig[cond_ids]
        base_dist = dist.Normal(torch.zeros(num_params, device=device), torch.ones(num_params, device=device)).to_event(1)
        locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(agent_mu, agent_sig)]))

    with pyro.plate('data', num_agents*num_trials):
        sigma_rate = torch.exp(locs[:,0]).unsqueeze(-1).expand(-1, num_trials)
        a_param = torch.exp(locs[:,1]).unsqueeze(-1).expand(-1, num_trials)
        b_param = torch.exp(locs[:,2]).unsqueeze(-1).expand(-1, num_trials)
        beta = torch.sigmoid(locs[:,3]).unsqueeze(-1).expand(-1, num_trials)*BETA_BOUND  #max inverse temperature
        
        sigma_combine = sigma_rate/(1+b_param*torch.exp(-a_param*data[:,:,2]))
        e_mean = (data[:,:,3])/(1 + sigma_combine**2) + 1e-8
        
        log1 = torch.log(e_mean) * beta
        log2 = torch.log(data[:,:,5]) * beta
        logits = torch.stack([log1, log2], dim=0)
        p = torch.softmax(logits, dim=0)[0]
    pyro.sample("obs", dist.Bernoulli(probs=p).to_event(2), obs=data[:,:,4])

def guide_m3(data, cond_ids):
    num_params, num_agents = 4, data.shape[0]
    trns = torch.distributions.biject_to(dist.constraints.positive)
    
    m_hyp = pyro.param('m_hyp', torch.zeros(2, 2*num_params, device=data.device))
    st_hyp = pyro.param('scale_tril_hyp', torch.eye(2*num_params, device=data.device).repeat(2, 1, 1), constraint=dist.constraints.lower_cholesky)
    
    # modified
    hyp = pyro.sample('hyp', dist.MultivariateNormal(m_hyp, scale_tril=st_hyp).to_event(1), infer={'is_auxiliary': True})
    
    unc_mu, unc_tau = hyp[..., :num_params], hyp[..., num_params:]
    c_tau = trns(unc_tau)
    
    ld_tau = -trns.inv.log_abs_det_jacobian(c_tau, unc_tau)
    
    # Sum everything to a scalar `[]` shape so Pyro accepts it
    ld_tau = ld_tau.sum() 
    
    #Set to event_dim=2 to mirror the model
    mu = pyro.sample("mu", dist.Delta(unc_mu, event_dim=2))
    tau = pyro.sample("tau", dist.Delta(c_tau, log_density=ld_tau, event_dim=2))
    
    m_locs = pyro.param('m_locs', torch.zeros(num_agents, num_params, device=data.device))
    st_locs = pyro.param('scale_tril_locs', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    
    with pyro.plate('ag_idx', num_agents):
        locs = pyro.sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))
        
    return {'tau': tau, 'mu': mu, 'locs': locs}

#M5: Hyperbolic (2 Params) -----
def model_m5(data, cond_ids):
    num_params = 2
    num_agents, num_trials = data.shape[0], data.shape[1]
    m = pyro.param('m', torch.zeros(2, num_params, device=device))
    s = pyro.param('s', torch.ones(2, num_params, device=device), constraint=dist.constraints.positive)
    
    with pyro.plate('ag_idx', num_agents):
        agent_m, agent_s = m[cond_ids], s[cond_ids]
        base_dist = dist.Normal(torch.zeros(num_params, device=device), torch.ones(num_params, device=device)).to_event(1)
        locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(agent_m, agent_s)]))
        
    with pyro.plate('data', num_agents*num_trials):
        k_param = torch.exp(locs[:,0]).unsqueeze(-1).expand(-1, num_trials)
        beta = torch.sigmoid(locs[:,1]).unsqueeze(-1).expand(-1, num_trials)*BETA_BOUND  #max inverse temperature
        e_mean = (data[:,:,3]) / (1 + k_param * data[:,:,2]) + 1e-8
        
        log1 = torch.log(e_mean) * beta
        log2 = torch.log(data[:,:,5]) * beta
        logits = torch.stack([log1, log2], dim=0)
        p = torch.softmax(logits, dim=0)[0]
    pyro.sample("obs", dist.Bernoulli(probs=p).to_event(2), obs=data[:,:,4])

def guide_m5(data, cond_ids):
    num_params, num_agents = 2, data.shape[0]
    m_locs = pyro.param('m_locs', torch.zeros(num_agents, num_params, device=data.device))
    st_locs = pyro.param('scale_tril_locs', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    with pyro.plate('ag_idx', num_agents):
        locs = pyro.sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))
    return {'locs': locs}

#M_LOG: Logarithmic (3 Params) -----
def model_m_log(data, cond_ids):
    num_params = 3
    num_agents, num_trials = data.shape[0], data.shape[1]
    m = pyro.param('m', torch.zeros(2, num_params, device=device))
    s = pyro.param('s', torch.ones(2, num_params, device=device), constraint=dist.constraints.positive)
    
    with pyro.plate('ag_idx', num_agents):
        agent_m, agent_s = m[cond_ids], s[cond_ids]
        base_dist = dist.Normal(torch.zeros(num_params, device=device), torch.ones(num_params, device=device)).to_event(1)
        locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(agent_m, agent_s)]))
        
    with pyro.plate('data', num_agents*num_trials):
        a_param = torch.exp(locs[:,0]).unsqueeze(-1).expand(-1, num_trials)
        b_param = torch.exp(locs[:,1]).unsqueeze(-1).expand(-1, num_trials)
        beta = torch.sigmoid(locs[:,2]).unsqueeze(-1).expand(-1, num_trials)*BETA_BOUND  #max inverse temperature
        
        sigma_combine = a_param * torch.log(1 + b_param * data[:,:,2])
        e_mean = (data[:,:,3]) / (1 + sigma_combine**2) + 1e-8
        
        log1 = torch.log(e_mean) * beta
        log2 = torch.log(data[:,:,5]) * beta
        logits = torch.stack([log1, log2], dim=0)
        p = torch.softmax(logits, dim=0)[0]
    pyro.sample("obs", dist.Bernoulli(probs=p).to_event(2), obs=data[:,:,4])

def guide_m_log(data, cond_ids):
    num_params, num_agents = 3, data.shape[0]
    m_locs = pyro.param('m_locs', torch.zeros(num_agents, num_params, device=data.device))
    st_locs = pyro.param('scale_tril_locs', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    with pyro.plate('ag_idx', num_agents):
        locs = pyro.sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))
    return {'locs': locs}

#M_FRAC: Fractional Power (3 Params)
def model_m_frac(data, cond_ids):
    num_params = 3
    num_agents, num_trials = data.shape[0], data.shape[1]
    m = pyro.param('m', torch.zeros(2, num_params, device=device))
    s = pyro.param('s', torch.ones(2, num_params, device=device), constraint=dist.constraints.positive)
    
    with pyro.plate('ag_idx', num_agents):
        agent_m, agent_s = m[cond_ids], s[cond_ids]
        base_dist = dist.Normal(torch.zeros(num_params, device=device), torch.ones(num_params, device=device)).to_event(1)
        locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [dist.transforms.AffineTransform(agent_m, agent_s)]))
        
    with pyro.plate('data', num_agents*num_trials):
        a_param = torch.exp(locs[:,0]).unsqueeze(-1).expand(-1, num_trials)
        s_param = torch.sigmoid(locs[:,1]).unsqueeze(-1).expand(-1, num_trials) * 2.0 
        beta = torch.sigmoid(locs[:,2]).unsqueeze(-1).expand(-1, num_trials)*BETA_BOUND #max inverse temperature
        
        sigma_combine = a_param * torch.pow(data[:,:,2] + 1e-8, s_param)
        e_mean = (data[:,:,3]) / (1 + sigma_combine**2) + 1e-8
        
        log1 = torch.log(e_mean) * beta
        log2 = torch.log(data[:,:,5]) * beta
        logits = torch.stack([log1, log2], dim=0)
        p = torch.softmax(logits, dim=0)[0]
    pyro.sample("obs", dist.Bernoulli(probs=p).to_event(2), obs=data[:,:,4])

def guide_m_frac(data, cond_ids):
    num_params, num_agents = 3, data.shape[0]
    m_locs = pyro.param('m_locs', torch.zeros(num_agents, num_params, device=data.device))
    st_locs = pyro.param('scale_tril_locs', torch.eye(num_params, device=data.device).repeat(num_agents, 1, 1), constraint=dist.constraints.lower_cholesky)
    with pyro.plate('ag_idx', num_agents):
        locs = pyro.sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))
    return {'locs': locs}


# %%Engine for Training and WAIC
def run_model_pipeline(model_name, data, cond_ids):
    print(f"\n{'='*40}\nSTARTING PIPELINE FOR {model_name}\n{'='*40}")
    pyro.clear_param_store()
    
    models = {"M1": (model_m1, guide_m1), "M3": (model_m3, guide_m3), 
              "M5": (model_m5, guide_m5), "M_LOG": (model_m_log, guide_m_log), 
              "M_FRAC": (model_m_frac, guide_m_frac)}
              
    active_model, active_guide = models[model_name]
        
    n_steps = 2 if ('CI' in os.environ) else 5000
    optimizer = Adam({"lr": 0.01})
    svi = SVI(active_model, active_guide, optimizer, loss=Trace_ELBO())

    loss = []
    pbar = tqdm(range(n_steps), position=0)
    for step in pbar:
        # NOTE: Passed cond_ids directly into SVI
        loss.append(torch.tensor(svi.step(data, cond_ids)))
        pbar.set_description(f"{model_name} Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
        if torch.isnan(loss[-1]):
            print("Encountered NaN loss. Breaking loop.")
            break

    plt.figure(figsize=(10, 4))
    plt.plot(loss)
    plt.title(f"{model_name} ELBO minimization (Between-Subjects)")
    plt.show()

    print("Extracting Posteriors and Calculating WAIC...")
    sample_num = 1000
    # NOTE: Passed cond_ids directly into active_guide
    locs = [active_guide(data, cond_ids)['locs'].detach().cpu().numpy() for _ in range(sample_num)]
    probs = []
    num_trials = data.shape[1]
    num_agents = data.shape[0]
    
    for i in locs:
        # Parameter Unpacking and WAIC Calculation based on Model
        if model_name == "M1":
            a_param = torch.exp(torch.tensor(i[:,0]).unsqueeze(-1).expand(-1, num_trials).to(device))
            beta = torch.sigmoid(torch.tensor(i[:,1]).unsqueeze(-1).expand(-1, num_trials).to(device)) * BETA_BOUND  #max inverse temperature
            sigma_combine = a_param * torch.sqrt(data[:,:,2])
            e_mean = (data[:,:,3]) / (1 + sigma_combine**2) + 1e-8
            
        elif model_name == "M3":
            sigma_rate = torch.exp(torch.tensor(i[:,0]).unsqueeze(-1).expand(-1, num_trials).to(device))
            a_param = torch.exp(torch.tensor(i[:,1]).unsqueeze(-1).expand(-1, num_trials).to(device))
            b_param = torch.exp(torch.tensor(i[:,2]).unsqueeze(-1).expand(-1, num_trials).to(device))
            beta = torch.sigmoid(torch.tensor(i[:,3]).unsqueeze(-1).expand(-1, num_trials).to(device)) * BETA_BOUND  
            sigma_combine = sigma_rate / (1 + b_param * torch.exp(-a_param * data[:,:,2]))
            e_mean = (data[:,:,3]) / (1 + sigma_combine**2) + 1e-8
            
        elif model_name == "M5":
            k_param = torch.exp(torch.tensor(i[:,0]).unsqueeze(-1).expand(-1, num_trials).to(device))
            beta = torch.sigmoid(torch.tensor(i[:,1]).unsqueeze(-1).expand(-1, num_trials).to(device)) * BETA_BOUND
            e_mean = (data[:,:,3]) / (1 + k_param * data[:,:,2]) + 1e-8
            
        elif model_name == "M_LOG":
            a_param = torch.exp(torch.tensor(i[:,0]).unsqueeze(-1).expand(-1, num_trials).to(device))
            b_param = torch.exp(torch.tensor(i[:,1]).unsqueeze(-1).expand(-1, num_trials).to(device))
            beta = torch.sigmoid(torch.tensor(i[:,2]).unsqueeze(-1).expand(-1, num_trials).to(device)) * BETA_BOUND
            sigma_combine = a_param * torch.log(1 + b_param * data[:,:,2])
            e_mean = (data[:,:,3]) / (1 + sigma_combine**2) + 1e-8
            
        elif model_name == "M_FRAC":
            a_param = torch.exp(torch.tensor(i[:,0]).unsqueeze(-1).expand(-1, num_trials).to(device))
            s_param = torch.sigmoid(torch.tensor(i[:,1]).unsqueeze(-1).expand(-1, num_trials).to(device)) * 2.0
            beta = torch.sigmoid(torch.tensor(i[:,2]).unsqueeze(-1).expand(-1, num_trials).to(device)) * BETA_BOUND
            sigma_combine = a_param * torch.pow(data[:,:,2] + 1e-8, s_param)
            e_mean = (data[:,:,3]) / (1 + sigma_combine**2) + 1e-8

        log1 = torch.log(e_mean) * beta
        log2 = torch.log(data[:,:,5]) * beta
        logits = torch.stack([log1, log2], dim=0)
        probs.append(torch.softmax(logits, dim=0)[0])

    likeli = []
    for i in probs:
        temp = i * data[:,:,4] + (1 - data[:,:,4]) * (1 - i)
        temp = torch.clamp(temp, min=1e-8, max=1.0) 
        likeli.append(temp.cpu())

    final = torch.stack(likeli).permute(1, 2, 0)
    log_likes = torch.log(final)
    lppd = torch.logsumexp(log_likes, dim=-1) - torch.log(torch.tensor(1000.))
    lppd_sum = lppd.sum(dim=1)
    p_waic = log_likes.var(dim=-1, unbiased=True).sum(dim=1)
    subj_waic = -2 * (lppd_sum - p_waic)
    total_waic = subj_waic.sum()
    se_waic = torch.sqrt(subj_waic.shape[0] * subj_waic.var(unbiased=True))

    print(f"--> {model_name} Total WAIC: {total_waic.item():.2f} (SE: {se_waic.item():.2f})")
    
    
    # ---------------------------------------------------------
    # POSTERIOR PREDICTIVE CHECKS (DETERMINISTIC ACCURACY)
    # ---------------------------------------------------------
    print("Calculating Deterministic Predictive Accuracy...")
    
    valid_mask = (data[:, :, 5] > 0).float() 
    valid_trials_per_subj = valid_mask.sum(dim=1)
    mean_probs = torch.stack(probs).mean(dim=0)

    predicted_choices = (mean_probs > 0.5).float()
    actual_choices = data[:, :, 4]

    correct_matches = (predicted_choices == actual_choices).float() * valid_mask
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
    
    df_acc = pd.DataFrame({
        'Subject_Idx': np.arange(num_agents),
        'Total_Trials': valid_trials_per_subj.cpu().numpy(),
        'Correct_Predictions': correct_per_agent.cpu().numpy(),
        'Accuracy_%': agent_accuracies.cpu().numpy()
    })
    
    file_name = f"{model_name}_CSV_Predictive_Accuracy.csv"
    df_acc.to_csv(file_name, index=False)
    print(f"--> Saved detailed participant accuracy to: {file_name}\n")
    
    return subj_waic


# %% [5] Execution & Comparison
if SELECTED_MODEL == "ALL":
    models_to_run = ["M1", "M3", "M5", "M_LOG", "M_FRAC"]
else:
    models_to_run = [SELECTED_MODEL]
    
results = {}

for m in models_to_run:
    results[m] = run_model_pipeline(m, data, cond_ids)

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

# %% [6] WAIC Comparison Plot
if len(results) > 1:
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
    plt.title('Model Comparison (Between-Subjects)\nError bars represent $2 \\times SE_{\\Delta}$')
    plt.gca().invert_yaxis() 
    plt.legend()
    plt.tight_layout()
    plt.show()