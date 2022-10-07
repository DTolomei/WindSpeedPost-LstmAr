import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions
from src.models.SchaakeEMOS import SchaakeEMOS as EMOS
from src.models.ARMOS import ARMOS
from src.models.validation import energy_score, variogram_score, schaake_shuffle

# ---------------- LOAD DATA ----------------
nobackup = '/net/pc160111/nobackup/users/teixeira/'
val_folder = nobackup + 'norm_test_data/'
train_folder = nobackup + 'norm_data/'

with open(nobackup + 'st-info.pkl', 'rb') as f:
    info = pkl.load(f)

val_files  = [val_folder + file for file in os.listdir(val_folder) if ('20181' in file or '20190' in file)]
train_files  = [train_folder + file for file in os.listdir(train_folder)]

var_list = ['PRESS', 'KINETIC', 'HUMID', 'GEOPOT']
omit = ['229', '285', '323']
sts  = [st for st in info.keys() if st not in omit]

with open(nobackup + 'LoadedTrainData.pkl', 'rb') as f:
    tag, S2, var, obs = pkl.load(f)
with open(nobackup + 'LoadedValData.pkl', 'rb') as f:
    val_tag, val_S2, val_var, val_obs = pkl.load(f)

# ---------------- LOAD MODELS ----------------

# Init models
emos = EMOS(5, 49)
armos1 = ARMOS(5, 1, 49)
armos2 = ARMOS(5, 2, 49)
armos3 = ARMOS(5, 3, 49)

# Load
with open('../res/models_final/GlobalEmos.pkl', 'rb') as f:
    params = pkl.load(f)
emos.a, emos.b, emos.c, emos.d = params['a'], params['b'], params['c'], params['d']

with open('../res/models_final/GlobalArmos1.pkl', 'rb') as f:
    params = pkl.load(f)
armos1.a, armos1.b, armos1.c, armos1.d, armos1.phi = params['a'], params['b'], params['c'], params['d'], params['phi']

with open('../res/models_final/GlobalArmos2.pkl', 'rb') as f:
    params = pkl.load(f)
armos2.a, armos2.b, armos2.c, armos2.d, armos2.phi = params['a'], params['b'], params['c'], params['d'], params['phi']

with open('../res/models_final/GlobalArmos3.pkl', 'rb') as f:
    params = pkl.load(f)
armos3.a, armos3.b, armos3.c, armos3.d, armos3.phi = params['a'], params['b'], params['c'], params['d'], params['phi']

# ---------------- MULTIVARIATE SCORES ----------------
emos_score = {'LS':[],'ES':[],'VS':[]}
armos1_score = {'LS':[],'ES':[],'VS':[]}
armos2_score = {'LS':[],'ES':[],'VS':[]}
armos3_score = {'LS':[],'ES':[],'VS':[]}
armos1_score_ss = {'LS':[],'ES':[],'VS':[]}
armos2_score_ss = {'LS':[],'ES':[],'VS':[]}
armos3_score_ss = {'LS':[],'ES':[],'VS':[]}

w = 0.5**np.array([[np.abs(i-j)-1 for i in range(49)] for j in range(49)], dtype='float32')

emos_samples  = emos.schaake_shuffle(val_tag, val_var, val_S2, tag, obs, n_samples=30)
armos1_samples = armos1.sample(val_var, val_S2, n_samples=30)
armos2_samples = armos2.sample(val_var, val_S2, n_samples=30)
armos3_samples = armos3.sample(val_var, val_S2, n_samples=30)
armos1_samples_ss = schaake_shuffle(val_tag, armos1_samples, tag, obs)
armos2_samples_ss = schaake_shuffle(val_tag, armos2_samples, tag, obs)
armos3_samples_ss = schaake_shuffle(val_tag, armos3_samples, tag, obs)

emos_score['LS'] = emos.loss(val_var, val_S2, val_obs).numpy().sum(-1).mean()
emos_score['ES'] = energy_score(emos_samples, val_obs).mean()
emos_score['VS'] = variogram_score(emos_samples, val_obs, w=w).mean()

armos1_score['LS'] = armos1.loss(val_var, val_S2, val_obs).numpy().sum(-1).mean()
armos1_score['ES'] = energy_score(armos1_samples, val_obs).mean()
armos1_score['VS'] = variogram_score(armos1_samples, val_obs, w=w).mean()

armos2_score['LS'] = armos2.loss(val_var, val_S2, val_obs).numpy().sum(-1).mean()
armos2_score['ES'] = energy_score(armos2_samples, val_obs).mean()
armos2_score['VS'] = variogram_score(armos2_samples, val_obs, w=w).mean()

armos3_score['LS'] = armos3.loss(val_var, val_S2, val_obs).numpy().sum(-1).mean()
armos3_score['ES'] = energy_score(armos3_samples, val_obs).mean()
armos3_score['VS'] = variogram_score(armos3_samples, val_obs, w=w).mean()

armos1_score_ss['LS'] = armos1.loss(val_var, val_S2, val_obs).numpy().sum(-1).mean()
armos1_score_ss['ES'] = energy_score(armos1_samples_ss, val_obs).mean()
armos1_score_ss['VS'] = variogram_score(armos1_samples_ss, val_obs, w=w).mean()

armos2_score_ss['LS'] = armos2.loss(val_var, val_S2, val_obs).numpy().sum(-1).mean()
armos2_score_ss['ES'] = energy_score(armos2_samples_ss, val_obs).mean()
armos2_score_ss['VS'] = variogram_score(armos2_samples_ss, val_obs, w=w).mean()

armos3_score_ss['LS'] = armos3.loss(val_var, val_S2, val_obs).numpy().sum(-1).mean()
armos3_score_ss['ES'] = energy_score(armos3_samples_ss, val_obs).mean()
armos3_score_ss['VS'] = variogram_score(armos3_samples_ss, val_obs, w=w).mean()

global_results = {
    'Emos':emos_score,
    'Armos1':armos1_score, 
    'Armos2':armos2_score, 
    'Armos3':armos3_score, 
    'Armos1/SS':armos1_score_ss,
    'Armos2/SS':armos2_score_ss,
    'Armos3/SS':armos3_score_ss
}
with open('../res/FinalResults/GlobalResults.pkl', 'wb') as f:
    pkl.dump(global_results,f)

# ---------------- UNIVARIATE VALIDATION ----------------
emos_samples  = emos.forecast_dist(val_var, val_S2).sample(100).numpy()
armos1_samples = armos1.sample(val_var, val_S2, n_samples=100)
armos2_samples = armos2.sample(val_var, val_S2, n_samples=100)
armos3_samples = armos3.sample(val_var, val_S2, n_samples=100)

with open('../res/models_final/TrainClimatology.pkl', 'rb') as f:
    climatology = pkl.load(f)
global_emos_ref = {}

# BRIER SKILL SCORE
thresh = np.arange(1.0, 21.0, 1.0)

for leadt in [12,24,36,48]:
    print('Leadt:', leadt)
    hi = (val_obs[..., None, leadt] >= thresh).numpy()

    clim_bs = climatology[leadt, :len(thresh)]
    clim_bs = (clim_bs - hi)**2
    clim_bs = clim_bs.mean(0)

    emos_bs = (emos_samples[..., None, leadt] >= thresh).mean(0)
    emos_bs = (emos_bs - hi)**2
    emos_bs =  emos_bs.mean(0)
    global_emos_ref[leadt] = emos_bs

    armos1_bs = (armos1_samples[..., None, leadt] >= thresh).mean(0)
    armos1_bs = (armos1_bs - hi)**2
    armos1_bs =  armos1_bs.mean(0)

    armos2_bs = (armos2_samples[..., None, leadt] >= thresh).mean(0)
    armos2_bs = (armos2_bs - hi)**2
    armos2_bs =  armos2_bs.mean(0)

    armos3_bs = (armos3_samples[..., None, leadt] >= thresh).mean(0)
    armos3_bs = (armos3_bs - hi)**2
    armos3_bs =  armos3_bs.mean(0)

    plt.hlines(0.0,1,20, lw=1, ls='dashed', color='k', label='Climat.')
    plt.plot(thresh, 1 - emos_bs  /clim_bs, label='EMOS')
    plt.plot(thresh, 1 - armos1_bs/clim_bs, label='ARMOS(1)')
    plt.plot(thresh, 1 - armos2_bs/clim_bs, label='ARMOS(2)')
    plt.plot(thresh, 1 - armos3_bs/clim_bs, label='ARMOS(3)')
    plt.xlim([1,20])
    plt.legend()
    plt.ylabel('BSS')
    plt.xlabel('Windspeed threshold (m/s)')
    ax = plt.gca()
    ax.figure.set_size_inches(5, 3)
    plt.savefig(f'../res/FinalResults/GlobalBrierSkillScore{leadt}h.png')
    plt.show()

# RELIABILITY DIAGRAMS
bins = np.arange(0.0, 1.1, 0.1)
probs = np.arange(0.05, 1.0, 0.1)

for leadt in [12,24,36,48]:
    for thresh in [15.0]:
        print('Leadt:', leadt, ', Thresh:', thresh)
        hi = (val_obs[..., leadt] >= thresh).numpy()

        emos_prob = (emos_samples[..., leadt] >= thresh).mean(0)
        emos_dig  = probs[np.digitize(emos_prob,bins[1:], right=True)]
        emos_rel  = np.array([hi[emos_dig == prob].mean() for prob in probs])    
        emos_hist = np.histogram(emos_prob, bins)[0] / len(val_tag)
        emos_hist = np.insert(emos_hist, [0,10], 0.0)

        armos1_prob = (armos1_samples[..., leadt] >= thresh).mean(0)
        armos1_dig  = probs[np.digitize(armos1_prob,bins[1:], right=True)]
        armos1_rel  = np.array([hi[armos1_dig == prob].mean() for prob in probs])    
        armos1_hist = np.histogram(armos1_prob, bins)[0] / len(val_tag)
        armos1_hist = np.insert(armos1_hist, [0,10], 0.0)

        armos2_prob = (armos2_samples[..., leadt] >= thresh).mean(0)
        armos2_dig  = probs[np.digitize(armos2_prob,bins[1:], right=True)]
        armos2_rel  = np.array([hi[armos2_dig == prob].mean() for prob in probs])    
        armos2_hist = np.histogram(armos2_prob, bins)[0] / len(val_tag)
        armos2_hist = np.insert(armos2_hist, [0,10], 0.0)

        armos3_prob = (armos3_samples[..., leadt] >= thresh).mean(0)
        armos3_dig  = probs[np.digitize(armos3_prob,bins[1:], right=True)]
        armos3_rel  = np.array([hi[armos3_dig == prob].mean() for prob in probs])    
        armos3_hist = np.histogram(armos3_prob, bins)[0] / len(val_tag)
        armos3_hist = np.insert(armos3_hist, [0,10], 0.0)

        bins = np.insert(bins, -1, 1.0)

        fig, axs = plt.subplots(2, 1, figsize=(5,8), gridspec_kw={'height_ratios': [3, 1]})

        axs[0].plot([0.0, 1.0], [0.0,1.0], 'k--', lw=1)
        axs[0].plot(probs, emos_rel, 'o-', label='EMOS')
        axs[0].plot(probs, armos1_rel, 'o-', label='ARMOS(1)')
        axs[0].plot(probs, armos2_rel, 'o-', label='ARMOS(2)')
        axs[0].plot(probs, armos3_rel, 'o-', label='ARMOS(3)')
        axs[0].set_xlim(0.0,1.0)
        axs[0].set_ylim(0.0,1.0)
        axs[0].set_aspect(1)
        axs[0].legend()
        axs[0].set_xlabel('Prob. forecast')
        axs[0].set_ylabel('Obs. freq.')

        axs[1].step(bins, emos_hist * 100, label='Emos')
        axs[1].step(bins, armos1_hist * 100, label='Armos1')
        axs[1].step(bins, armos2_hist * 100, label='Armos2')
        axs[1].step(bins, armos3_hist * 100, label='Armos3')
        axs[1].set_xlabel('Prob. forecast')
        axs[1].set_ylabel('Count (%)')
        plt.savefig(f'../res/FinalResults/GlobalReliabilityDiagram{leadt}h{thresh}ms.png')
        plt.show()

# PIT diagrams
for leadt in range(12,49,12):
    # Estimate F(X)
    print('Leadt:', leadt)
    x_emos = (emos_samples[...,leadt] <= val_obs[..., leadt].numpy()).mean(0)
    x_emos.sort()

    x_armos1 = (armos1_samples[...,leadt] <= val_obs[..., leadt].numpy()).mean(0)
    x_armos1.sort()

    x_armos2 = (armos2_samples[...,leadt] <= val_obs[..., leadt].numpy()).mean(0)
    x_armos2.sort()

    x_armos3 = (armos3_samples[...,leadt] <= val_obs[..., leadt].numpy()).mean(0)
    x_armos3.sort()

    y = np.linspace(0.0, 1.0, len(val_tag) + 1)[1:]
    plt.plot(x_emos, y, label='EMOS')
    plt.plot(x_armos1, y, label='ARMOS(1)')
    plt.plot(x_armos2, y, label='ARMOS(2)')
    plt.plot(x_armos3, y, label='ARMOS(3)')


    plt.plot([0.0,1.0],[0.0,1.0], 'k--', lw=1)
    plt.xlabel('Prob.')
    plt.ylabel('Obs. freq.')
    plt.xlim(0.0,1.0)
    plt.ylim(0.0,1.0)
    ax = plt.gca() 
    ax.set_aspect(1)
    plt.legend()
    plt.savefig(f'../res/FinalResults/GlobalPitDiagram{leadt}h.png')
    plt.show()
