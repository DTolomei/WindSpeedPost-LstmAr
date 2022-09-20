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
emos   = {st:EMOS(5, 49) for st in sts}
armos1 = {st:ARMOS(5, 1, 49) for st in sts}
armos2 = {st:ARMOS(5, 2, 49) for st in sts}
armos3 = {st:ARMOS(5, 3, 49) for st in sts}

# Load
with open('../res/models_final/LocalEmos.pkl', 'rb') as f:
    params = pkl.load(f)
for st, p in params.items():
    emos[st].a, emos[st].b, emos[st].c, emos[st].d = p['a'], p['b'], p['c'], p['d']
    
with open('../res/models_final/LocalArmos1.pkl', 'rb') as f:
    params = pkl.load(f)
for st, p in params.items():
    armos1[st].a, armos1[st].b, armos1[st].c, armos1[st].d, armos1[st].phi = p['a'], p['b'], p['c'], p['d'], p['phi']
    
with open('../res/models_final/LocalArmos2.pkl', 'rb') as f:
    params = pkl.load(f)
for st, p in params.items():
    armos2[st].a, armos2[st].b, armos2[st].c, armos2[st].d, armos2[st].phi = p['a'], p['b'], p['c'], p['d'], p['phi']
    
with open('../res/models_final/LocalArmos3.pkl', 'rb') as f:
    params = pkl.load(f)
for st, p in params.items():
    armos3[st].a, armos3[st].b, armos3[st].c, armos3[st].d, armos3[st].phi = p['a'], p['b'], p['c'], p['d'], p['phi']
    

# ---------------- MULTIVARIATE SCORES ----------------
emos_score   = {st:{'LS':[],'ES':[],'VS':[]} for st in sts}
armos1_score = {st:{'LS':[],'ES':[],'VS':[]} for st in sts}
armos2_score = {st:{'LS':[],'ES':[],'VS':[]} for st in sts}
armos3_score = {st:{'LS':[],'ES':[],'VS':[]} for st in sts}
armos1_score_ss = {st:{'LS':[],'ES':[],'VS':[]} for st in sts}
armos2_score_ss = {st:{'LS':[],'ES':[],'VS':[]} for st in sts}
armos3_score_ss = {st:{'LS':[],'ES':[],'VS':[]} for st in sts}

w = 0.5**np.array([[np.abs(i-j)-1 for i in range(49)] for j in range(49)], dtype='float32')

for st in tqdm(sts): 
    # Select data
    ind = [i for i, t in enumerate(val_tag.numpy().astype('str')) if '_' + st in t]
    loc_tag = tf.gather(val_tag, ind)
    loc_S2  = tf.gather(val_S2 , ind) 
    loc_var = tf.gather(val_var, ind)
    loc_obs = tf.gather(val_obs, ind)
    
    emos_samples   = emos[st].schaake_shuffle(loc_tag, loc_var, loc_S2, tag, obs, n_samples=30)
    armos1_samples = armos1[st].sample(loc_var, loc_S2, n_samples=30)
    armos2_samples = armos2[st].sample(loc_var, loc_S2, n_samples=30)
    armos3_samples = armos3[st].sample(loc_var, loc_S2, n_samples=30)
    armos1_samples_ss = schaake_shuffle(loc_tag, armos1_samples, tag, obs)
    armos2_samples_ss = schaake_shuffle(loc_tag, armos2_samples, tag, obs)
    armos3_samples_ss = schaake_shuffle(loc_tag, armos3_samples, tag, obs)

    emos_score[st]['LS'] = emos[st].loss(loc_var, loc_S2, loc_obs).numpy().sum(-1)
    emos_score[st]['ES'] = energy_score(emos_samples, loc_obs)
    emos_score[st]['VS'] = variogram_score(emos_samples, loc_obs, w=w)

    armos1_score[st]['LS'] = armos1[st].loss(loc_var, loc_S2, loc_obs).numpy().sum(-1)
    armos1_score[st]['ES'] = energy_score(armos1_samples, loc_obs)
    armos1_score[st]['VS'] = variogram_score(armos1_samples, loc_obs, w=w)

    armos2_score[st]['LS'] = armos2[st].loss(loc_var, loc_S2, loc_obs).numpy().sum(-1)
    armos2_score[st]['ES'] = energy_score(armos2_samples, loc_obs)
    armos2_score[st]['VS'] = variogram_score(armos2_samples, loc_obs, w=w)

    armos3_score[st]['LS'] = armos3[st].loss(loc_var, loc_S2, loc_obs).numpy().sum(-1)
    armos3_score[st]['ES'] = energy_score(armos3_samples, loc_obs)
    armos3_score[st]['VS'] = variogram_score(armos3_samples, loc_obs, w=w)

    armos1_score_ss[st]['LS'] = armos1[st].loss(loc_var, loc_S2, loc_obs).numpy().sum(-1)
    armos1_score_ss[st]['ES'] = energy_score(armos1_samples_ss, loc_obs)
    armos1_score_ss[st]['VS'] = variogram_score(armos1_samples_ss, loc_obs, w=w)

    armos2_score_ss[st]['LS'] = armos2[st].loss(loc_var, loc_S2, loc_obs).numpy().sum(-1)
    armos2_score_ss[st]['ES'] = energy_score(armos2_samples_ss, loc_obs)
    armos2_score_ss[st]['VS'] = variogram_score(armos2_samples_ss, loc_obs, w=w)

    armos3_score_ss[st]['LS'] = armos3[st].loss(loc_var, loc_S2, loc_obs).numpy().sum(-1)
    armos3_score_ss[st]['ES'] = energy_score(armos3_samples_ss, loc_obs)
    armos3_score_ss[st]['VS'] = variogram_score(armos3_samples_ss, loc_obs, w=w)

local_results = {
    'Emos':{
        'LS':np.concatenate([score['LS'] for score in emos_score.values()]).mean(),
        'ES':np.concatenate([score['ES'] for score in emos_score.values()]).mean(),
        'VS':np.concatenate([score['VS'] for score in emos_score.values()]).mean()
    },
    'Armos1':{
        'LS':np.concatenate([score['LS'] for score in armos1_score.values()]).mean(),
        'ES':np.concatenate([score['ES'] for score in armos1_score.values()]).mean(),
        'VS':np.concatenate([score['VS'] for score in armos1_score.values()]).mean()
    }, 
    'Armos2':{
        'LS':np.concatenate([score['LS'] for score in armos2_score.values()]).mean(),
        'ES':np.concatenate([score['ES'] for score in armos2_score.values()]).mean(),
        'VS':np.concatenate([score['VS'] for score in armos2_score.values()]).mean()
    }, 
    'Armos3':{
        'LS':np.concatenate([score['LS'] for score in armos3_score.values()]).mean(),
        'ES':np.concatenate([score['ES'] for score in armos3_score.values()]).mean(),
        'VS':np.concatenate([score['VS'] for score in armos3_score.values()]).mean()
    }, 
    'Armos1/SS':{
        'LS':np.concatenate([score['LS'] for score in armos1_score_ss.values()]).mean(),
        'ES':np.concatenate([score['ES'] for score in armos1_score_ss.values()]).mean(),
        'VS':np.concatenate([score['VS'] for score in armos1_score_ss.values()]).mean()
    },
    'Armos2/SS':{
        'LS':np.concatenate([score['LS'] for score in armos2_score_ss.values()]).mean(),
        'ES':np.concatenate([score['ES'] for score in armos2_score_ss.values()]).mean(),
        'VS':np.concatenate([score['VS'] for score in armos2_score_ss.values()]).mean()
    },
    'Armos3/SS':{
        'LS':np.concatenate([score['LS'] for score in armos3_score_ss.values()]).mean(),
        'ES':np.concatenate([score['ES'] for score in armos3_score_ss.values()]).mean(),
        'VS':np.concatenate([score['VS'] for score in armos3_score_ss.values()]).mean()
    }
}
with open('../res/FinalResults/LocalResults.pkl', 'wb') as f:
    pkl.dump(local_results,f)
    

# ---------------- UNIVARIATE VALIDATION ----------------

emos_samples = np.zeros((100, len(val_tag), 49))
armos1_samples = np.zeros((100, len(val_tag), 49))
armos2_samples = np.zeros((100, len(val_tag), 49))
armos3_samples = np.zeros((100, len(val_tag), 49))

for st in tqdm(sts):
    ind = [i for i, t in enumerate(val_tag.numpy().astype('str')) if '_' + st in t]
    loc_tag = tf.gather(val_tag, ind)
    loc_S2  = tf.gather(val_S2 , ind) 
    loc_var = tf.gather(val_var, ind)
    loc_obs = tf.gather(val_obs, ind)
    
    emos_samples[:,ind]   = emos[st].schaake_shuffle(loc_tag, loc_var, loc_S2, tag, obs, n_samples=100)
    armos1_samples[:,ind] = armos1[st].sample(loc_var, loc_S2, n_samples=100)
    armos2_samples[:,ind] = armos2[st].sample(loc_var, loc_S2, n_samples=100)
    armos3_samples[:,ind] = armos3[st].sample(loc_var, loc_S2, n_samples=100)

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
    plt.plot(thresh, 1 - global_emos_ref[leadt]  /clim_bs, 'k', label='Global EMOS')
    plt.plot(thresh, 1 - emos_bs  /clim_bs, label='Local EMOS')
    plt.plot(thresh, 1 - armos1_bs/clim_bs, label='ARMOS(1)')
    plt.plot(thresh, 1 - armos2_bs/clim_bs, label='ARMOS(2)')
#     plt.plot(thresh, 1 - armos3_bs/clim_bs, label='ARMOS(3)')
    plt.xlim([1,20])
    plt.legend()
    plt.ylabel('BSS')
    plt.xlabel('Windspeed threshold (m/s)')
    ax = plt.gca()
    ax.figure.set_size_inches(5, 3)
    plt.savefig(f'../res/FinalResults/LocalBrierSkillScore{leadt}h.png')
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
        # axs[0].set_title(f'Reliability diagram for {leadt}h at  \n{thresh} m/s threshold')

        axs[1].step(bins, emos_hist * 100, label='EMOS')
        axs[1].step(bins, armos1_hist* 100, label='ARMOS(1)')
        axs[1].step(bins, armos2_hist* 100, label='ARMOS(2)')
        axs[1].step(bins, armos3_hist* 100, label='ARMOS(3)')
        axs[1].set_xlabel('Prob. forecast')
        axs[1].set_ylabel('Count (%)')
        # axs[1].legend()
        
        plt.savefig(f'../res/FinalResults/LocalReliabilityDiagram{leadt}h{thresh}ms.png')
        plt.show()
        

# PIT diagrams
for leadt in [12, 24, 36,48]:
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
#     plt.title(f'Local PIT {leadt}h')
    plt.savefig(f'../res/FinalResults/LocalPitDiagram{leadt}h.png')
    plt.show()