from src.models.losses import EmosLogScore, EmosEnergyScore, EmosVariogramScore
from src.models.losses import Armos1LogScore, Armos2LogScore, Armos3LogScore, ArmosL2
from src.models.validation import EmosSample, ArmosSample
from src.models.validation import energy_score, variogram_score
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions
import pickle as pkl
import matplotlib.pyplot as plt

# --------------- LOAD NET OUTPUTS ---------------
with open('../res/pred/NetTags.pkl', 'rb') as f:
    val_tag = pkl.load(f)
with open('../res/pred/NetObs.pkl', 'rb') as f:
    val_obs = pkl.load(f)
with open('../res/pred/EmosNetPred.pkl', 'rb') as f:
    emosnet_pred = pkl.load(f)
with open('../res/pred/LstmEmosNetPred.pkl', 'rb') as f:
    lstmemosnet_pred = pkl.load(f)
with open('../res/pred/Armos1NetPred.pkl', 'rb') as f:
    armos1net_pred = pkl.load(f)
with open('../res/pred/Armos2NetPred.pkl', 'rb') as f:
    armos2net_pred = pkl.load(f)
with open('../res/pred/Armos3NetPred.pkl', 'rb') as f:
    armos3net_pred = pkl.load(f)
    
loc, scale = emosnet_pred[...,0], emosnet_pred[...,1]
emosnet_samples = EmosSample(loc, scale, n_samples=100)
# emosnet_samples = schaake_shuffle(val_tag, emosnet_samples, tag, obs)

loc, scale = lstmemosnet_pred[...,0], lstmemosnet_pred[...,1]
lstmemosnet_samples = EmosSample(loc, scale, n_samples=100)
# lstmemosnet_samples = schaake_shuffle(val_tag, lstmemosnet_samples, tag, obs)

loc, scale, phi = armos1net_pred[...,:49], armos1net_pred[...,49:-1], armos1net_pred[-1:]
armos1net_samples = ArmosSample(loc, scale, phi, n_samples=100)
# armos1net_samples_ss = schaake_shuffle(val_tag, armos1net_samples, tag, obs)

loc, scale, phi = armos2net_pred[...,:49], armos2net_pred[...,49:-2], armos2net_pred[-2:]
armos2net_samples = ArmosSample(loc, scale, phi, n_samples=100)
# armos2net_samples_ss = schaake_shuffle(val_tag, armos2net_samples, tag, obs)

loc, scale, phi = armos3net_pred[...,:49], armos3net_pred[...,49:-3], armos3net_pred[-3:]
armos3net_samples = ArmosSample(loc, scale, phi, n_samples=100)
# armos3net_samples_ss = schaake_shuffle(val_tag, armos3net_samples, tag, obs)


# --------------- MULTIVARIATE SCORES ---------------
net_results = {
    'EmosNet':{
        'LS':EmosLS(val_obs, emosnet_pred).numpy().mean() * 49,
        'ES':energy_score(emosnet_samples[:30], val_obs).mean(),
        'VS':variogram_score(emosnet_samples[:30], val_obs).mean()
    },
    'LstmEmosNet':{
        'LS':EmosLS(val_obs, lstmemosnet_pred).numpy().mean() * 49,
        'ES':energy_score(lstmemosnet_samples[:30], val_obs).mean(),
        'VS':variogram_score(lstmemosnet_samples[:30], val_obs).mean()
    },
    'Armos1Net':{
        'LS':Armos1LS(val_obs, armos1net_pred).numpy().mean() * 49,
        'ES':energy_score(armos1net_samples[:30], val_obs).mean(),
        'VS':variogram_score(armos1net_samples[:30], val_obs).mean()
    },
    'Armos2Net':{
        'LS':Armos2LS(val_obs, armos2net_pred).numpy().mean() * 49,
        'ES':energy_score(armos2net_samples[:30], val_obs).mean(),
        'VS':variogram_score(armos2net_samples[:30], val_obs).mean()
    },
    'Armos3Net':{
        'LS':Armos3LS(val_obs, armos3net_pred).numpy().mean() * 49,
        'ES':energy_score(armos3net_samples[:30], val_obs).mean(),
        'VS':variogram_score(armos3net_samples[:30], val_obs).mean()
    },
    'Armos1NetSS':{
        'LS':Armos1LS(val_obs, armos1net_pred).numpy().mean() * 49,
        'ES':energy_score(armos1net_samples_ss[:30], val_obs).mean(),
        'VS':variogram_score(armos1net_samples_ss[:30], val_obs).mean()
    },
    'Armos2NetSS':{
        'LS':Armos2LS(val_obs, armos2net_pred).numpy().mean() * 49,
        'ES':energy_score(armos2net_samples_ss[:30], val_obs).mean(),
        'VS':variogram_score(armos2net_samples_ss[:30], val_obs).mean()
    },
    'Armos3NetSS':{
        'LS':Armos3LS(val_obs, armos3net_pred).numpy().mean() * 49,
        'ES':energy_score(armos3net_samples_ss[:30], val_obs).mean(),
        'VS':variogram_score(armos3net_samples_ss[:30], val_obs).mean()
    }   
}

with open('../res/FinalResults/NetResults.pkl', 'wb') as f:
    pkl.dump(net_results,f)
    

# --------------- UNIVARIATE VALIDATION ---------------

# BRIER SKILL SCORE  
thresh = np.arange(1.0, 21.0, 1.0)

for leadt in [12,24,36,48]:
    print('Leadt:', leadt)
    hi = (val_obs[..., None, leadt] >= thresh).numpy()

    clim_bs = climatology[leadt, :len(thresh)]
    clim_bs = (clim_bs - hi)**2
    clim_bs = clim_bs.mean(0)

    emosnet_bs = (emosnet_samples[..., None, leadt] >= thresh).mean(0)
    emosnet_bs = (emosnet_bs - hi)**2
    emosnet_bs =  emosnet_bs.mean(0)

    lstmemosnet_bs = (lstmemosnet_samples[..., None, leadt] >= thresh).mean(0)
    lstmemosnet_bs = (lstmemosnet_bs - hi)**2
    lstmemosnet_bs =  lstmemosnet_bs.mean(0)

    armos1net_bs = (armos1net_samples[..., None, leadt] >= thresh).mean(0)
    armos1net_bs = (armos1net_bs - hi)**2
    armos1net_bs =  armos1net_bs.mean(0)

    armos2net_bs = (armos2net_samples[..., None, leadt] >= thresh).mean(0)
    armos2net_bs = (armos2net_bs - hi)**2
    armos2net_bs =  armos2net_bs.mean(0)

    armos3net_bs = (armos3net_samples[..., None, leadt] >= thresh).mean(0)
    armos3net_bs = (armos3net_bs - hi)**2
    armos3net_bs =  armos3net_bs.mean(0)


    plt.hlines(0.0,1,20, lw=1, ls='dashed', color='k', label='Climat.')
    plt.plot(thresh, 1 - global_emos_ref[leadt]/clim_bs, 'k', label='Global EMOS')
    plt.plot(thresh, 1 - emosnet_bs/clim_bs, label='EMOSNet')
    plt.plot(thresh, 1 - lstmemosnet_bs/clim_bs, 'purple', label='LSTM/EMOSNet')
    plt.plot(thresh, 1 - armos1net_bs/clim_bs, label='ARMOS(1)net')
    plt.plot(thresh, 1 - armos2net_bs/clim_bs, label='ARMOS(2)net')
    plt.plot(thresh, 1 - armos3net_bs/clim_bs, label='ARMOS(3)net')
    plt.legend()
    plt.xlim([1,20])
    #     plt.title(f'BSS relative EmosNet {leadt}h')
    plt.ylabel('BSS')
    plt.xlabel('Windspeed threshold (m/s)')
    ax = plt.gca()
    ax.figure.set_size_inches(5, 3)
    plt.savefig(f'../res/FinalResults/NetBrierSkillScore{leadt}h.png')
    plt.show()


# RELIABILITY DIAGRAMS

for leadt in [12,24,36,48]:
    for thresh in [15.0]:
        print('Leadt:', leadt,', Thresh:',thresh)
        bins = np.arange(0.0, 1.1, 0.1)
        probs = np.arange(0.05, 1.0, 0.1)

        hi = (val_obs[..., leadt] >= thresh).numpy()

        emosnet_prob = (emosnet_samples[..., leadt] >= thresh).mean(0)
        emosnet_dig  = probs[np.digitize(emosnet_prob,bins[1:], right=True)]
        emosnet_rel  = np.array([hi[emosnet_dig == prob].mean() for prob in probs])    
        emosnet_hist = np.histogram(emosnet_prob, bins)[0] / len(val_tag)
        emosnet_hist = np.insert(emosnet_hist, [0,10], 0.0)

        lstmemosnet_prob = (lstmemosnet_samples[..., leadt] >= thresh).mean(0)
        lstmemosnet_dig  = probs[np.digitize(lstmemosnet_prob,bins[1:], right=True)]
        lstmemosnet_rel  = np.array([hi[lstmemosnet_dig == prob].mean() for prob in probs])    
        lstmemosnet_hist = np.histogram(lstmemosnet_prob, bins)[0] / len(val_tag)
        lstmemosnet_hist = np.insert(lstmemosnet_hist, [0,10], 0.0)

        armos1net_prob = (armos1net_samples[..., leadt] >= thresh).mean(0)
        armos1net_dig  = probs[np.digitize(armos1net_prob,bins[1:], right=True)]
        armos1net_rel  = np.array([hi[armos1net_dig == prob].mean() for prob in probs])    
        armos1net_hist = np.histogram(armos1net_prob, bins)[0] / len(val_tag)
        armos1net_hist = np.insert(armos1net_hist, [0,10], 0.0)

        armos2net_prob = (armos2net_samples[..., leadt] >= thresh).mean(0)
        armos2net_dig  = probs[np.digitize(armos2net_prob,bins[1:], right=True)]
        armos2net_rel  = np.array([hi[armos2net_dig == prob].mean() for prob in probs])    
        armos2net_hist = np.histogram(armos2net_prob, bins)[0] / len(val_tag)
        armos2net_hist = np.insert(armos2net_hist, [0,10], 0.0)

        armos3net_prob = (armos3net_samples[..., leadt] >= thresh).mean(0)
        armos3net_dig  = probs[np.digitize(armos3net_prob,bins[1:], right=True)]
        armos3net_rel  = np.array([hi[armos3net_dig == prob].mean() for prob in probs])    
        armos3net_hist = np.histogram(armos3net_prob, bins)[0] / len(val_tag)
        armos3net_hist = np.insert(armos3net_hist, [0,10], 0.0)

        bins = np.insert(bins, -1, 1.0)

        fig, axs = plt.subplots(2, 1, figsize=(5,8), gridspec_kw={'height_ratios': [3, 1]})

        axs[0].plot([0.0, 1.0], [0.0,1.0], 'k--', lw=1)
        axs[0].plot(probs, emosnet_rel, 'o-', label='EMOSnet')
        axs[0].plot(probs, lstmemosnet_rel, 'o-', c='purple', label='LSTM/EMOSnet')
        axs[0].plot(probs, armos1net_rel, 'o-', label='ARMOS(1)net')
        axs[0].plot(probs, armos2net_rel, 'o-', label='ARMOS(2)net')
        axs[0].plot(probs, armos3net_rel, 'o-', label='ARMOS(3)net')
        axs[0].set_xlim(0.0,1.0)
        axs[0].set_ylim(0.0,1.0)
        axs[0].set_aspect(1)
        axs[0].legend()
        axs[0].set_xlabel('Prob. forecast')
        axs[0].set_ylabel('Obs. freq.')
        # axs[0].set_title(f'Reliability diagram for {leadt}h at  \n{thresh} m/s threshold')

        axs[1].step(bins, emosnet_hist * 100, label='EMOSnet')
        axs[1].step(bins, lstmemosnet_hist* 100, 'purple', label='LSTM/EMOSnet')
        axs[1].step(bins, armos1net_hist* 100, label='ARMOS(1)net')
        axs[1].step(bins, armos2net_hist* 100, label='ARMOS(2)net')
        axs[1].step(bins, armos3net_hist* 100, label='ARMOS(3)net')
        axs[1].set_xlabel('Prob. forecast')
        axs[1].set_ylabel('Count (%)')
        # axs[1].legend()
        plt.savefig(f'../res/FinalResults/NetReliabilityDiagram{leadt}h{thresh}ms.png')
        plt.show()
        

# PIT diagrams
for leadt in [12,24,36,48]:
    print('Leadt:', leadt)
    x_emosnet = (emosnet_samples[...,leadt] <= val_obs[..., leadt].numpy()).mean(0)
    x_emosnet.sort()

    x_lstmemosnet = (lstmemosnet_samples[...,leadt] <= val_obs[..., leadt].numpy()).mean(0)
    x_lstmemosnet.sort()

    x_armos1net = (armos1net_samples[...,leadt] <= val_obs[..., leadt].numpy()).mean(0)
    x_armos1net.sort()

    x_armos2net = (armos2net_samples[...,leadt] <= val_obs[..., leadt].numpy()).mean(0)
    x_armos2net.sort()

    x_armos3net = (armos3net_samples[...,leadt] <= val_obs[..., leadt].numpy()).mean(0)
    x_armos3net.sort()

    y = np.linspace(0.0, 1.0, len(val_tag) + 1)[1:]
    plt.plot(x_emosnet, y, label='EMOSnet')
    plt.plot(x_lstmemosnet, y, c='purple', label='LSTM/EMOSnet')
    plt.plot(x_armos1net, y, label='ARMOS(1)net', lw=1)
    plt.plot(x_armos2net, y, label='ARMOS(2)net')
    plt.plot(x_armos3net, y, label='ARMOS(3)net')


    plt.plot([0.0,1.0],[0.0,1.0], 'k--', lw=1)
    plt.xlabel('Prob.')
    plt.ylabel('Obs. freq.')
    plt.xlim(0.0,1.0)
    plt.ylim(0.0,1.0)
    ax = plt.gca() 
    ax.set_aspect(1)
    plt.legend()
#     plt.title(f'Global PIT {leadt}h')
    plt.savefig(f'../res/FinalResults/NetPitDiagram{leadt}h.png')
    plt.show()
