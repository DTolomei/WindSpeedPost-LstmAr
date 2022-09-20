import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from src.data.dataset import gen_dataset, load_to_memory

st_select = ['248', '258', '278', '290', '348']

# --------------------- LOAD DATA ---------------------
nobackup = '/net/pc160111/nobackup/users/teixeira/'

with open(nobackup + 'st-info.pkl', 'rb') as file:
    info = pkl.load(file)
    
data_folder = nobackup + 'data/'
files   = sorted([data_folder + file for file in os.listdir(data_folder)]) 
omit    = ['229', '285', '323']
sts     = [st for st in info.keys() if st not in omit]
dataset = gen_dataset(files, sts=sts, var=['WIND850'], grid_size=1, shuffle=False)
tag, nwp, _, obs = load_to_memory(dataset, 1)

# Get residues (forecast - observation) and normalized residues
tag, nwp, obs = tag.numpy().astype('str'), nwp[...,1,1].numpy(), obs.numpy()
res = nwp - obs
mean, std = res.mean(0), res.std(0)
norm_res  = (res - mean) / std

# --------------------- STATISTICS PER LEAD TIME ---------------------
fig, axs = plt.subplots(2, 3, figsize=(12, 10))

for i, ax in enumerate(axs.flatten()[:-1]):
    st = st_select[i]
    name = info[st]['NAME']
    ind = np.array([i for i, s in enumerate(tag) if '_' + st in s], dtype='int16')
    mean, std = res[ind].mean(0), res[ind].std(0)
    ax.plot(mean,'r')
    ax.plot(std, 'b')
    ax.set_title(name + ' ' + st)
    ax.set_xlabel('Lead time (h)')
    ax.set_ylabel('Wind speed (m/s)')
    ax.set_ylim([0.5,3.0])
    ax.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
    ax.set_xlim([0, 48])
mean, std = res.mean(0), res.std(0)
axs[1,2].plot(mean,'r')
axs[1,2].plot(std, 'b')
axs[1,2].set_title('GLOBAL')
axs[1,2].set_xlabel('Lead time (h)')
axs[1,2].set_ylabel('Wind speed (m/s)')
axs[1,2].set_ylim([0.5,3.0])
axs[1,2].set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
axs[1,2].set_xlim([0, 48])

plt.savefig('stat.png')
plt.show()

# --------------------- CORRELATION MATRIX ---------------------
# Get correlation matrix
mat = norm_res.T @ norm_res / norm_res.shape[0]

# Plot matrix
fig, ax = plt.subplots(1, figsize=(5,5))

cax = ax.matshow(mat, cmap='RdYlGn', vmin=-1, vmax=1)
ax.set_title('CORRELATION MATRIX')
ax.set_xlabel('j')
ax.set_ylabel('i')
ax.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
ax.set_yticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
ax.xaxis.set_label_position('top')
fig.colorbar(cax)

plt.savefig('corr_matrix.png')
plt.show()

# --------------------- TEMPORAL AUTOCORRELATION ---------------------
fig, axs = plt.subplots(2, 3, figsize=(12, 10))

for i, ax in enumerate(axs.flatten()[:-1]):
    st = st_select[i]
    name = info[st]['NAME']
    ind = np.array([i for i, s in enumerate(tag) if '_' + st in s], dtype='int16')
    mean, std = res[ind].mean(0), res[ind].std(0)
    corr = np.zeros(49)
    corr[0]=1.0
    for i in range(1,49):
        corr[i] = (norm_res[...,i:] * norm_res[...,:-i]).mean()
    ax.plot(corr,'r')
    ax.set_title(name + ' ' + st)
    ax.set_xlabel('Lag (h)')
    ax.set_ylabel('Autocorrelation')
    ax.set_ylim([0.0,1.0])
    ax.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
    ax.set_xlim([0, 48])
mean, std = res.mean(0), res.std(0)
norm_res = (res - mean) / std
mean, std = norm_res.mean(0), norm_res.std(0)
norm_res = (norm_res - mean) / std
corr = np.zeros(49)
corr[0]=1.0
for i in range(1,49):
    corr[i] = (norm_res[...,i:] * norm_res[...,:-i]).mean()
axs[-1,-1].plot(corr,'r')
axs[-1,-1].set_title('GLOBAL')
axs[-1,-1].set_ylim([0.0,1.0])
axs[-1,-1].set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
axs[-1,-1].set_xlim([0, 48])

plt.savefig('corr.png')
plt.show()