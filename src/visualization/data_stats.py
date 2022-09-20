import os
import pickle as pkl
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load st_info and get files
nobackup = '/net/pc160111/nobackup/users/teixeira/'
data_folder = nobackup + 'data/'

with open(nobackup + 'st-info.pkl', 'rb') as f:
    st_info = pkl.load(f)

pkl_files = sorted(os.listdir(data_folder))

# Get stats per trimester
trims = sorted(list({file[:5] for file in os.listdir(data_folder)}))

res = {}
for trim in tqdm(trims, leave=True):
    trim_files = [file for file in pkl_files if trim in file]
    data = {}
    for file in trim_files:
        date = file[:8]
        with open(data_folder + file, 'rb') as f:
            data[date] = pkl.load(f)
            
        st_stats = {code: {'COUNT':0, 'MEAN':0, 'STD':0} for code in st_info.keys()}
        for date, samples in data.items():
            grid = samples['GRID']
            for tag, sample in samples.items():
                if not tag == 'GRID':
                    code = tag[-3:]
                    st_stats[code]['COUNT'] += 1
                    st_stats[code]['MEAN'] += grid[:,sample['LOC']].mean()
                    st_stats[code]['STD'] += grid[:,sample['LOC']].var()

        for stat in st_stats.values():
            if stat['COUNT'] > 0:
                stat['MEAN'] /= stat['COUNT']
                stat['STD'] = np.sqrt(stat['STD'] / stat['COUNT'])
        
        res[trim] = st_stats
        
trims = sorted(list(res.keys()))

counts, means, stds = [], [], []
for code in st_info.keys():
    counts.append(list(map(lambda trim: res[trim][code]['COUNT'], trims)))
    means.append(list(map(lambda trim: res[trim][code]['MEAN'], trims)))
    stds.append(list(map(lambda trim: res[trim][code]['STD'], trims)))


counts = np.array(counts)
means = np.array(means)
stds = np.array(stds)

# Plot stats
fig, axs = plt.subplots(1,3, figsize=(12,20))

axs[0].imshow(counts, cmap='gnuplot')
axs[1].imshow(means , cmap='gnuplot')
axs[2].imshow(stds  , cmap='gnuplot')

for ax in axs:
    ax.set_xticks(np.arange(len(trims)))
    ax.set_yticks(np.arange(len(st_info.keys())))
    ax.set_xticklabels(trims)
    ax.set_yticklabels(st_info.keys())
    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), rotation=45, va="baseline", ha="left",
             rotation_mode="default")

# Loop over data dimensions and create text annotations.
for i in range(len(st_info.keys())):
    for j in range(len(trims)):
        axs[0].text(j, i, counts[i, j],
                   ha="center", va="center", color="k")
        axs[1].text(j, i, f'{means[i, j]:.2f}',
                   ha="center", va="center", color="k")
        axs[2].text(j, i, f'{stds[i, j]:.2f}',
                   ha="center", va="center", color="k")
        

axs[0].set_title("Number of samples (station x trimester)")
axs[1].set_title("Mean windspeed (station x trimester)")
axs[2].set_title("Std. windspeed (station x trimester)")
fig.tight_layout()
fig.savefig("data_stats.png")
plt.show()
