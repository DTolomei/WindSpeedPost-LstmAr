import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from random import sample

folder = '/net/pc160111/nobackup/users/teixeira/data/'

# Open st_info
with open(folder + '../st-info.pkl', 'rb') as file:
    st_info = pkl.load(file)
files = os.listdir(folder)

# Select files randomly
files = sample(files, 6)

# Load data
data  = []
for file in files:
    with open(folder + file, 'rb') as f:
        data.append(pkl.load(f))

# Get forecast error series
station = '348'
loc     = st_info[station]['GRID']

obs_nwp = {}
for d in data:
    key = [k for k in d.keys() if '_' + station in k][0]
    obs_nwp[key] = (d[key]['OBS'], d['GRID'][:, loc[0], loc[1]])

# Plot predict and observations series
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for (key, (obs, nwp)), ax in zip(obs_nwp.items(), axs.flatten()): 
    date, code = key.split('_') 
    name = st_info[code]['NAME']
    ax.plot(nwp, 'r--', label='NWP forecast')
    ax.plot(obs, 'g'  , label='Observation')
    ax.set_title(f'{name}  {code}, {date}')
    ax.set_xlabel('Leadt (h)')
    ax.set_ylabel('Speed (m/s)')
    ax.legend()

plt.savefig('plot_10d_348.png')
plt.show()
