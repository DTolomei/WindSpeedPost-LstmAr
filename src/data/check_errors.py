import os
import json
import numpy as np
import pickle as pkl
from tqdm import tqdm

# Get Pickle files, station info and initialize error list
nobackup    = '/net/pc160111/nobackup/users/teixeira/'
data_folder = nobackup + 'refactorTest/' 
pkl_files   = [data_folder + file for file in sorted(os.listdir(data_folder))]
with open(nobackup + 'st-info.pkl', 'rb') as file:
    st_info = pkl.load(file)
errors      = []


# Check data consistency for all files, all values should be non negative
for pkl_file in tqdm(pkl_files):
    
    # Load .pkl file
    with open(pkl_file, 'rb') as f:
        data = pkl.load(f)
        
    # Get wind speed 10m grid data
    # This way it won't get stuck in the loop below
    grid = data.pop('GRID')
    
    # Check local data for all stations
    for stationTag, locData in data.items():
        date, code = stationTag.split('_')
        # Check for correct location
        if not st_info[code]['GRID'] == locData.pop('LOC'):
            print(f'ERROR: {stationTag} has wrong location.')
            errors.append((stationTag, 'LOC'))
        # Check for valid data for all variables
        for var, values in locData.items():
            if not np.all(values >= 0):
                # ROUGH is a problem variable that was later removed
                if not var == 'ROUGH':
                    print(f'ERROR: {stationTag}.{var} has invalid data.')
                    errors.append((stationTag, var))

# Save error list
with open('errors.json', 'w') as f:
    json.dump(errors, f)