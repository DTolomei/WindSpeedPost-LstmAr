import os
import json
import pickle as pkl
from tqdm import tqdm

nobackup    = '/net/pc160111/nobackup/users/teixeira/'
data_folder = nobackup +'refactorTest/'

# Remove dictionary entries with inconsistent data
# Read error list from errors.txt
with open('errors.json', 'r') as f:
    errors = json.load(f)
    
# Loop over dates and remove samples with errors from Pickle files
error_dates = list({error[0][:8] for error in errors})

for date in tqdm(error_dates):
    # Get pickle file
    file = data_folder + date + '.pkl'
    with open(file, 'rb') as f:
        data = pkl.load(f)
    # Remove dictionary keys with errors
    keys  = [key for key, _ in errors if date in key]
    for key in keys:
        data.pop(key)
    # Save error free file
    with open(file, 'wb') as f:
        pkl.dump(data,f)