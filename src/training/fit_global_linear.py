from src.models.SchaakeEMOS import SchaakeEMOS as EMOS
from src.models.ARMOS import ARMOS
from src.data.dataset import gen_dataset, load_to_memory

# Load data
train_dataset  = gen_dataset(train_files, sts=sts, var=var_list, grid_size=5)
tag, S2, var, obs = load_to_memory(train_dataset, grid_size=5, grid_S2=True)

# Init models
emos = EMOS(5, 49)
armos1 = ARMOS(5, 1, 49)
armos2 = ARMOS(5, 2, 49)
armos3 = ARMOS(5, 3, 49)

# Fit
h = emos.fit(var, S2, obs, steps=2500)
h = armos1.fit(var, S2, obs, steps=2500)
h = armos2.fit(var, S2, obs, steps=2500)
h = armos3.fit(var, S2, obs, steps=2500)

# Save
emos_params = {'T':49, 'vars':var_list, 'a':emos.a, 'b':emos.b, 'c':emos.c, 'd':emos.d}
with open('../../res/models_final/GlobalEmos.pkl', 'wb') as f:
    pkl.dump(emos_params,f)
    print('GlobalEmos saved')    
armos1_params = {'T':49, 'vars':var_list, 'p':1, 'a':armos1.a, 'b':armos1.b, 'c':armos1.c, 'd':armos1.d, 'phi':armos1.phi}
with open('../../res/models_final/GlobalArmos1.pkl', 'wb') as f:
    pkl.dump(armos1_params,f)
    print('GlobalArmos1 saved')
armos2_params = {'T':49, 'vars':var_list, 'p':2, 'a':armos2.a, 'b':armos2.b, 'c':armos2.c, 'd':armos2.d, 'phi':armos2.phi}
with open('../../res/models_final/GlobalArmos2.pkl', 'wb') as f:
    pkl.dump(armos2_params,f)
    print('GlobalArmos2 saved')
armos3_params = {'T':49, 'vars':var_list, 'p':3, 'a':armos3.a, 'b':armos3.b, 'c':armos3.c, 'd':armos3.d, 'phi':armos3.phi}
with open('../../res/models_final/GlobalArmos3.pkl', 'wb') as f:
    pkl.dump(armos3_params,f)
    print('GlobalArmos3 saved')