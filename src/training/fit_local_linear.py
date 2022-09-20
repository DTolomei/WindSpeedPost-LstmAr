from src.models.SchaakeEMOS import SchaakeEMOS as EMOS
from src.models.ARMOS import ARMOS
from src.data.dataset import gen_dataset, load_to_memory
from tqdm import tqdm

emos   = {st:EMOS(5, 49) for st in sts}
armos1 = {st:ARMOS(5, 1, 49) for st in sts}
armos2 = {st:ARMOS(5, 2, 49) for st in sts}
armos3 = {st:ARMOS(5, 3, 49) for st in sts}

train_dataset  = gen_dataset(train_files, sts=sts, var=var_list, grid_size=5)
tag, S2, var, obs = load_to_memory(train_dataset, grid_size=5, grid_S2=True)

for st in tqdm(sts):
    print('Fitting',st)
    
    # Select data
    ind = [i for i, t in enumerate(tag.numpy().astype('str')) if '_' + st in t]
    loc_tag = tf.gather(tag, ind)
    loc_S2  = tf.gather(S2 , ind) 
    loc_var = tf.gather(var, ind)
    loc_obs = tf.gather(obs, ind)
    
    # Fit local models
    emos[st].a, emos[st].b, emos[st].c, emos[st].d = tf.Variable(a), tf.Variable(b), tf.Variable(c), tf.Variable(d)
    h = emos[st].fit(loc_var, loc_S2, loc_obs, steps=2500)
    h = armos1[st].fit(loc_var, loc_S2, loc_obs, steps=2500)
    h = armos2[st].fit(loc_var, loc_S2, loc_obs, steps=2500)
    h = armos3[st].fit(loc_var, loc_S2, loc_obs, steps=2500)

# Save
emos_params = {
    st:{
        'T':49, 'vars':var_list, 'a':model.a, 'b':model.b, 'c':model.c, 'd':model.d
    } for st, model in emos.items()
}
with open('../../res/models_final/LocalEmos.pkl', 'wb') as f:
    pkl.dump(emos_params,f)
    print('LocalEmos saved')  
    
armos1_params = {
    st:{
        'T':49, 'vars':var_list, 'p':1, 'a':model.a, 'b':model.b, 'c':model.c, 'd':model.d, 'phi':model.phi
    } for st, model in armos1.items()
}
with open('../../res/models_final/LocalArmos1.pkl', 'wb') as f:
    pkl.dump(armos1_params,f)
    print('LocalArmos1 saved')
    
armos2_params = {
    st:{
        'T':49, 'vars':var_list, 'p':2, 'a':model.a, 'b':model.b, 'c':model.c, 'd':model.d, 'phi':model.phi
       } for st, model in armos2.items()
}
with open('../../res/models_final/LocalArmos2.pkl', 'wb') as f:
    pkl.dump(armos2_params,f)
    print('LocalArmos2 saved')
    
armos3_params = {
    st:{
        'T':49, 'vars':var_list, 'p':3, 'a':model.a, 'b':model.b, 'c':model.c, 'd':model.d, 'phi':model.phi
       } for st, model in armos3.items()
}
with open('../../res/models_final/LocalArmos3.pkl', 'wb') as f:
    pkl.dump(armos3_params,f)
    print('LocalArmos3 saved')