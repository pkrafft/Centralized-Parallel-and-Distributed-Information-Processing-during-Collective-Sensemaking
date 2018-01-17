
import pandas as pd
import numpy as np
from sklearn.manifold import spectral_embedding

import matplotlib.pyplot as plt

import utils
reload(utils)

out_dir = '../plots/'

xmin = 0
xmax = 6 * 60 # minutes

df = pd.read_csv('./data/autotaggedperms.csv')
df = df.loc[~df['datetime'].isnull()]

times = utils.get_times(df)
rumor_df = utils.get_rumor_features(df, ['rip','young','sandy','phony','boy'])

hour = 4
inds = (times < 60 * 60 * (hour + 1)) & (times > 60 * 60 * hour)


sub = rumor_df.loc[inds]

cooc = np.dot(np.transpose(sub), sub)
degree = np.diag(cooc)

eigs = spectral_embedding(np.array(cooc > 0,dtype=int), n_components = 2)
eigs += np.random.normal(scale = 0.2*np.std(eigs), size = eigs.shape)

for i in range(len(cooc)-1):
    for j in range(i+1,len(cooc)):
        if cooc[i,j] > 0:
            plt.plot([eigs[i,0],eigs[j,0]],[eigs[i,1],eigs[j,1]], 'k-')

for i,tag in enumerate(rumor_df.columns):
    if degree[i] > 0:
        print degree[i], tag
        plt.text(eigs[i,0],eigs[i,1],tag,color = 'blue')
        
plt.xlim(np.min(eigs[:,0]), np.max(eigs[:,0]))
plt.ylim(np.min(eigs[:,1]), np.max(eigs[:,1]))




norm = np.transpose(np.array([np.diag(cooc) for i in range(len(cooc))], dtype = float))

ncooc = cooc / norm
