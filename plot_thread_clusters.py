
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

import sys

import matplotlib.pyplot as plt

import utils
reload(utils)

out_dir = '../plots/'

xmin = 0
xmax = 6 * 60 # minutes

df = pd.read_csv('./data/autotaggedperms.csv')
df = df.loc[~df['datetime'].isnull()]

times = utils.get_times(df)
rumor_df = utils.get_rumor_features(df)

kmeans = KMeans(n_clusters = 10)
fit = kmeans.fit(rumor_df)

for i in np.round(fit.cluster_centers_):
    rumor_df.columns[i == 1]

