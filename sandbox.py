
import pandas as pd
import numpy as np
from sklearn.manifold import spectral_embedding

from collections import Counter

import matplotlib.pyplot as plt

import utils
reload(utils)

out_dir = '../plots/'

xmin = 0
xmax = 6 * 60 # minutes

df = pd.read_csv('./data/autotaggedperms.csv')
df = df.loc[~df['datetime'].isnull()]

times = utils.get_times(df)
rumor_df = utils.get_rumor_features(df, [])

collapsed,sorted_hypotheses = utils.get_collapsed(rumor_df)

c = Counter(collapsed)

print rumor_df.columns
for i in c.most_common():
    print i

x = np.sum(rumor_df, 0)
y = np.sum(np.array(map(lambda x: x.split('-'), set(collapsed)), dtype = int), 0)

plt.scatter(x, y)
for i in range(len(x)):
    plt.text(x[i],y[i],rumor_df.columns[i], size = 7)
plt.xlim(-0.1*max(x),1.1*max(x))
plt.ylim(-0.1*max(y),1.1*max(y))

cooc = np.dot(np.transpose(rumor_df), rumor_df)
print rumor_df.columns
print cooc

collapsed_short = [collapsed[i][:7] for i in range(len(collapsed))]

d = Counter(collapsed_short)

print rumor_df.columns
for i in d.most_common():
    print i

np.sum(rumor_df, 0)
