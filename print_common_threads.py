
import pandas as pd
import numpy as np
from sklearn.manifold import spectral_embedding

from collections import Counter

import matplotlib.pyplot as plt

import utils
reload(utils)

out_dir = '../plots/'

df = pd.read_csv('./data/autotaggedperms.csv')
df = df.loc[~df['datetime'].isnull()]

times = utils.get_times(df)
rumor_df = utils.get_rumor_features(df, [])

collapsed,sorted_hypotheses = utils.get_collapsed(rumor_df)

c = Counter(collapsed)

print
for i in c.most_common():
    print i
