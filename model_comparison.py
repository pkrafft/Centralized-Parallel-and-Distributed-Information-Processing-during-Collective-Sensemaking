# TODO: if a feature is discovered in a higher order variant, show dependence from higher order
import pandas as pd
import numpy as np

import sys

import matplotlib.pyplot as plt

import utils
reload(utils)

out_dir = '../plots/'

df = pd.read_csv('./data/autotaggedperms.csv')
df = df.loc[~df['datetime'].isnull()]

times = utils.get_times(df)
rumor_df = utils.get_rumor_features(df)


