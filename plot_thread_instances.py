# TODO: if a feature is discovered in a higher order variant, show dependence from higher order
import pandas as pd
import numpy as np

import sys
import copy

from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

import seaborn as sns

import utils
reload(utils)

sns.set(context = 'paper', font='serif', style = 'white')

out_dir = '../plots/'

pars = {'beginning':False}

ax1 = plt.axes(frameon=False)
ax1.set_frame_on(False)
ax1.get_xaxis().tick_bottom()
ax1.axes.get_yaxis().set_visible(False)

ax1.tick_params(labelsize=20)

df = pd.read_csv('./data/autotaggedperms.csv')
df = df.loc[~df['datetime'].isnull()]

times = utils.get_times(df)
rumor_df = utils.get_rumor_features(df, ignore = ['26_miles','5k','age_limit','bib','cassella','fuck_stupid','grass','phony','qualify','retweet_for','rip','social_media'])

#times = utils.get_times(df)
#rumor_df = utils.get_rumor_features(df)

collapsed,sorted_hypotheses = utils.get_collapsed(rumor_df)

rumor_index = dict(zip(sorted_hypotheses, range(len(sorted_hypotheses))))

dep_counts = {'innovations':0, 'mutations':0, 'deletions':0, 'merges':0}

pops = Counter(collapsed)

pop_sorted = [x[0] for x in pops.most_common()]
max_pop = float(max(pops.values()))

start_sorted = []

starts = {}
ends = {}
j = 0
for i in range(len(collapsed)):

    r = collapsed[i]
    
    if r not in starts:

        start_sorted += [r]
        starts[r] = j
        j += 1

    ends[r] = j

alphas = {}
positions = {}
widths = {}
    
for i in range(len(pops)):
    
    r = pop_sorted[i]

    if i == 0:
        positions[r] = 0.0
    else:
        positions[r] =  ( (i+1)/2 * (-1) ** i ) * 2
    #positions[r] = (1 - (i+1)/2)/(1 - 0.5) * (1 + -2 * (i % 2))
    #positions[r] = (1 - 0.9**((i+1)/2))/(1 - 0.5) * (1 + -2 * (i % 2))

    alphas[r] = np.log(pops[r]+1) / np.log(max_pop+1)
    
    min_width = 0.01
    widths[r] = (min_width + (1 - min_width) * (pops[r] / max_pop))
    
for i in range(len(pops)):

    r = start_sorted[i]

    these = np.array(list(times[collapsed == r]))
    
    x = list(these/(60.0*60*24))
    y = [positions[r]]*len(these)

    if pars['beginning']:
        to_plot = min(x) < 1
    else:
        to_plot = True

    if to_plot:
        ax1.scatter(x,y,alpha = 0.5, s = 10)#,s = widths[r])
        
        features,tags = utils.get_features(r, rumor_df.columns)
        name = ','.join(tags)
        #plt.text(min(x), positions[r] + 0.05*alphas[r], name, size = 8)# * alphas[r], alpha = alphas[r])


if pars['beginning']:
    plt.xlim(-0.1, 1)    
else:
    plt.xlim(-0.1, 1.01*max(times/(60.0*60*24)))
plt.ylim(1.05*min(positions.values()), 1.02*max(positions.values()))

plt.xlabel('Time (Days)', fontsize = 20)

xmin, xmax = ax1.get_xaxis().get_view_interval()
ymin, ymax = ax1.get_yaxis().get_view_interval()
ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))

keys = sorted(pars.keys())
plt_file_base = '-'.join([k + '=' + str(pars[k]) for k in keys])
plt.savefig(out_dir + 'rumors-instances-' + plt_file_base + '.pdf', bbox_inches = 'tight')

plt.close()

