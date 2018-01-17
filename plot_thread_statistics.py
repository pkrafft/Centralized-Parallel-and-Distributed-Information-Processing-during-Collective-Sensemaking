# TODO: if a feature is discovered in a higher order variant, show dependence from higher order
import pandas as pd
import numpy as np

import sys
import copy

from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

import scipy.stats as stats

import utils
reload(utils)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

out_dir = '../plots/'

df = pd.read_csv('./data/autotaggedperms.csv')
df = df.loc[~df['datetime'].isnull()]

# times = utils.get_times(df)
# rumor_df = utils.get_rumor_features(df)

times = utils.get_times(df)
rumor_df = utils.get_rumor_features(df, ignore = ['26_miles','5k','age_limit','bib','cassella','fuck_stupid','grass','phony','qualify','retweet_for','rip','social_media'])


collapsed,sorted_hypotheses = utils.get_collapsed(rumor_df)

stat_df = []

start_sorted = []

for i in range(len(collapsed)):

    r = collapsed[i]
    
    if r not in start_sorted:

        start_sorted += [r]

all_obs_features = set([])
all_obs_tags = set([])
all_obs_threads = set([])

for i in range(len(start_sorted)):

    r = start_sorted[i]
    
    these = np.array(list(times[collapsed == r]))

    num_obs = len(these)

    features,tags = utils.get_features(r, rumor_df.columns)
    tag_nums = utils.get_tag_nums(r)
    
    num_tags = len(tags)
    
    assert num_tags == len(tag_nums)
    
    intro = min(these)
    duration = max(these) - min(these)

    innovation = 0    
    possible_mutation = 0
    possible_deletion = 0
    possible_merge = 0
        
    if num_tags == 0:
        assert i == 0
        innovation = 1
    else:
        
        s = utils.remove_new_tags(r, all_obs_tags)
        
        if r != s:
            innovation = 1

        # TODO: if context is not conditional on rumor, need to check
        # that s has tags to carry on, because otherwise it is an
        # innovation, but with conditional context it can be treated
        # as a mutation
        if s in all_obs_threads:
            possible_mutation = 1
        else:
            if s in all_obs_features:
                possible_deletion = 1
            else:
                possible_merge = 1

    if possible_mutation == 1 or possible_deletion == 1:
        dep_type = 'Single'
    elif possible_merge == 1:
        dep_type = 'Multiple'
    else:
        assert i == 0
        dep_type = 'Single'
    
    all_obs_features = all_obs_features.union([r])                
    all_obs_features = all_obs_features.union(features)
    all_obs_tags = all_obs_tags.union(tag_nums)
    all_obs_threads = all_obs_threads.union([r])

    stat_df += [[r, ','.join(tags), num_obs, num_tags, intro / (60.0 * 60), duration / (60.0 * 60 * 24), innovation, possible_mutation, possible_deletion, possible_merge, dep_type]]

stat_df = pd.DataFrame(stat_df)
stat_df.columns = ['Thread','Features','Num. Tweets','Num. Features','Hour Introduced','Duration (Days)','Innovation','Mutation','Deletion','Merge','Dependencies']

stat_df['Thread log(Vol.)'] = np.log(stat_df['Num. Tweets'])

sns.set(context = 'paper', font_scale = 5, font='serif', style = 'white')

sns.distplot(stat_df['Thread log(Vol.)'], kde = False, bins = 7)
plt.ylabel('Frequency')
plt.savefig(out_dir + 'pop-hist.pdf', bbox_inches = 'tight')
plt.close()

sns.distplot(stat_df['Num. Features'], kde = False, bins = 5)
plt.savefig(out_dir + 'tag-hist.pdf', bbox_inches = 'tight')
plt.close()


sns.distplot(stat_df['Hour Introduced'], kde = False)
plt.savefig(out_dir + 'start-hist.pdf', bbox_inches = 'tight')
plt.close()

sns.distplot(stat_df['Duration (Days)'], kde = False, bins = 5)
plt.savefig(out_dir + 'duration-hist.pdf', bbox_inches = 'tight')
plt.close()

sns.set(context = 'paper', font_scale = 4, font='serif', style = 'white')

sns.regplot(stat_df['Duration (Days)'], stat_df['Thread log(Vol.)'], scatter_kws={"s": 100})
plt.savefig(out_dir + 'duration-pop.pdf', bbox_inches = 'tight')
plt.close()

sns.regplot(stat_df['Hour Introduced'], stat_df['Thread log(Vol.)'], fit_reg = False, scatter_kws={"s": 100})
ax = plt.axes()
xmin, xmax = ax.get_xaxis().get_view_interval()
ymin, ymax = ax.get_yaxis().get_view_interval()
plt.xlim(-0.05*xmax,xmax)
plt.ylim(-0.05*ymax,ymax)
plt.savefig(out_dir + 'start-pop.pdf', bbox_inches = 'tight')
plt.close()


sns.boxplot(stat_df['Dependencies'], stat_df['Thread log(Vol.)'])
sns.swarmplot(stat_df['Dependencies'], stat_df['Thread log(Vol.)'], color=".25",alpha=0.5, size=10)
plt.title('MELB Method')
plt.savefig(out_dir + 'dep-pop.pdf', bbox_inches = 'tight')
plt.close()

print np.sum(stat_df.loc[stat_df['Dependencies'] == 'Single']['Num. Tweets']) / float(np.sum(stat_df['Num. Tweets']))

print stats.ttest_ind(stat_df.loc[stat_df['Dependencies'] == 'Single']['Thread log(Vol.)'], stat_df.loc[stat_df['Dependencies'] == 'Multiple']['Thread log(Vol.)'])
