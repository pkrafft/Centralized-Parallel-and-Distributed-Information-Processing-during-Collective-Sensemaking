# TODO: if a feature is discovered in a higher order variant, show dependence from higher order
import pandas as pd
import numpy as np

import sys
import copy

from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

import utils
reload(utils)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

out_dir = '../plots/'

pars = {}
pars['simplest'] = True
pars['beginning'] = True
pars['points'] = False
pars['lines'] = True
if pars['lines']:
    if len(sys.argv) > 1:
        pars['num_tags'] = int(sys.argv[1])
    else:
        pars['num_tags'] = 'all'


df = pd.read_csv('./data/autotaggedperms.csv')
df = df.loc[~df['datetime'].isnull()]

times = utils.get_times(df)
rumor_df = utils.get_rumor_features(df, ignore = ['26_miles','5k','age_limit','bib','cassella','fuck_stupid','grass','phony','qualify','retweet_for','rip','social_media'])

xmin = 0
if pars['beginning']:
    if pars['points']:
        xmax = 12 * 60 # minutes
    else:
        xmax = 6 * 60 # minutes
        xmax = 3.5 * 60 # minutes
else:
    xmax = max(times)/60
xmin *= -1.01
xmax *= 1.01
    
# if pars['beginning']:
#     keep = times/60.0 < xmax
#     times = times[keep]
#     rumor_df = rumor_df[keep]

collapsed,sorted_hypotheses = utils.get_collapsed(rumor_df)

if pars['simplest']:
    dependencies = utils.get_simplest_dependencies(rumor_df)
else:
    dependencies = utils.get_dependencies(times, collapsed, sorted_hypotheses, rumor_df.columns)

pops = Counter(collapsed)
max_pop = float(max(pops.values()))
    
rumor_index = dict(zip(sorted_hypotheses, range(len(sorted_hypotheses))))

dep_counts = {'innovations':0, 'mutations':0, 'deletions':0, 'merges':0}

x = []
y = []

nf = -1
for r in sorted_hypotheses:
    
    these = np.array(list(times[collapsed == r]))

    this_nf = utils.num_features(r)

    x += list(these/60)
    y += [rumor_index[r]]*len(these)

    alpha = np.sum(collapsed == r)/max_pop

    if pars['lines']:
        plt.plot([min(these)/60, max(these)/60], [rumor_index[r], rumor_index[r]], linestyle='-')#, linewidth=alpha)

    if pars['points']:
        plt.plot([min(these)/60, xmax], [rumor_index[r], rumor_index[r]], linewidth = 0.5, linestyle='-', alpha = 0.5)#, linewidth=alpha)
        
    x1 = min(these)/60
    y1 = rumor_index[r]
    
    features,tags = utils.get_features(r, rumor_df.columns)
    name = ','.join(tags)
    plt.text(xmax + 1, y1 - 0.25, name, size = 2)
    
    if pars['lines']:
        
        if this_nf == pars['num_tags'] or (this_nf > 0 and pars['num_tags'] == 'all'):

            deps = copy.deepcopy(dependencies[r])

            if r in deps:
                if len(deps) == 1:
                    dep_counts['innovations'] += 1
                    plt.scatter(x1, y1, marker = '>')
                else:
                    if this_nf == 2:
                        assert len(deps) == 2
                    dep_counts['mutations'] += 1
                    plt.scatter(x1, y1, marker = '>')
                deps.remove(r)
            else:
                if len(deps) == 1:
                    dep_counts['deletions'] += 1

            if len(deps) > 1:
                if this_nf == 2:
                    assert len(deps) == 2
                dep_counts['merges'] += 1
                
            for j,f in enumerate(deps):
                x2 = min(these)/60 - 5*j
                y2 = rumor_index[f]
                plt.plot([x1, x2], [y1, y2], linestyle='-', linewidth=0.5)
            

                    
    if this_nf > nf:
        print rumor_index[r]
        nf = this_nf
        plt.plot([min(times), max(times)], [rumor_index[r] - 0.5, rumor_index[r] - 0.5], color='k', linestyle='--', linewidth=1)
        

if pars['points']:
    plt.scatter(x, y, marker = '.', alpha = 0.5)

plt.xlim(xmin, xmax)    
plt.ylim(-0.01*max(y), 1.01*max(y))

plt.xlabel('Time Since Start (Minutes)')
plt.ylabel('Rumor Label')
if pars['lines']:
    plt.title('Dependencies of Threads with ' + str(pars['num_tags']) + ' Tags')

keys = sorted(pars.keys())
plt_file_base = '-'.join([k + '=' + str(pars[k]) for k in keys])
plt.savefig(out_dir + 'rumors-' + plt_file_base + '.pdf', bbox_inches = 'tight')

plt.close()

print dep_counts


######

time_sorted = []
for r in collapsed:
    if r not in time_sorted:
        time_sorted += [r]

rumor_index = dict(zip(time_sorted, range(len(time_sorted))))

for i in range(27):

    r = time_sorted[i]
    
    this_nf = utils.num_features(r)
    
    features,tags = utils.get_features(r, rumor_df.columns)
    name = ','.join(tags)

    if pops[r] > 100:
        plt.text(i - 0.5, -0.1, name, size = 7, rotation = 'vertical')
        alpha = 1
    else:
        alpha = 0.1
    #alpha = np.sqrt(pops[r]/max_pop)

    plt.scatter([i]*3,[-1,0,1],s=10*alpha)
    plt.plot([i,i],[-1,1], 'k-', alpha = 0.1, linewidth = 0.1)
        
    if this_nf > 0:

        deps = copy.deepcopy(dependencies[r])

        if r in deps:
            if len(deps) == 1:
                dep_counts['innovations'] += 1
                plt.scatter(i, 0, marker = '>', s = alpha)
                deps.remove(r)
            else:
                if this_nf == 2:
                    assert len(deps) == 2
                dep_counts['mutations'] += 1
                plt.scatter(i, 0, marker = '>', s = alpha)
                deps.remove(r)
                x2 = rumor_index[deps[0]]
                plt.plot([i, x2], [0, 1], 'k-', alpha = alpha)
        else:
            if len(deps) == 1:
                dep_counts['deletions'] += 1
                x2 = rumor_index[deps[0]]
                plt.plot([i,x2],[0,1], 'k-', alpha = alpha)

        if len(deps) > 1:
            if this_nf == 2:
                assert len(deps) == 2
            dep_counts['merges'] += 1

            for j,f in enumerate(deps):
                x2 = rumor_index[f]
                plt.plot([i, x2], [0, -1], 'k-', alpha = alpha)

plt.axis('off')

keys = sorted(pars.keys())
plt_file_base = '-'.join([k + '=' + str(pars[k]) for k in keys])
plt.savefig(out_dir + 'rumors-tripartite-' + plt_file_base + '.pdf', bbox_inches = 'tight')

plt.close()


######

sns.set(context = 'paper', font='serif', style = 'white')

default_dep = '-'.join(['0'] * (len(collapsed[0])/2 + 1))

num_plot = len(start_sorted)

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

positions = {}
alphas = {}
widths = {}

def recurse_alpha(alphas,r,a = None):

    if r in alphas:
        a = max(alphas[r],a)

    #min_alpha = 0.2
    #my_alpha = min_alpha + (1-min_alpha)*np.sqrt(pops[r] / max_pop)

    my_alpha = np.log(pops[r]+1) / np.log(max_pop+1)
    
    a = max(a, my_alpha)
    
    alphas[r] = a

    if r in dependencies:
        for s in dependencies[r]:
            if s != r:
                alphas = recurse_alpha(alphas,s,a)

    return alphas
    
for i in range(len(pops)):
    
    r = pop_sorted[i]

    #positions[r] = np.log((i + 1)/2 + 1) * (1 + -2 * (i % 2))
    positions[r] = (1 - 0.9**((i+1)/2))/(1 - 0.5) * (1 + -2 * (i % 2))
    
    min_width = 0.1
    widths[r] = 2 * (min_width + (1 - min_width) * (pops[r] / max_pop))

    #widths[r] = 2 * np.log(pops[r]+1) / np.log(max_pop+1)
    
    alphas = recurse_alpha(alphas,r)
    
for i in range(num_plot):

    r = start_sorted[i]

    assert starts[r] < num_plot
    
    plt.scatter(starts[r],positions[r],alpha = alphas[r],s = 4*alphas[r])
    
    plt.plot([starts[r],ends[r]],
             [positions[r],positions[r]],
             'k-',
             linewidth = widths[r],
             alpha = alphas[r])
    
    this_nf = utils.num_features(r)

    features,tags = utils.get_features(r, rumor_df.columns)
    name = ','.join(tags)
    plt.text(starts[r]+0.05*alphas[r], positions[r] + 0.05*alphas[r], name, size = 5 * alphas[r], alpha = alphas[r])
    
    if this_nf > 0:        
        deps = copy.deepcopy(dependencies[r])
        if len(deps) == 1 and deps[0] == r:
            deps += [default_dep]
    else:
        deps = [r]

    for j,f in enumerate(deps):
        if f != r:
            x1 = starts[r]
            x2 = starts[r] - 1 + 0.5 ** (j+1)
            y1 = positions[r]
            y2 = positions[f]
            plt.plot([x1,x2], [y1, y2], 'k-', alpha = alphas[r], linewidth = widths[r])

ax = plt.axes()            
x = [num_plot/2 - num_plot/8,num_plot/2 + num_plot/8]
y = [min(positions.values())*1.1]*2
ax.arrow(x[0], y[0], num_plot/4, 0, head_width=0.05, head_length=0.2, fc='k', ec='k', linewidth=0.2)
plt.text(num_plot/2,min(positions.values())*1.15,'Time',size=5)
            
plt.xlim(-0.2,num_plot)
plt.axis('off')

keys = sorted(pars.keys())
plt_file_base = '-'.join([k + '=' + str(pars[k]) for k in keys])
plt.savefig(out_dir + 'rumors-flow-' + plt_file_base + '.pdf', bbox_inches = 'tight')

plt.close()

