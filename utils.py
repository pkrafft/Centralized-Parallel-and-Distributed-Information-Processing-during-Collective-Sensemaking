
# TODO: if a feature is discovered in a higher order variant, show dependence from higher order

# TODO: stop using stupid string representation, and refactor to just use tuples

import pandas as pd
import numpy as np

from collections import Counter
import copy

def get_times(df):
    """
    >>> df = pd.DataFrame([['','2013-04-15 17:41:09',1,0,0],
    ...                    ['','2013-04-15 17:41:10',0,0,0],
    ...                    ['','2013-04-15 17:41:12',0,1,1],
    ...                    ['','2013-04-15 17:41:16',1,1,0]])
    >>> df.columns = ['id','datetime','a','b','c']
    >>> get_times(df)
    0    0
    1    1
    2    3
    3    7
    Name: datetime, dtype: int64
    """

    df['datetime'] = pd.to_datetime(df['datetime'])
    times = (df['datetime'] - df['datetime'].iloc[0]).apply(lambda x: int(x.total_seconds()))
    #times = (df['datetime'] - df['datetime'].iloc[0]).apply(lambda x: int(x.total_seconds()/60))

    return times

def get_rumor_features(df, ignore = ['rip']):
    """
    >>> df = pd.DataFrame([['','2013-04-15 17:41:09',1,0,0,1],
    ...                    ['','2013-04-15 17:42:37',0,0,0,1],
    ...                    ['','2013-04-15 17:44:24',0,1,1,1],
    ...                    ['','2013-04-15 17:48:11',1,1,0,1]])
    >>> df.columns = ['id','datetime','a','b','c','rip']
    >>> df = get_rumor_features(df)
    >>> df
       a  b  c
    0  1  0  0
    1  0  0  0
    2  0  1  1
    3  1  1  0
    """
    
    tags = set(df.iloc[:,2:].columns).difference(ignore)
    rumor_features = df.loc[:,tags]
    rumor_features = rumor_features.iloc[:,np.argsort(-np.sum(rumor_features, 0))]
    
    return rumor_features

def num_features(x):
    """
    >>> num_features('0-1-1') 
    2
    >>> num_features('0-0-0') 
    0
    """
    return sum(np.array(x.split('-')) == '1')

def get_features(x, tag_names):
    """
    >>> df = pd.DataFrame([['','','']])
    >>> df.columns = ['a','b','c']
    >>> get_features('1-1-0', df.columns)
    (['1-1-0', '0-1-0', '1-0-0'], ['a', 'b'])
    """
    
    feature_nums = np.nonzero(np.array(x.split('-')) == '1')[0]

    combos = get_all_combinations(feature_nums)
    
    features = []
    for inds in combos:
        
        if len(inds) == 0:
            continue
        
        f = np.zeros(len(tag_names), dtype = int)
        f[inds] = 1
        
        features += ['-'.join(map(str, f))]
    
    return features, list(tag_names[feature_nums])

def get_tag_nums(x):
    """
    >>> get_tag_nums('1-1-0')
    array([0, 1])
    """
    
    feature_nums = np.nonzero(np.array(x.split('-')) == '1')[0]

    return feature_nums

def remove_new_tags(x, old_tags):
    """
    >>> remove_new_tags('1-0-1-0-1-1',[0,1,5])
    '1-0-0-0-0-1'
    >>> remove_new_tags('1-0-1-0-1-1',range(6))
    '1-0-1-0-1-1'
    """

    x = np.array(x.split('-'))
    feature_nums = np.nonzero(x == '1')[0]

    new_tags = set(feature_nums).difference(old_tags)

    feature_nums = set(feature_nums).difference(new_tags)

    return make_string_from_tag_nums(feature_nums, len(x))

def make_string_from_tag_nums(tag_nums, total_length):
    """
    >>> make_string_from_tag_nums([0,2],4)
    '1-0-1-0'
    """
    
    x = np.zeros(total_length,dtype=int)
    x[list(tag_nums)] = 1
    x = '-'.join(map(str, x))
    return x
    

def get_all_combinations(inds):
    """
    >>> get_all_combinations([])
    [[]]
    >>> get_all_combinations([1])
    [[1], []]
    >>> get_all_combinations([1,3])
    [[1, 3], [3], [1], []]
    >>> get_all_combinations([1,3,4])
    [[1, 3, 4], [3, 4], [1, 4], [4], [1, 3], [3], [1], []]
    """

    if len(inds) == 0:
        return [[]]

    combos = get_all_combinations(inds[:-1])
    
    for c in combos:
        c += [inds[-1]]
    
    combos += get_all_combinations(inds[:-1])
    
    return combos

def get_collapsed(rumor_features):
    """
    >>> df = pd.DataFrame([[1,0,0],
    ...                    [0,0,0],
    ...                    [0,1,1],
    ...                    [1,1,0]])
    >>> collapsed, labels = get_collapsed(df)
    >>> collapsed
    0    1-0-0
    1    0-0-0
    2    0-1-1
    3    1-1-0
    dtype: object
    >>> labels
    ['0-0-0', '1-0-0', '0-1-1', '1-1-0']
    """
    
    collapsed = rumor_features.apply(lambda x: '-'.join(map(str, x)), axis = 1)
    
    sorted_hypotheses = sorted(set(collapsed),
                               key = lambda x: (num_features(x), x))
    
    return collapsed, sorted_hypotheses

def get_tag_starts(times, collapsed, sorted_hypotheses, tag_labels):
    """
    >>> df = pd.DataFrame([['','2013-04-15 17:41:09',1,0,0],
    ...                    ['','2013-04-15 17:41:10',0,0,0],
    ...                    ['','2013-04-15 17:41:18',0,1,1],
    ...                    ['','2013-04-15 17:41:20',1,1,0],
    ...                    ['','2013-04-15 17:41:20',1,1,1]])
    >>> df.columns = ['id','datetime','a','b','c']
    >>> times = get_times(df)
    >>> rumor_features = get_rumor_features(df)
    >>> collapsed, sorted_hypotheses = get_collapsed(rumor_features)
    >>> collapsed
    0    1-0-0
    1    0-0-0
    2    0-1-1
    3    1-1-0
    4    1-1-1
    dtype: object
    >>> get_tag_starts(times, collapsed, sorted_hypotheses, rumor_features.columns)
    ({'0-1-0': '0-1-1', '0-1-1': '0-1-1', '1-0-0': '1-0-0', '1-0-1': '1-1-1', '1-1-1': '1-1-1', '1-1-0': '1-1-0', '0-0-1': '0-1-1'}, {'0-1-0': 9, '0-1-1': 9, '1-0-0': 0, '1-0-1': 11, '1-1-1': 11, '1-1-0': 11, '0-0-1': 9})
    """

    rumor_pointers = {}
    rumor_starts = {}
    
    for r in sorted_hypotheses:
        
        these = list(times[collapsed == r])
        
        features,tags = get_features(r, tag_labels)
        
        for f in features:
            
            if f not in rumor_pointers or min(these) < rumor_starts[f]:
                rumor_pointers[f] = r
                rumor_starts[f] = min(these)
    
    return rumor_pointers, rumor_starts

def get_dependencies(times, collapsed, sorted_hypotheses, tag_labels):
    """
    >>> df = pd.DataFrame([['','2013-04-15 17:41:09',1,0,0,0],
    ...                    ['','2013-04-15 17:45:37',0,0,0,0],
    ...                    ['','2013-04-15 17:50:24',0,1,1,0],
    ...                    ['','2013-04-15 17:51:24',0,1,0,0],
    ...                    ['','2013-04-15 17:55:11',1,1,0,1],
    ...                    ['','2013-04-15 17:55:12',1,1,0,0],
    ...                    ['','2013-04-15 17:55:13',1,1,1,0]])
    >>> df.columns = ['id','datetime','a','b','c','d']
    >>> times = get_times(df)
    >>> rumor_features = get_rumor_features(df)
    >>> collapsed, sorted_hypotheses = get_collapsed(rumor_features)
    >>> collapsed
    0    0-1-0-0
    1    0-0-0-0
    2    1-0-1-0
    3    1-0-0-0
    4    1-1-0-1
    5    1-1-0-0
    6    1-1-1-0
    dtype: object
    >>> get_dependencies(times, collapsed, sorted_hypotheses, rumor_features.columns)
    {'1-1-1-0': ['1-0-1-0', '1-1-0-1'], '1-1-0-0': ['1-1-0-1'], '1-1-0-1': ['0-1-0-0', '1-0-1-0', '1-1-0-1'], '1-0-1-0': ['1-0-1-0'], '0-1-0-0': ['0-1-0-0'], '1-0-0-0': ['1-0-1-0'], '0-0-0-0': []}
    """

    # TODO: dependencies for hypothesis complexities greater than 2 are underdetermined
    
    rumor_pointers, rumor_starts = get_tag_starts(times, collapsed, sorted_hypotheses, tag_labels)

    dependencies = {}
    
    for r in sorted_hypotheses:

        dependencies[r] = []
        
        features,tags = get_features(r, tag_labels)

        sorted_combos = sorted(features,
                               key = lambda x: (-num_features(x), x))

        feature_dict = dict(zip(features, [1]*len(features)))
        
        for f in sorted_combos:
            
            if rumor_pointers[f] != r and f in feature_dict:
                
                dependencies[r] += [rumor_pointers[f]]
                
                inside_features,tags = get_features(f, tag_labels)
                
                for g in inside_features:

                    if g in feature_dict:
                        del(feature_dict[g])

        if r not in get_features(merge_hypotheses(dependencies[r], tag_labels),tag_labels)[0]:
            if num_features(r) > 0:
                dependencies[r] += [r]

    return dependencies

def merge_hypotheses(hypotheses, tag_names):
    """
    >>> merge_hypotheses(['1-0-1','1-1-0'], ['a','b','c'])
    '1-1-1'
    >>> merge_hypotheses(['1-0-0-0-0','1-0-0-0-0','0-0-1-0-1','0-0-0-1-0'], ['a','b','c','d','e'])
    '1-0-1-1-1'
    """
    
    feature_nums = []
    
    for h in hypotheses:
        feature_nums += list(np.nonzero(np.array(h.split('-')) == '1')[0])

    f = np.zeros(len(tag_names), dtype = int)
    f[feature_nums] = 1
    
    return '-'.join(map(str, f))

def merge(u, v):
    """
    >>> merge((1,0,0),(0,0,0))
    (1, 0, 0)
    >>> merge((1,0,0),(0,0,1))
    (1, 0, 1)
    """
    return tuple((np.array(u,dtype=bool) + np.array(v,dtype=bool)).astype(int))

def contains(u, v):
    """
    >>> contains((1,0,0),(0,0,0))
    True
    >>> contains((1,1,0),(0,1,0))
    True
    >>> contains((0,1,0),(1,1,0))
    True
    >>> contains((1,1,0),(1,1,0))
    True
    >>> contains((1,1,0),(0,1,1))
    False
    """
    diff = np.array(u) - np.array(v)
    return (sum(diff <= 0) == len(diff)) or (sum(diff >= 0) == len(diff))

class Merger():

    def __init__(self):
        self.combos = {}
        self.combos[0] = set()
        self.tags = set()

    def add(self, thread):
        """
        >>> m = Merger()
        >>> m.add((1,0,0,0))
        >>> m.add((1,0,1,0))
        >>> m.combos
        {0: set([(1, 0, 1, 0), (1, 0, 0, 0)])}
        >>> m.tags
        set([0, 2])
        """

        assert sum(np.array(thread) > 0) > 0
        
        thread = tuple(thread)
        
        assert thread not in self.combos[0]
        
        self.combos[0] = self.combos[0].union([thread])
        self.tags = self.tags.union(np.nonzero(thread)[0])
        
        max_num = max(self.combos.keys())
        
        for num in range(1,max_num+1):
            
            for u in self.combos[num - 1]:

                self.try_adding(self.combos[num], thread, u)
        
    def expand(self):
        """
        >>> m = Merger()
        >>> m.add((1,0,0,0))
        >>> m.add((1,0,1,0))
        >>> m.expand()
        >>> m.combos
        {0: set([(1, 0, 1, 0), (1, 0, 0, 0)]), 1: {}}
        >>> m.add((0,0,1,1))
        >>> m.add((0,0,0,1))
        >>> m.combos
        {0: set([(1, 0, 1, 0), (0, 0, 0, 1), (1, 0, 0, 0), (0, 0, 1, 1)]), 1: {(1, 0, 1, 1): [((0, 0, 1, 1), (1, 0, 1, 0)), ((0, 0, 1, 1), (1, 0, 0, 0)), ((0, 0, 0, 1), (1, 0, 1, 0))], (1, 0, 0, 1): [((0, 0, 0, 1), (1, 0, 0, 0))]}}
        >>> m.expand()
        >>> m.expand()
        >>> m.expand()
        >>> m.combos
        {0: set([(1, 0, 1, 0), (0, 0, 0, 1), (1, 0, 0, 0), (0, 0, 1, 1)]), 1: {(1, 0, 1, 1): [((0, 0, 1, 1), (1, 0, 1, 0)), ((0, 0, 1, 1), (1, 0, 0, 0)), ((0, 0, 0, 1), (1, 0, 1, 0))], (1, 0, 0, 1): [((0, 0, 0, 1), (1, 0, 0, 0))]}, 2: {(1, 0, 1, 1): [((1, 0, 0, 1), (1, 0, 1, 0)), ((1, 0, 0, 1), (0, 0, 1, 1))]}, 3: {}, 4: {}}
        >>> m = Merger()
        >>> m.add((1,0,0,0))
        >>> m.add((0,0,0,1))
        >>> m.expand()
        >>> m.combos
        {0: set([(0, 0, 0, 1), (1, 0, 0, 0)]), 1: {(1, 0, 0, 1): [((0, 0, 0, 1), (1, 0, 0, 0))]}}
        """

        max_num = max(self.combos.keys())

        self.combos[max_num + 1] = {}

        singletons = list(self.combos[0])
        
        if max_num == 0:
            for i in range(len(singletons)-1):
                for j in range(i+1,len(singletons)):
                    self.try_adding(self.combos[1], singletons[i], singletons[j])
        else:
            for t in self.combos[max_num]:
                for u in singletons:
                    self.try_adding(self.combos[max_num + 1], t, u)


    def try_adding(self,combos,t,u):
        
        if contains(t,u):
            return
        
        v = merge(t, u)
        
        if v in combos:
            combos[v] += [(t,u)]
        else:
            combos[v] = [(t,u)]
        
                    
    def find(self, thread):
        """
        >>> m = Merger()
        >>> m.add((0,1,0,0))
        >>> m.add((1,0,0,0))
        >>> m.find((1,1,0,0))
        [[(1, 0, 0, 0), (0, 1, 0, 0)]]
        >>> m.find((0,0,1,1))
        [[(0, 0, 1, 1)]]
        >>> m = Merger()
        >>> m.add((0,0,0,1))
        >>> m.add((1,0,1,0))
        >>> m.add((1,0,0,0))
        >>> m.add((0,0,1,0))
        >>> m.find((1,0,1,1))
        [[(1, 0, 1, 0), (0, 0, 0, 1)]]
        >>> m.find((1,1,1,1))
        [[(1, 0, 1, 0), (0, 0, 0, 1), (1, 1, 1, 1)]]
        >>> m.add((0,0,1,1))
        >>> m.find((1,1,1,1))
        [[(1, 0, 1, 0), (0, 0, 0, 1), (1, 1, 1, 1)], [(0, 0, 1, 1), (1, 0, 1, 0), (1, 1, 1, 1)], [(0, 0, 1, 1), (1, 0, 0, 0), (1, 1, 1, 1)]]
        """
        
        thread = tuple(thread)

        orig_thread = copy.copy(thread)
        
        deps = []
        
        if len(self.combos[0]) == 0 or thread in self.combos[0]:
            return [[thread]]
        
        thread = self.remove_new_tags(thread)

        if sum(thread) == 0:
            return [[orig_thread]]
        else:
            if thread in self.combos[0]:
                return [[thread,orig_thread]]

        max_merges = sum(thread)
        #max_merges = len(self.combos[0])
        
        for num_mods in range(1,max_merges+1):
            
            deps += self.find_restricted(thread, num_mods)
            
            if len(deps) > 0:
                break

        assert len(deps) > 0

        if thread != orig_thread:
            for d in deps:
                d += [orig_thread]

        # if max(self.combos) > 3:
        #     for num_mods in range(4,max(self.combos)):
        #         del(self.combos[num_mods])
                
        return deps

    def find_restricted(self, thread, num_mods):
        """
        >>> m = Merger()
        >>> m.add((0,0,0,1))
        >>> m.add((0,0,1,0))
        >>> m.add((1,0,1,0))
        >>> m.add((0,0,1,1))
        >>> m.find_restricted((1,0,1,1),1)
        [[(1, 0, 1, 0), (0, 0, 0, 1)], [(1, 0, 1, 0), (0, 0, 1, 1)]]
        >>> m = Merger()
        >>> m.add((1,0,1,0))
        >>> m.add((0,1,0,0))
        >>> m.add((0,0,0,1))
        >>> m.find_restricted((0,1,1,1),1)
        []
        >>> m.find_restricted((0,1,1,1),2)
        []
        >>> m.find_restricted((0,1,1,1),3)
        [[(1, 0, 1, 0), (0, 0, 0, 1), (0, 1, 0, 0)], [(1, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)], [(0, 0, 0, 1), (0, 1, 0, 0), (1, 0, 1, 0)]]
        """
        
        deps = []
        
        while num_mods not in self.combos:
            self.expand()
        
        if thread in self.combos[num_mods]:
            deps += self.unroll_deps(thread)

        for num_merges in range(num_mods):
            
            parents = self.check_diffs(thread, num_merges, num_mods)
            
            for u in parents:
                deps += self.unroll_deps(u)
        
        return deps

    def check_diffs(self, thread, num_merges, num_mods):
        """
        >>> m = Merger()
        >>> m.add((0,0,1,0))
        >>> m.add((1,0,1,0))
        >>> m.add((0,0,1,1))
        >>> m.add((1,0,1,1))
        >>> m.check_diffs((0,0,1,0), 0, 1)
        array([[1, 0, 1, 0],
               [0, 0, 1, 1]])
        >>> m.check_diffs((0,0,1,1), 0, 1)
        array([[1, 0, 1, 1]])
        >>> m.check_diffs((1,1,1,1), 0, 1)
        array([], shape=(0, 4), dtype=int64)
        >>> m.expand()
        >>> m.check_diffs((0,0,1,0), 0, 2)
        array([[1, 0, 1, 1]])
        >>> m.check_diffs((0,0,1,1), 0, 2)
        array([], shape=(0, 4), dtype=int64)
        >>> m.add((0,1,0,0))
        >>> m.check_diffs((1,1,0,0), 1, 2)
        array([[1, 1, 1, 0]])
        """

        combos = np.array(list(self.combos[num_merges]))
        if len(combos) == 0:
            return []
        
        diffs = combos - np.array(thread)
        
        keep = np.sum(diffs < 0, 1) == 0

        totals = np.sum(diffs, 1)
        
        inds = (totals + num_merges == num_mods) & keep

        return combos[inds]
    
    def unroll_deps(self, thread):
        """
        >>> m = Merger()
        >>> m.add((1,0,0,0))
        >>> m.add((1,0,1,0))
        >>> m.add((0,0,1,1))
        >>> m.add((0,0,0,1))
        >>> x = [m.expand() for i in range(4)]
        >>> m.unroll_deps((1,0,0,0))
        [[(1, 0, 0, 0)]]
        >>> m.unroll_deps((1,0,0,1))
        [[(0, 0, 0, 1), (1, 0, 0, 0)]]
        >>> m.unroll_deps((1,0,1,1))
        [[(1, 0, 1, 0), (0, 0, 0, 1)], [(1, 0, 1, 0), (0, 0, 1, 1)], [(1, 0, 0, 0), (0, 0, 1, 1)]]
        >>> m = Merger()
        >>> m.add((1,0,0,0))
        >>> m.add((0,1,0,0))
        >>> m.add((0,0,1,0))
        >>> m.add((0,0,0,1))
        >>> x = [m.expand() for i in range(4)]
        >>> y = m.unroll_deps((1,1,1,1)) # TODO: simplify deps return 
        >>> assert np.array([(np.sum(a,1) == np.array([1,1,1,1])).all() for a in y]).all()
        """
        
        thread = tuple(thread)

        if thread in self.combos[0]:
            return [[thread]]

        deps = []

        max_mods = max(self.combos.keys())
        
        for num_mods in range(1, max_mods + 1):

            if thread in self.combos[num_mods]:

                for parents in self.combos[num_mods][thread]:

                    assert len(parents) == 2
                    
                    all_paths_left = self.unroll_deps(parents[0])
                    all_paths_right = self.unroll_deps(parents[1])
                    
                    for p in all_paths_left:
                        for q in all_paths_right:
                            deps += [p + q]
                
                break
            
        return deps
    
    def remove_new_tags(self, thread):
        """
        >>> m = Merger()
        >>> m.tags = set([0,2])
        >>> m.remove_new_tags((0,1,1,1))
        (0, 0, 1, 0)
        """
        thread = np.array(thread)
        new_tags = self.get_new_tags(thread)
        thread[new_tags] = 0
        return tuple(thread)
        
    def get_new_tags(self, thread):
        """
        >>> m = Merger()
        >>> m.tags = set([0,2])
        >>> m.get_new_tags((0,1,1,1))
        [1, 3]
        """
        return list(set(np.nonzero(thread)[0]).difference(self.tags))

def get_simplest_dependencies(threads):
    """
    >>> threads = [(0,0,0,0),(1,0,0,0),(0,0,0,0),(1,1,0,0),(0,0,0,0),(1,1,0,0),(0,1,0,1),(1,1,0,1)]
    >>> get_simplest_dependencies(threads)
    {'1-1-0-0': ['1-0-0-0', '1-1-0-0'], '1-1-0-1': ['0-1-0-1', '1-1-0-0'], '0-1-0-1': ['1-1-0-0', '0-1-0-1'], '1-0-0-0': ['1-0-0-0']}
    """

    threads = np.array(threads)
    
    merger = Merger()

    counts = Counter()

    deps = {}
    
    for thread in threads:

        thread = tuple(thread)
        
        if thread not in counts and sum(thread) > 0:

            candidates = merger.find(thread)
            deps[thread] = get_most_popular(candidates, counts)

            assert len(deps[thread]) > 0
            
            merger.add(thread)
        
        counts.update([thread])

    return convert_deps_to_collapsed(deps)

def convert_deps_to_collapsed(deps):

    for k in deps.keys():
        s = '-'.join(map(str, k))
        deps[s] = []
        for t in deps[k]:
            deps[s] += ['-'.join(map(str, t))]
        del(deps[k])

    return deps
        

def get_most_popular(candidates, counts):
    """
    >>> candidates = [[(1,0,0,0),(0,1,0,1)],[(1,1,0,0),(0,1,0,1)]]
    >>> counts = Counter([(1,0,0,0),(1,1,0,0),(1,1,0,0),(0,1,0,1)])
    >>> get_most_popular(candidates, counts)
    [(1, 1, 0, 0), (0, 1, 0, 1)]
    """
    
    pops = get_popularities(candidates, counts)
    
    return candidates[np.argmax(pops)]

def get_popularities(candidates, counts):

    """
    >>> candidates = [[(1,0,0,0),(0,1,0,1)],[(1,1,0,0),(0,1,0,1)]]
    >>> counts = Counter([(1,0,0,0),(1,1,0,0),(1,1,0,0),(0,1,0,1)])
    >>> get_popularities(candidates, counts)
    [1.0, 1.5]
    """
    pops = []
    
    for c in candidates:
        pops += [np.mean([counts[thread] for thread in c])]

    return pops

if __name__ == "__main__":
    import doctest
    doctest.testmod()



    
