# -*- coding: utf-8 -*-
"""
@author: hyeongyuy
"""
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

workingdir = r'working'
os.chdir(os.path.join(workingdir, 'modules' ))
from usertree import userTree as utr
import utils

"""
########################################################################
Set parameters
"""
# target attribute
target_att = 'target'

#model parameters
MAX_DEPTH = 100
sample_ratio = 0.01

os.chdir(os.path.join(workingdir))
"""
########################################################################
Set directory & Load data
"""
# load data
d_set = 'votes.csv'
data = pd.read_csv(os.path.join(workingdir, 'dataset', d_set))
colnm = data.columns
n_samples = round(sample_ratio * len(data))
in_feature = list(data.columns [data.columns != target_att])
X = data.loc[:, in_feature]
y = data.loc[:, target_att]

## split data
test_ratio = 0.2
X_train, X_test, _, _ = train_test_split(X, y, test_size=test_ratio, \
                stratify=y, shuffle =True)
train_idx, test_idx = X_train.index, X_test.index
train, test = data.loc[train_idx,:], data.loc[test_idx,:]
y_train = train['target']

"""
########################################################################
Modeling
"""
## train
#criterion : entropy
entropy_tree_ins = utr(n_samples, MAX_DEPTH, params = ['entropy'])
entropy_tree, entropy_graph_tree = \
    entropy_tree_ins.fit(train, target_attribute_name = "target")
      
#criterion : gini
gini_tree_ins = utr(n_samples, MAX_DEPTH, params = ['gini'])
gini_tree, gini_graph_tree = \
    gini_tree_ins.fit(train, target_attribute_name = "target")
    
##predict
#entropy
entropy_tree_pred, entropy_tree_pred_prob = \
    entropy_tree_ins.predict(test, entropy_tree)
#gini
gini_tree_pred, gini_tree_pred_prob = \
    gini_tree_ins.predict(test, gini_tree)

"""
########################################################################
Check performance
"""
entropy_tree_perform  = \
        utils.perform_check(test['target'],\
        entropy_tree_pred, entropy_tree_pred_prob, \
        entropy_tree_ins.NUM_CLASSES, entropy_tree_ins.CLASS_DICT_, average='micro')
gini_tree_perform  = \
        utils.perform_check(test['target'],\
        gini_tree_pred, gini_tree_pred_prob, \
        gini_tree_ins.NUM_CLASSES, gini_tree_ins.CLASS_DICT_, average='micro')

perform_base_str = '{} : ACCURACY :{}, RECALL :{}, PRECISION : {}, F1 : {}, AUC : {}'
print(perform_base_str .format('entropy\t', *np.round(np.array(entropy_tree_perform), 3)))
print(perform_base_str .format('gini\t', *np.round(np.array(gini_tree_perform), 3)))


"""
########################################################################
save & Load models
"""
#save
sava_models_dir = 'models'
model_dict ={}
model_dict['entropy'] = entropy_tree_ins
model_dict['gini'] = gini_tree_ins

utils.save_obj(model_dict, os.path.join(sava_models_dir , 'user_tree'))

#load
model_dict = utils.load_obj(os.path.join(sava_models_dir , 'user_tree'))
model_dict.keys()
entropy_model_ins2 = model_dict['entropy']
gini_model_ins2 = model_dict['gini']

## check loaded model
#entropy
entropy_tree_pred2, entropy_tree_pred_prob2 = \
    entropy_model_ins2.predict(test, entropy_tree)
#gini
gini_tree_pred2, gini_tree_pred_prob2 = \
    gini_model_ins2.predict(test, gini_tree)

#performance check
entropy_tree_perform2  = \
        utils.perform_check(test['target'],\
        entropy_tree_pred2, entropy_tree_pred_prob2, \
        entropy_model_ins2.NUM_CLASSES, entropy_model_ins2.CLASS_DICT_, average='micro')
gini_tree_perform2  = \
        utils.perform_check(test['target'],\
        gini_tree_pred2, gini_tree_pred_prob2, \
        gini_model_ins2.NUM_CLASSES, gini_model_ins2.CLASS_DICT_, average='micro')

print(perform_base_str .format('entropy(saved model)\n\t', *np.round(np.array(entropy_tree_perform), 3)))
print(perform_base_str .format('entropy(loaded model)\n\t', *np.round(np.array(entropy_tree_perform2), 3)))
print(perform_base_str .format('gini(saved model)\n\t', *np.round(np.array(gini_tree_perform), 3)))
print(perform_base_str .format('gini(loaded model)\n\t', *np.round(np.array(gini_tree_perform2), 3)))


"""
########################################################################
Visualization
"""
import graphviz
graph_dir = 'graph'
#entropy
node, edge = entropy_tree_ins.graph.tree_to_graph(entropy_model_ins2.graph_tree)
entropy_tree_graph = graphviz.Source(node + edge+'\n}')
#gini
node, edge = gini_tree_ins.graph.tree_to_graph(gini_model_ins2.graph_tree)
gini_tree_graph = graphviz.Source(node + edge+'\n}')

"""
########################################################################
Define a new splitting criterion
"""
#tsallis entropy
tsallis_model_ins = utr(n_samples, MAX_DEPTH, params = ['tsallis', 2])
tsallis_tree, tsallis_graph_tree = \
    tsallis_model_ins.fit(train, target_attribute_name = "target")
# predict
tsallis_tree_pred, tsallis_tree_pred_prob = \
    tsallis_model_ins.predict(test, tsallis_tree)

## performance check
tsallis_tree_perform  = \
        utils.perform_check(test['target'],\
        tsallis_tree_pred, tsallis_tree_pred_prob, \
        tsallis_model_ins.NUM_CLASSES, tsallis_model_ins.CLASS_DICT_, average='micro')
print(perform_base_str .format('tsallis\n\t', *np.round(np.array(tsallis_tree_perform), 3)))
