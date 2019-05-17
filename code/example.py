# -*- coding: utf-8 -*-
"""
@author: hyeongyuy
"""
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import sys
workingdir = r'working'
os.chdir(os.path.join(workingdir, 'modules' ))
#sys.path.extend([os.path.abspath(".")])
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
        entropy_tree_ins.NUM_CLASSES, \
        entropy_tree_ins.CLASS_DICT_, average='micro')
gini_tree_perform  = \
        utils.perform_check(test['target'],\
        gini_tree_pred, gini_tree_pred_prob, \
        gini_tree_ins.NUM_CLASSES, gini_tree_ins.CLASS_DICT_, average='micro')

perform_base_str = '{} : ACCURACY :{}, RECALL :{}, PRECISION : {}, F1 : {}, AUC : {}'
print(perform_base_str .format('entropy\n  ', \
                               *np.round(np.array(entropy_tree_perform), 3)))
print(perform_base_str .format('gini\n  ', \
                               *np.round(np.array(gini_tree_perform), 3)))


"""
########################################################################
save & Load models
"""
#save
sava_models_dir = 'models'
if not os.path.exists(sava_models_dir):
    os.makedirs(sava_models_dir)
model_dict ={}
model_dict['entropy'] = entropy_tree_ins
model_dict['gini'] = gini_tree_ins

utils.save_obj(model_dict, os.path.join(sava_models_dir , 'user_tree'))

#load
model_dict = utils.load_obj(os.path.join(sava_models_dir , 'user_tree'))
model_dict.keys()
entropy_tree2 = model_dict['entropy']
gini_tree2 = model_dict['gini']

## check loaded model
#entropy
entropy_tree_pred2, entropy_tree_pred_prob2 = \
    entropy_tree2.predict(test, entropy_tree)
#gini
gini_tree_pred2, gini_tree_pred_prob2 = \
    gini_tree2.predict(test, gini_tree)

#performance check
entropy_tree_perform2  = \
        utils.perform_check(test['target'],\
        entropy_tree_pred2, entropy_tree_pred_prob2, \
        entropy_tree2.NUM_CLASSES, entropy_tree2.CLASS_DICT_, average='micro')
gini_tree_perform2  = \
        utils.perform_check(test['target'],\
        gini_tree_pred2, gini_tree_pred_prob2, \
        gini_tree2.NUM_CLASSES, gini_tree2.CLASS_DICT_, average='micro')

print(perform_base_str .format('entropy(saved model)\n  ', \
                               *np.round(np.array(entropy_tree_perform), 3)))
print(perform_base_str .format('entropy(loaded model)\n  ', \
                               *np.round(np.array(entropy_tree_perform2), 3)))
print(perform_base_str .format('gini(saved model)\n  ', \
                               *np.round(np.array(gini_tree_perform), 3)))
print(perform_base_str .format('gini(loaded model)\n  ', \
                               *np.round(np.array(gini_tree_perform2), 3)))

"""
########################################################################
Visualization
"""
import graphviz
graph_dir = 'graph'
#entropy
node, edge = entropy_tree_ins.graph.tree_to_graph(entropy_tree2.graph_tree)
entropy_tree_graph = graphviz.Source(node + edge+'\n}')
#gini
node, edge = gini_tree_ins.graph.tree_to_graph(gini_tree2.graph_tree)
gini_tree_graph = graphviz.Source(node + edge+'\n}')
#save graph
entropy_tree_graph.render('{}/entropy_tree'.format(graph_dir))
gini_tree_graph.render('{}/gini_tree'.format(graph_dir))

"""
########################################################################
Define a new splitting criterion
"""
#GainRatio(entropy)
entropy_GR_tree_ins = utr(n_samples, MAX_DEPTH, params = ['entropy_GR'])
entropy_GR_tree, entropy_GR_graph_tree = \
    entropy_GR_tree_ins.fit(train, target_attribute_name = "target")
#tsallis entropy
tsallis_tree_ins = utr(n_samples, MAX_DEPTH, params = ['tsallis', 2])
tsallis_tree, tsallis_graph_tree = \
    tsallis_tree_ins.fit(train, target_attribute_name = "target")
#Gainratio(tsallis)
tsallis_GR_tree_ins = utr(n_samples, MAX_DEPTH, params = ['tsallis_GR', 2])
tsallis_GR_tree, tsallis_GR_graph_tree = \
    tsallis_GR_tree_ins.fit(train, target_attribute_name = "target")
    
# predict
entropy_GR_tree_pred, entropy_GR_tree_pred_prob = \
    tsallis_GR_tree_ins.predict(test, entropy_GR_tree)
tsallis_tree_pred, tsallis_tree_pred_prob = \
    entropy_GR_tree_ins.predict(test, tsallis_tree)
tsallis_GR_tree_pred, tsallis_GR_tree_pred_prob = \
    tsallis_tree_ins.predict(test, tsallis_GR_tree)

## performance check
entropy_GR_tree_perform  = \
        utils.perform_check(test['target'],\
        entropy_GR_tree_pred, entropy_GR_tree_pred_prob, \
        entropy_GR_tree_ins.NUM_CLASSES, entropy_GR_tree_ins.CLASS_DICT_, average='micro')
        
tsallis_tree_perform  = \
        utils.perform_check(test['target'],\
        tsallis_tree_pred, tsallis_tree_pred_prob, \
        tsallis_tree_ins.NUM_CLASSES, tsallis_tree_ins.CLASS_DICT_, average='micro')
tsallis_GR_tree_perform  = \
        utils.perform_check(test['target'],\
        tsallis_GR_tree_pred, tsallis_GR_tree_pred_prob, \
        tsallis_GR_tree_ins.NUM_CLASSES, tsallis_GR_tree_ins.CLASS_DICT_, average='micro')
print(perform_base_str .format('entropy GainRatio\n  ', \
                               *np.round(np.array(entropy_GR_tree_perform), 3)))        
print(perform_base_str .format('tsallis\n  ', \
                               *np.round(np.array(tsallis_tree_perform), 3)))
print(perform_base_str .format('tsallis GainRatio\n  ', \
                               *np.round(np.array(tsallis_GR_tree_perform), 3)))


"""
#######################################################################
get leaves information
"""
#Information about all leaves
train_node_info = utils.get_usrt_info(train, tsallis_tree_ins, tsallis_tree, -1)
test_node_info = utils.get_usrt_info(test, tsallis_tree_ins, tsallis_tree, -1)
train_node_info

#Information about leaves below depth
depth = 3
train_CART_tsallis_sample_ratio, \
train_CART_tsallis_max_prob, \
CART_tsallis_N_rule \
    = utils.get_usrt_info(train, tsallis_tree_ins, tsallis_tree, depth )
test_CART_tsallis_sample_ratio, \
test_CART_tsallis_max_prob, _ \
    =utils.get_usrt_info(test, tsallis_tree_ins, tsallis_tree, depth)
base_str = '{}, sample_ratio : {}, max_probability : {}'
print('depth = ~{}'.format(depth))
print(base_str.format('train', \
          train_CART_tsallis_sample_ratio, train_CART_tsallis_max_prob))
print(base_str.format('test', \
          test_CART_tsallis_sample_ratio, test_CART_tsallis_max_prob))
