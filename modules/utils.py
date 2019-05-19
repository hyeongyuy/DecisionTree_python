# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 04:39:33 2018

@author: hyeongyuy
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, \
precision_score, f1_score
import copy
from functools import reduce  
import operator
import pickle
import pandas as pd
from collections import Counter



def count_class(class_idx_att, n_class):
    return [sum(class_idx_att == i) for i in range(n_class)]

def setInDict(dataDict, mapList, value):
    """
    https://python-decompiler.com/article/2013-02/
    access-nested-dictionary-items-via-a-list-of-keys
    """
    def getFromDict(dataDict, mapList):
        return reduce(operator.getitem, mapList, dataDict)
    temp_dict = copy.deepcopy(dataDict)
    getFromDict(temp_dict, mapList[:-1])[mapList[-1]] = value
    return temp_dict

#model read & write
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_leaf_rule(tree_dict, rule_list, rule, leaf_info):
    tree = copy.deepcopy(tree_dict)
    k_list = list(tree.keys())

    if k_list[0] != 'Root_node':
        left, right = k_list[0], k_list[1]
        for direct in [left, right]:
            if not isinstance(tree[direct], dict):
                if leaf_info:
                    rule_list.append(rule + [direct] + [tree[direct]])
                else:
                    rule_list.append(rule + [direct])
            else:
                get_leaf_rule(tree[direct], rule_list, rule + [direct], leaf_info=leaf_info)
        return rule_list
    else:
        return [[k_list[0]] + [tree[k_list[0]]]]



def perform_check(real, pred, prob, n_class, val_idx_dict, average = 'macro'):
    """
    #http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    #sklearn.metrics.roc_auc_score
    """
    try:
        real_v_num = np.array([val_idx_dict[v] for v in real.values.reshape(-1)])
        pred_v_num = np.array([val_idx_dict[v] for v in pred.values.reshape(-1)])
    except AttributeError:
        real_v_num = np.array([val_idx_dict[v] for v in real])
        pred_v_num = np.array([val_idx_dict[v] for v in pred])

    if n_class ==2:
        accr = accuracy_score(real_v_num, pred_v_num)
        recall = recall_score(real_v_num, pred_v_num)
        precision = precision_score(real_v_num, pred_v_num)
        f1 = f1_score(real_v_num, pred_v_num)
        auc = roc_auc_score(real_v_num, prob['1'])

    else:
        prob_v = prob.astype(float)
        accr = accuracy_score(real_v_num, pred_v_num)
        recall = recall_score(real_v_num, pred_v_num, average = average, \
                            labels = np.unique(real_v_num)) 
        precision = precision_score(real_v_num, pred_v_num, \
                            average = average, labels = np.unique(real_v_num))
        f1 = f1_score(real_v_num, pred_v_num, average = average, \
                            labels =  np.unique(real_v_num))

        real_v_dummy = []
        for v in real:
            zeros = np.zeros(n_class) 
            zeros[val_idx_dict[v]] = 1
            real_v_dummy.append(zeros)
        real_v_dummy = np.array(real_v_dummy)
        auc = roc_auc_score(real_v_dummy, prob_v, average = average)

    return [accr, recall, precision, f1, auc]
    
def recur_split(test, split_rule_list, idx=0, n_class=0):
    df= copy.deepcopy(test)
    cont_cond = ['>=', '<']
    cat_cond = ['==', '!=']
    if split_rule_list[0] == 'Root_node':
        print("""Untrained model.(Only root node.)
        The index for the input data and the ratio value
        for each class of the target variable are returned.""")
        if n_class == 0:
            return df.index
        else:
            cnt_list = np.array(count_class(split_rule_list[-1][-1], n_class))
            pred_value = cnt_list/sum(cnt_list)
            return df.index, pred_value
    
    if idx == len(split_rule_list) -1:
        if n_class == 0:
            return df.index
        else:
            cnt_list = np.array(count_class(split_rule_list[-1][-1], n_class))
            pred_value = cnt_list/sum(cnt_list)
            return df.index, pred_value
            
    else:
        att, cond, value = split_rule_list[idx].split()
        if cond == cont_cond[0]:
            sub_set = df.loc[df[att] >= float(value),:]
        elif cond == cont_cond[1] :
            sub_set = df.loc[df[att] < float(value),:]
        elif cond == cat_cond[0]:
            sub_set = df.loc[df[att] == value,:]
        else:
            sub_set = df.loc[df[att] != value,:]
        return recur_split(sub_set, split_rule_list, idx + 1, n_class=n_class)

def get_usrt_info(df, tree_ins, tree_model, cut_depth, target_att= 'target'):
    tree_rule = get_leaf_rule(tree_model, [], [], leaf_info = True)
    if cut_depth == -1:
        info_list =  []
        for s_rule in tree_rule:
            sidx = recur_split(df, s_rule)
            node_df = df.loc[sidx,:]
            depth = len(s_rule)
            if len(node_df) != 0:
                #If there is no value that satisfies this rule, it is ignored.
                simple_max_prob = max(Counter(node_df[target_att]).values())/len(node_df)
                sample_ratio = len(node_df)/len(df)
                info_list.append([depth-1, simple_max_prob, sample_ratio])
        return pd.DataFrame(info_list, \
                columns=['depth', 'max_prob', 'sample_ratio']).sort_values('depth')
    
    else:
        max_simple_rules = [i for i in tree_rule if len(i[:-1]) <= cut_depth]
        simple_idx = []
        for s_rule in max_simple_rules:
            sidx = recur_split(df, s_rule)
            simple_idx.extend(list(sidx))
        
        N_simple_rule = len(max_simple_rules)
        simple_df = df.loc[simple_idx,:]
        simple_max_prob = np.mean(np.max(tree_ins.predict(simple_df, tree_model)[1],1))
        simple_ratio = len(simple_df)/len(df)
        if tree_rule[0] == 'Root_node':
            return 0,0,0
        else:
            return simple_ratio, simple_max_prob, N_simple_rule