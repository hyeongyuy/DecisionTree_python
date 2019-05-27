# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 04:39:33 2018

@author: hyeongyuy
"""

import pandas as pd
import numpy as np

class splitCrit(object):
    def __init__(self, min_samples, criterion):
        self.MIN_SAMPLES = min_samples
        self.CRITERION = criterion
        self.CRITERION_LIST = ['gini', 'entropy']
                      
    def mk_p_list(self, values):
        elements, counts = np.unique(values, return_counts = True)
        sum_c = np.sum(counts)
        return [counts[i]/sum_c for i in range(len(elements))]

    def homogeneity(self, p_list):
        #elements, counts = np.unique(target_col,return_counts = True)
        if 'gini' in self.CRITERION:
            homogeneity_ =  1 - np.sum([p**2 for p in p_list])
            return homogeneity_
        elif 'entropy' in self.CRITERION:
            homogeneity_ = -np.sum([p * np.log2(p) for p in p_list])
            return homogeneity_

    def split_criteria(self, left, right, target_values):
        bf_split = self.homogeneity(self.mk_p_list(target_values))

        left_ratio = np.sum(left) /len(target_values)
        right_ratio = 1 - left_ratio
        left_node_homog = (left_ratio) * \
            self.homogeneity(self.mk_p_list(target_values[left]))
        right_node_homog = (right_ratio) * \
            self.homogeneity(self.mk_p_list(target_values[right]))
        aft_split = np.nansum([left_node_homog, right_node_homog])

        return bf_split - aft_split

    def get_feature_info(self, data, target_attribute_name):
        feature = data.columns[data.columns != target_attribute_name]
        dtype_dict = {}
        value_dict = {}        
        cand = []
        for f in feature:
            if np.issubdtype(data.loc[:, f].dtype, np.number):
                dtype_dict[f] = 'n'
                value_dict[f] = data.loc[:,f].values
                pre = np.unique(value_dict[f])[1:]
                post = np.unique(value_dict[f])[:-1]
                c_values = (pre + post)/2
                for c in c_values:
                    cand.append((f, c))
            else:
                dtype_dict[f] = 'c'
                value_dict[f] = data.loc[:,f].values
                for c in np.unique(value_dict[f]):
                    cand.append((f, c))

        return dtype_dict, value_dict, cand

    def best_split(self, data, target_attribute_name):
        base_gain=0
        slt_dtype=''
        best_cut=None
        best_feature=''
        left_node_sub_data, right_node_sub_data = \
            pd.DataFrame(columns = data.columns), pd.DataFrame(columns = data.columns)

        target_values =data[target_attribute_name].values
        dtype_dict, value_dict, cand = \
            self.get_feature_info(data, target_attribute_name)
        
        for c in cand:
            dtype = dtype_dict[c[0]]
            feature_value = value_dict[c[0]]
            if dtype =='n':
                left_condtion , right_condtion = \
                    feature_value < c[1], feature_value >= c[1]
            else:
                left_condtion , right_condtion = \
                    feature_value != c[1], feature_value == c[1]

            if (np.sum(left_condtion) >= self.MIN_SAMPLES) \
                    and (np.sum(right_condtion) >= self.MIN_SAMPLES):
                
                gain = self.split_criteria(left_condtion, right_condtion, target_values)

                if (gain > base_gain):
                    base_gain = gain
                    slt_dtype = dtype
                    best_cut = c[1]
                    best_feature = c[0]
                    left_node_sub_data = data.loc[left_condtion, : ]
                    right_node_sub_data = data.loc[right_condtion, : ]

        return slt_dtype, best_cut, best_feature, left_node_sub_data, \
             right_node_sub_data


class baselineSplitCrit(splitCrit):
   # #[Default] 
   # def __init__(self, min_samples, params):
   #     criterion = params[0]
   #     super(baselineSplitCrit, self).__init__(min_samples, criterion)
   #     assert self.CRITERION in self.CRITERION_LIST, \
   #          '{} is not defined criterion. criterion list : {}'.\
   #                    format(self.CRITERION, self.CRITERION_LIST)
    #[Add tsallis entropy]
    def __init__(self, min_samples, params):
        super(baselineSplitCrit, self).__init__(min_samples, params[0])
        self.CRITERION_LIST += ['tsallis', 'tsallis_GR', 'entropy_GR']
        assert self.CRITERION in self.CRITERION_LIST, \
            '{} is not defined criterion. criterion list : {}'.\
                      format(self.CRITERION, self.CRITERION_LIST)
        if 'tsallis' in params[0]:
            self.Q_PARAMS = params[1]
            assert len(params) == 2, \
                'if criterion == tsallis or tsallis_GR, prams = [\'tsallis\', q]'
                      
    def homogeneity(self, p_list):
        #tsallis entropy
        if 'tsallis' in self.CRITERION :
            if self.Q_PARAMS != 1:
                homogeneity_ = (1/(1 - self.Q_PARAMS)) * \
                        (np.sum([p**self.Q_PARAMS for p in p_list]) - 1)
            else:
                homogeneity_ = -np.sum([p * np.log(p) for p in p_list])
            return homogeneity_
        else:
            return super(baselineSplitCrit, self).homogeneity(p_list)

    def split_criteria(self, left, right, target_values):
        if 'GR' in self.CRITERION:
            bf_split = self.homogeneity(self.mk_p_list(target_values))

            left_ratio = np.sum(left) /len(target_values)
            right_ratio = 1 - left_ratio
            left_node_homog = (left_ratio) * self.homogeneity(self.mk_p_list(target_values[left]))
            right_node_homog = (right_ratio) * self.homogeneity(self.mk_p_list(target_values[right]))
            aft_split = np.nansum([left_node_homog, right_node_homog])
            
            return (bf_split - aft_split) / self.homogeneity([left_ratio, right_ratio])

        else:
            return super(baselineSplitCrit, self).split_criteria(left, right, target_values)
