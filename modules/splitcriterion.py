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
        
    def homogeneity(self, target_col):
        elements, counts = np.unique(target_col,return_counts = True)
        if self.CRITERION == 'gini':
            homogeneity_ =  1 - np.sum([(counts[i]/np.sum(counts))**2 \
                for i in range(len(elements))])
            return homogeneity_
        elif self.CRITERION == 'entropy':
            homogeneity_ = -np.sum([
                (counts[i]/np.sum(counts)) * np.log2((counts[i]/np.sum(counts))) \
                    for i in range(len(elements))])
            return homogeneity_

    def split_criteria(self, left, right, target_values):
        left_ratio = np.sum(left) /len(target_values)
        right_ratio = 1 - left_ratio
        l_w_homogeneity = (left_ratio) * self.homogeneity(target_values[left])
        r_w_homogeneity = (right_ratio) * self.homogeneity(target_values[right])
        if np.isnan(l_w_homogeneity) : l_w_homogeneity = 0
        if np.isnan(r_w_homogeneity) : r_w_homogeneity = 0
        return l_w_homogeneity, r_w_homogeneity

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
        base_information_gain=0
        slt_dtype=''
        best_cut=None
        best_feature=''
        left_node_sub_data, right_node_sub_data = \
            pd.DataFrame(columns = data.columns), pd.DataFrame(columns = data.columns)

        target_values =data[target_attribute_name].values
        dtype_dict, value_dict, cand = \
            self.get_feature_info(data, target_attribute_name)

        parants_homogen = self.homogeneity(target_values)
        for c in cand:
            dtype = dtype_dict[c[0]]
            feature_value = value_dict[c[0]]
            if dtype =='n':
                left_condtion , right_condtion = feature_value < c[1],\
                 feature_value >= c[1]
            else:
                left_condtion , right_condtion = feature_value != c[1],\
                 feature_value == c[1]

            if (np.sum(left_condtion) >= self.MIN_SAMPLES) \
                    and (np.sum(right_condtion) >= self.MIN_SAMPLES):
                left_split_criteria, right_split_criteria = \
                    self.split_criteria(left_condtion, right_condtion, target_values)
                chaild_homogen = np.sum([left_split_criteria, right_split_criteria])
                aft_information_gain = parants_homogen - chaild_homogen
                
                if (aft_information_gain > base_information_gain):
                    base_information_gain = aft_information_gain
                    slt_dtype = dtype
                    best_cut = c[1]
                    best_feature = c[0]
                    left_node_sub_data = data.loc[left_condtion, : ]
                    right_node_sub_data = data.loc[right_condtion, : ]

        return slt_dtype, best_cut, best_feature, left_node_sub_data, \
             right_node_sub_data


class baselineSplitCrit(splitCrit):
    """
    reference : 
        Wang, Yisen, and Shu-Tao Xia. "Unifying attribute splitting criteria of decision trees by Tsallis entropy." 
        2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2017.
    If you want to use only gini and entropy, \
    activate the [Default] code below and deactivate [Add tsallis entropy] code.
    """

#    #[Default] 
#    def __init__(self, min_samples, params):
#        criterion = params[0]
#        super(baselineSplitCrit, self).__init__(min_samples, criterion)
#                      
#    def homogeneity(self, target_col):
#        return super(baselineSplitCrit, self).homogeneity(target_col)

    #[Add tsallis entropy]
    def __init__(self, min_samples, params):
        criterion = params[0]
        super(baselineSplitCrit, self).__init__(min_samples, criterion)
        if params[0] == 'tsallis':
            self.Q_PARAMS = params[1]
            self.CRITERION_LIST += ['tsallis']
            assert len(params) == 2, \
                'if criterion == tsallis, prams = [\'tsallis\', q]'
                
        assert self.CRITERION in self.CRITERION_LIST, \
            '{} is not defined criterion. criterion list : {}'.\
                      format(self.CRITERION, self.CRITERION_LIST)
                      
    def homogeneity(self, target_col):
        #tsallis entropy
        if self.CRITERION == 'tsallis':
            elements, counts = np.unique(target_col,return_counts = True)
            if self.Q_PARAMS != 1:
                homogeneity_ = (1/(1 - self.Q_PARAMS)) * \
                    (np.sum([(counts[i]/np.sum(counts))**self.Q_PARAMS \
                        for i in range(len(elements))]) - 1)
            else:
                homogeneity_ = -np.sum([
                (counts[i]/np.sum(counts)) * np.log((counts[i]/np.sum(counts))) \
                    for i in range(len(elements))])
            return homogeneity_
        else:
            return super(baselineSplitCrit, self).homogeneity(target_col)

    def split_criteria(self, left, right, target_values):
        return super(baselineSplitCrit, self).split_criteria(left, right, target_values)


    def best_split(self, data, target_attribute_name):
        return super(baselineSplitCrit, self).best_split(data, target_attribute_name)
