# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 04:39:33 2018

author: hyeongyuy
"""

import pandas as pd
import numpy as np
from functools import reduce
from collections import Counter

import utils as ut
import visgraph as vg
import splitcriterion as sc

class userTree(object):
    def __init__(self, min_samples, max_depth, params, simplify = True):
        ### parameters

        #terminate criteria
        self.MIN_SAMPLES = min_samples
        self.MAX_DEPTH = max_depth

        #simplify
        self.SIMPLIFY = simplify

        # call instance
        self.crt = sc.baselineSplitCrit(self.MIN_SAMPLES, params)
        
        self.graph = vg.visGraph()

    # 한 노드에서 분기 된 leaf node의 class가 같은 경우 하나의 노드로 통합함
    def recur_simplify(self, tree_org, graph_tree_org):
        bf_rule_list = ut.get_leaf_rule(tree_org, [], [], leaf_info=False)
        tree_rule = ut.get_leaf_rule(tree_org, [], [], leaf_info=True)
        graph_rule_list = ut.get_leaf_rule(graph_tree_org, [], [], leaf_info=False)

        all_rules = [tuple(i[:-2] + [i[-2].replace('!=', '==').replace('<', '>=')] + [i[-1][0]]) for i in tree_rule]
        
        all_graph_tree_rules = []
        for cond_rule, graph_rule in zip(tree_rule, graph_rule_list):
            graph_rule = graph_rule[:-1] + [graph_rule [-1].replace('!=', '==').replace('<', '>=')] + [str(cond_rule[-1][0])]
            all_graph_tree_rules.append(tuple(graph_rule))
        
        #leaf 노드 부모노드까지의 rule과 예측값이 중복되는 모든 rule들
        dup_rule = [r for r,c in Counter(all_rules).items() if c >=2]
        dup_graph_rule = [r for r,c in Counter(all_graph_tree_rules).items() if c >=2]

        for n, r in enumerate(dup_rule):
            new_parent_rule = list(r)[:-2]
            new_parent_graph_rule = list(dup_graph_rule[n])[:-2]

            sub_dict = reduce(dict.get, tuple(new_parent_rule), tree_org)
            
            if isinstance(sub_dict, dict):
                #부모노드까지의 rule과 예측값이 동일한 rule들의 sub set concat
                concat_child_df = pd.concat([i[1] for i in sub_dict.values()])
                cnt_list = ut.count_class(concat_child_df.values, self.NUM_CLASSES)

                if len(new_parent_rule) ==0: #상위노드가 없을 경우 SIMPLIFY 할 수 없음
                    tree_org = {'Root_node' : [np.argmax(cnt_list), concat_child_df]}
                    leaf_print= {'Root_node' : self.graph.node_info(cnt_list, self.N_DATA, root=True)}
                    return tree_org, leaf_print
                else:
                    tree_org = ut.setInDict(tree_org , new_parent_rule, \
                                                [np.argmax(cnt_list), concat_child_df])
                    
                    leaf_print= self.graph.node_info(cnt_list, self.N_DATA, root=False)
                    graph_tree_org = \
                        ut.setInDict(graph_tree_org , new_parent_graph_rule, leaf_print)

        aft_rule_list = ut.get_leaf_rule(tree_org, [], [], leaf_info=False)
        
        #단순화 종료 기준 : 단순화 하기 전후의 rule_list가 같을 때 종료
        if bf_rule_list == aft_rule_list :
            return tree_org, graph_tree_org
        else:
            return self.recur_simplify(tree_org, graph_tree_org)

    #############################################################################################
    def growing_tree(self, data, target_attribute_name, depth = 1):
        target_values = data[target_attribute_name]
        ####################
        #해당 노드에 들어 있는 모든 class 값이 동일할 경우 그 값을 반환
        cnt_list = ut.count_class(target_values.values, self.NUM_CLASSES)
        leaf_node_class = [np.argmax(cnt_list), target_values]

        if(depth > self.MAX_DEPTH) or (len(data)==0) or \
            (len(np.unique(target_values.values)) == 1):
            return leaf_node_class, self.graph.node_info(cnt_list, self.N_DATA, root=False)

        else:
            [slt_dtype, best_cut, best_feature, left_sub_data, right_sub_data] = \
                        self.crt.best_split(data, target_attribute_name)

            #분기 할 변수 없으면 종료
            if best_feature =='':
                if depth == 1:
                    tree_org = {'Root_node' : leaf_node_class}
                    leaf_print= {'Root_node' : self.graph.node_info(cnt_list, self.N_DATA, root=True)}
                    return  tree_org, leaf_print
                else:
                    return leaf_node_class, self.graph.node_info(cnt_list, self.N_DATA, root=False)
            
            #split(수치 : 범위, 카테고리 : 값)
            condition = ['<', '>='] if slt_dtype =='n' else ['!=', '==']

            left_subtree, graph_left_subtree = self.growing_tree(left_sub_data, \
                target_attribute_name, depth= depth +1)

            right_subtree, graph_right_subtree = self.growing_tree(right_sub_data,\
            target_attribute_name, depth= depth +1)

            tree = {}
            tree['{} {} {}'.format(best_feature, condition[0], best_cut)] = left_subtree
            tree['{} {} {}'.format(best_feature, condition[1], best_cut)] = right_subtree

            graph_tree = self.graph.get_graph_tree(best_feature, best_cut, cnt_list, condition, \
                                        [graph_left_subtree, graph_right_subtree])

        return tree, graph_tree


    #############################################################################################
    def fit(self, data, target_attribute_name, depth = 1):
        data = data.copy()
        target_values = data[target_attribute_name].values
        elements = np.unique(target_values)
        self.NUM_CLASSES = len(elements)
        self.CLASS_DICT = {i:v for i, v in enumerate(elements)}
        self.CLASS_DICT_ = {v:i for i, v in enumerate(elements)}
        self.N_DATA = len(data)
        data[target_attribute_name] = [self.CLASS_DICT_[v] for v in target_values]

        tree, graph_tree = self.growing_tree(data, target_attribute_name, depth)
        
        if self.SIMPLIFY:
            self.tree, self.graph_tree = self.recur_simplify(tree, graph_tree)
            return self.tree, self.graph_tree
        else:
            self.tree, self.graph_tree = tree, graph_tree
            return self.tree, self.graph_tree

    ############################################################################################
    def predict(self, test, tree):
        rule_list = ut.get_leaf_rule(tree, [], [], leaf_info=True)

        predict_class = pd.DataFrame(columns=["class"], index=test.index)
        predict_prob = pd.DataFrame(columns=[str(i) \
                            for i in range(self.NUM_CLASSES)], index=test.index)

        for rule in rule_list:
            idx, pred = ut.recur_split(test, rule, n_class=self.NUM_CLASSES)
            if len(idx)!=0:
                predict_class.loc[idx, 'class'] = [self.CLASS_DICT[np.argmax(pred)]]
                predict_prob.loc[idx, [str(i) \
                    for i in range(self.NUM_CLASSES)]] = [pred] * len(idx)

        return predict_class, predict_prob