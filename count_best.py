# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 11:51:32 2016

@author: fox
"""
import pickle

def calculate_best(reservoir_size=1000,corpus=373,instance=10):

    file_name='outputs/predictions/'+str(reservoir_size)+'res/sentence_error_index.pkl'
    with open(file_name) as f:
        sent_index_dict=pickle.load(f)

    common_errors = set.intersection(*[set(sent_index_dict[i+1]) for i in range(instance)])
    print common_errors

    print ''+str(len(common_errors))


    return (float(len(common_errors))/corpus)*100


print calculate_best(reservoir_size=500)



