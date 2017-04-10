# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 11:51:32 2016

@author: fox
This script is used to calculate the "best" error used in experiment 6 and 7 of the thesis.
The best generalization error here represents the percentage of sentences whose meanings were predicted
incorrectly in common by all 10 model instances

The pickle file here contains the index of the sentences which were predicted incorrectly in each model instance.
"""
import pickle

def calculate_best(reservoir_size=1000,corpus=373,input_dim=20,learning_mode='SCL',instance=10):

    #file_name='outputs/input_dim/'+learning_mode +'/'+str(input_dim)+'dim_sentence_error_index.pkl'
    file_name='outputs/predictions/'+str(reservoir_size)+'res/sentence_error_index.pkl'
    with open(file_name) as f:
        sent_index_dict=pickle.load(f)

    instance_error= [len(sent_index_dict[i+1]) for i in range(instance) if i==11]
    print instance_error
    print '\n'

    common_errors = set.intersection(*[set(sent_index_dict[i+1]) for i in range(instance) if i==9])
    print common_errors

    print 'Count of Common Errors :: '+str(len(common_errors))


    return (float(len(common_errors))/corpus)*100


print calculate_best(reservoir_size=500,input_dim=50,learning_mode='SCL')



