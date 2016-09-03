"""
Created on Sun Mar 27 13:40:22 2016

@author: fox
"""
#! /usr/bin/env python
import mdp
import csv
import itertools
import time
import requests
import base64
import numpy as np
from Oger.nodes import LeakyReservoirNode
from Oger.evaluation import n_fold_random,leave_one_out
from Oger.utils import rmse,make_inspectable
from copy import deepcopy
from random import shuffle
from tra_error import ThematicRoleError,keep_max_for_each_time_step_with_default
from reservoir_weights import generate_sparse_w, generate_sparse_w_in
from Oger.nodes import RidgeRegressionNode
from tra_plot import PlotRoles
try:
    import cPickle as pickle
except:
    import pickle as pickle

class ThematicRoleModel(ThematicRoleError,PlotRoles):

    def __init__(self,corpus='373',subset=range(0,462),reservoir_size= 1000,input_dim=50,save_predictions=False,
                 spectral_radius=1.9, input_scaling= 2.3, bias_scaling=0,leak_rate=0.1,plot_activations=False,
                 ridge = 1e-6, _instance=0, n_folds = 0, seed=2, learning_mode='SFL', verbose=True):

                 self.COPRUS_W2V_MODEL_DICT='data/corpus_word_vectors/corpus-word-vectors-'+str(input_dim)+'dim.pkl'

                 #all the parameters required for ESN
                 self.input_dim=input_dim
                 self.reservoir_size = reservoir_size
                 self.spectral_radius = spectral_radius
                 self.input_scaling = input_scaling
                 self.seed=seed
                 self.bias_scaling=bias_scaling
                 self.subset=subset
                 self.ridge = ridge
                 self.leak_rate=leak_rate
                 self._instance=_instance
                 self.corpus=corpus
                 self.verbose=verbose
                 self.save_predictions=save_predictions
                 self.plot_activations=plot_activations

                 if learning_mode=='SFL':
                    self.teaching_start='<end>'
                 elif learning_mode=='SCL':
                    self.teaching_start='<start>'

                 self.n_folds=n_folds

                 subjects=6
                 actions=2
                 roles=['P','S','L']

                 self.unique_labels=['X'+str(s+1)+'-'+r+str(a+1) for s in range(subjects) for a in range(actions) for r in roles ]

                 #load raw sentneces and labels from the files and compute several other meta info.
                 self.sentences,self.labels=self.__load_corpus(corpus_size=corpus,subset=subset)

                 #Shuffle sentence words and not labels: same as exp.6 of xavier's paper
                 #for sent in self.sentences:
                 #    shuffle(sent)

                 self.labels_to_index=dict([(label,index) for index, label in enumerate(self.unique_labels)])
                 self.sentences_len=[len(sent) for sent in self.sentences]
                 self.max_sent_len=max(self.sentences_len) #calculate max sentence length
                 self.sentences_offsets=[self.max_sent_len - sent_len for sent_len in self.sentences_len]
                 self.X_data,self.Y_data=self.__generate_training_data() # generate input data for ESN

    def __load_corpus(self,corpus_size='373',subset=range(0,462)):
        file_name='data/corpus_'+str(corpus_size)+'_parsed.txt'
        sentences=[None]*len(subset)
        labels=[None]*len(subset)
        with open(file_name,'rb') as fh:
            i=0
            for index,line in enumerate(fh):
                if index in subset:
                   sentences[i],labels[i]=self.__process_line(line)
                   i+=1
        return sentences,labels

    def __process_line(self,line):
        s_line=line.strip().split('#')
        sentence=s_line[0].strip()
        label=s_line[1].strip()
        sentence_words= sentence.split()
        # insert two new tokens to mark the beginning and end of sentence i.e. <start>, <end>
        sentence_words.insert(0,'<start>')
        sentence_words.insert(len(sentence_words),'<end>')
        sentence_labels= label.split()
        return sentence_words,sentence_labels

    def __generate_sentence_matrix(self,tokenized_sentence,sent_index):
        """
            tokenized_sentence:a list containing tokenized sentence where each element of list is a word
            i.e. ['dog','ate','the','bone']

            returns
                a matrix of dimension (sentence length * word2vec dimension)
        """
        sentence_matrix=mdp.numx.zeros((self.sentences_len[sent_index],self.input_dim))
        for idx,word in enumerate(tokenized_sentence):
            sentence_matrix[idx]=self.get_word_vector(word.lower())
        return sentence_matrix

    def __generate_label_matrix(self,tokenized_labels,sent_index,start='<start>'):
        """
            tokenized_labels: tokenized list of a labels index
                i.e.
            index: index of sentence for which labels are given
            start: when to start giving teacher labels wrt to sentence, use 0 to start for the beginning
                    of sentence, 'end' to present at the end of sentence
            returns:
                a matrix of dimension (sentence length * len(self.unique_labels))
        """
        teaching_start= self.sentences_len[sent_index]-2 if start=='<end>' else 1

        # activate only the labels which are present in the sentence
        binary_label_array=mdp.numx.zeros((self.sentences_len[sent_index], len(self.unique_labels)))
        binary_label_array[:]=-1
        for lbl in tokenized_labels:
            binary_label_array[teaching_start:,self.unique_labels.index(lbl)]=1
        return binary_label_array

    def generate_x_data(self,sentences):
        '''
            sentence: list of list of tokenized sentences
            returns:
                x_data: create a list of sentence arrays where each array corresponds to a sentence and have shape (no of words,word2vec vector dimension)
        '''
        x_data=[self.__generate_sentence_matrix(sentence,sent_index) for sent_index,sentence in enumerate(sentences)]
        return x_data

    def __generate_y_data(self,labels):
        '''
            labels: list of list of labels for training sentences
            returns:
                y_data: create a list of labels array for the corresponding sentence of dimensions (no of words, len of unique_labels)
        '''
        y_data=[self.__generate_label_matrix(label,sent_index,start=self.teaching_start) for sent_index,label in enumerate(labels)]
        return y_data

    def __generate_training_data(self):
        '''
            Generate the sequences for each sentences and labels to be used as input and output in ESN

        '''

        # Check if the w2v converted data format of raw sentence is available in the pkl file if yes then read from pickle file
        # else load word2vec model and generate a pickle file for further loading

        with open(self.COPRUS_W2V_MODEL_DICT,'r') as f:
            print 'Please Wait!! Loading data from file...'
            self.w2v_model=pickle.load(f)
            print 'Data Loaded Successfully.'

        x_data=self.generate_x_data(sentences=self.sentences)
        y_data=self.__generate_y_data(labels=self.labels)

        return (x_data, y_data)

    def initialize_esn(self,verbose=False):
        #generate sparse reservoir weights and input weights, The results are good with sparse weights
        w_r=generate_sparse_w(output_size=self.reservoir_size,specrad=self.spectral_radius,seed=self.seed)
        w_in=generate_sparse_w_in(output_size=self.reservoir_size,input_size=self.input_dim,scaling=self.input_scaling,seed=self.seed)
        w_bias=generate_sparse_w_in(output_size=1,input_size=self.reservoir_size,scaling=self.bias_scaling,seed=self.seed)

        ## Instansiate reservoir node, read-out and flow
        self.reservoir = LeakyReservoirNode(nonlin_func=mdp.numx.tanh,input_dim=self.input_dim,output_dim=self.reservoir_size,
                                            leak_rate=self.leak_rate,w=w_r,w_in=w_in,w_bias=w_bias,_instance=self._instance)

        self.read_out = RidgeRegressionNode(ridge_param=self.ridge, use_pinv=True, with_bias=True)
        self.flow = mdp.Flow([self.reservoir, self.read_out],verbose=self.verbose)

    def trainModel(self,training_sentences,training_labels):
        '''
            inputs:
                training_sentences: Sentences on which ESN will be trained (list of arrays)
                training_labels: labels for corresponding training_sentences (list of arrays)
            returns:
                A copy of flow trained of the training_sentences
        '''
        f_copy=deepcopy(self.flow) # create a deep copy of initial flow for current train-test set
        if self.n_folds==0:
            f_copy=self.flow
        data=[training_sentences, zip(training_sentences,training_labels)]
        f_copy.train(data)
        return f_copy

    def testModel(self,f_copy,test_sentences,fold=0):
        '''
            f_copy: A copy of trained flow
            test_sentences= a list of senteces for testing

            return:
                list of arrays where each array is activation of the test sentence
        '''

        test_sentences_activations=[]

        for sent_index in range(len(test_sentences)):
            test_sentences_activations.append(f_copy(test_sentences[sent_index]))

        return test_sentences_activations

    def apply_nfold(self):
        """
        Split the data into training and test data depending on the n_folds
        return:
            train_indices , test_indices: list of arrays containg indicies for training and testing correponding to folds
        """
        if self.n_folds==0 or self.n_folds==1 or self.n_folds is None:
             train_indices=[range(len(self.sentences))] # train on all sentences
             test_indices=train_indices
        elif self.n_folds < 0: # if negative or 1
             train_indices, test_indices = leave_one_out(len(self.sentences))
        else:
             train_indices, test_indices = n_fold_random(n_samples=len(self.sentences),n_folds=self.n_folds)

        return train_indices, test_indices

    def execute(self,verbose=False,train_sent_indices=None,test_sent_indices=None):

        #instansiate the error and plot objects specified as parents class
        super(ThematicRoleModel,self).__init__()

        #obtain the training and test sentences by applying n_folds
        if train_sent_indices is None or test_sent_indices is None:
            train_indices, test_indices=self.apply_nfold()
        else:
            train_indices=train_sent_indices
            test_indices=test_sent_indices

        # containers to receive mean rmse, meaning and sentence error for test sentences on all the folds
        all_mean_meaning_err = []
        all_mean_sentence_err = []
        all_mean_rmse = []

        iteration = range(len(train_indices))

        #prepare a list of training and test sentences arrays based on train_indices and test_indices respectively
        for fold in iteration:
            #generating training sentences and labels data for each fold
            curr_train_sentences=[self.X_data[index] for index in train_indices[fold]]
            curr_train_labels=[self.Y_data[index] for index in train_indices[fold]]

            #generating test sentences and labels data for each fold
            curr_test_sentences=[self.X_data[index] for index in test_indices[fold]]
            curr_test_labels=[self.Y_data[index] for index in test_indices[fold]]

            test_sentences_subset=[index for index in test_indices[fold]]

            # Training:- return a flow trained on current fold training sentences
            f_copy=self.trainModel(curr_train_sentences,curr_train_labels)

            fold_meaning_error=[]
            fold_sentence_error=[]
            fold_rmse=[]

            #Testing:- collect activations of all test sentences in current fold in a list
            test_sentences_activations=self.testModel(f_copy,curr_test_sentences,fold=fold)
            if self.save_predictions:
                y_true_lbl,y_pred_lbl=self.decode_read_out_activations(y_pred=test_sentences_activations,y_true=curr_test_labels)
                self.save_test_predictions(test_sentences_subset,y_true_lbl,y_pred_lbl,fold)

            if self.n_folds==0:
                # saved activations will be used to draw graphs
                #self.save_test_activations(test_sentences_activations)
                pass

            for sent_no,sent_activation in enumerate(test_sentences_activations):
                    #compute error method returns a tuple of errors
                    errors=self.compute_error(sent_activation,curr_test_labels[sent_no])
                    meaning_error, sentence_error=errors
                    fold_meaning_error.append(meaning_error)
                    fold_sentence_error.append(sentence_error)
                    fold_rmse.append(rmse(sent_activation,curr_test_labels[sent_no]))

            all_mean_rmse.append(mdp.numx.mean(fold_rmse))
            all_mean_meaning_err.append(mdp.numx.mean(fold_meaning_error))
            all_mean_sentence_err.append(mdp.numx.mean(fold_sentence_error))
            if self.plot_activations:
                self.plot_outputs(test_sentences_activations,test_sentences_subset,plot_subtitle='')

        if verbose:
            print '\n mean rmse::',mdp.numx.mean(all_mean_rmse)
            print ' SD in mean nrmse::',mdp.numx.std(all_mean_rmse)
            print '\n mean meaning error::',mdp.numx.mean(all_mean_meaning_err)
            print ' SD in mean meaning error::',mdp.numx.std(all_mean_meaning_err)
            print '\n mean sentence error::',mdp.numx.mean(all_mean_sentence_err)
            print ' SD in mean sentence error::',mdp.numx.std(all_mean_sentence_err)

        return mdp.numx.mean(all_mean_rmse),mdp.numx.std(all_mean_rmse),\
                mdp.numx.mean(all_mean_meaning_err),mdp.numx.std(all_mean_meaning_err), \
                mdp.numx.mean(all_mean_sentence_err),mdp.numx.std(all_mean_sentence_err)

    def save_test_predictions(self,test_sent_indices,y_true_lbl,y_pred_lbl,fold):

        test_sent = [self.sentences[sent_index] for sent_index in test_sent_indices]

        file_name='outputs/predictions/corpus-'+self.corpus+'.txt'
        labels=np.array(self.unique_labels)
        with open(file_name,'a') as f:
            for sent_index, sent in enumerate(test_sent):
                f.write(35*'#'+' '+str(fold+1)+'-fold '+ 35*'#' +'\n')
                role_mask=self.Y_data[test_sent_indices[sent_index]]>0
                role_mask=role_mask[1,:]
                yt=labels[role_mask].tolist()
                yp=y_pred_lbl[sent_index]
                match= (yt==yp)
                f.write('Sentence-'+str(self.subset[sent_index])+':'+' '.join(sent) + '\n')
                f.write('True Roles :: '+' '.join(yt) +'\n')
                f.write('Pred Roles :: '+' '.join(yp) +'\n')
                f.write('CORRECT :: '+ str(match) +'\n')
                f.write('\n')

    def decode_read_out_activations(self,y_true, y_pred, threshold=0):
        """
            y_true: list of arrays of sentence true activations
            y_pred: list of arrays of sentence pred activations
            returns:
                y_true_lbl: a list of list containing true roles for each senteces
                y_pred_lbl: a list of list containing true roles for each senteces

        """

        labels=np.array(self.unique_labels)
        y_true_lbl, y_pred_lbl=[],[]
        for idx in range(len(y_true)):
            sent_true_role, sent_pred_role=[],[]
            (NVassoc_contributing_anwser, NVassoc_not_contributing_answer_but_present, NVassoc_not_present_in_sentence) = \
                self._get_NVassoc_sliced(input_signal=y_pred[idx],target_signal= y_true[idx], verbose=False)

            for nva_tuple in NVassoc_contributing_anwser + NVassoc_not_contributing_answer_but_present:
                nva_index = nva_tuple[0]

                nv_true = nva_tuple[2]
                nv_true_max_act=keep_max_for_each_time_step_with_default(input_signal=nv_true)
                nv_true_max_act=np.reshape(nv_true_max_act,3)
                mask=(nv_true_max_act > threshold)
                pred_lbl=labels[nva_index*3:nva_index*3+3]
                sent_true_role+=pred_lbl[mask].tolist()

                nv_pred=nva_tuple[1]
                nv_pred_max_act=keep_max_for_each_time_step_with_default(input_signal=nv_pred)
                nv_pred_max_act=np.reshape(nv_pred_max_act,3)
                mask=(nv_pred_max_act > threshold)
                pred_lbl=labels[nva_index*3:nva_index*3+3]
                sent_pred_role+=pred_lbl[mask].tolist()

            y_true_lbl.append(sent_true_role)
            y_pred_lbl.append(sent_pred_role)
        return y_true_lbl, y_pred_lbl

    def save_test_activations(self,test_activations):
        pkl_file='outputs/activations/corpus-'+str(self.corpus)+'.act'

        output_data=[]
        output_data.append(test_activations)
        output_data.append(self.sentences)

        with open(pkl_file, 'w') as fhandle:
            pickle.dump(output_data, fhandle)
            print 'Activation Dumped successfully.'

    def get_word_vector(self,word):
        """
            This methods returns the word embedding for the input word
        """
        try:
            url='http://127.0.0.1:5000/word2vec/model?word='+word
            response=requests.get(url).text
            response=base64.decodestring(response)
            word_vector=np.frombuffer(response,dtype=np.float32)
        except Exception:
            word_vector=self.w2v_model[word]
        return word_vector

    def grid_search(self,output_csv_name=None,progress=True,verbose=False):
        '''
            this execute method does a grid search over reservoir parameters and log the errors in a csv file w.r.t to
            gridsearch parameters
        '''
        if output_csv_name is None:
            ct=time.strftime("%d-%m_%H:%M")
            out_csv='outputs/tra-'+str(self.corpus)+'-'+\
                     str(self.reservoir_size)+'res-'+\
                     str(self.n_folds)+'folds-'+\
                     str(self.ridge)+'ridge-'+\
                     str(self.input_dim)+'w2vdim-'+\
                     ct+'.csv'
        else:
            out_csv=output_csv_name

        #dictionary of parameter to do grid search on
        #Note the parameter key should match the name with variable of this class
        gridsearch_parameters = {
                                'spectral_radius':mdp.numx.arange(1.5, 2.6, 0.1),
                                'input_scaling':mdp.numx.arange(2.0, 3.1, 0.1),
                                'leak_rate':mdp.numx.arange(0.01,0.11,0.02)
                                }
        parameter_ranges = []
        parameters_lst = []

        # Construct the parameter space
        # Loop over all nodes that need their parameters set
        for node_key in gridsearch_parameters.keys():
            # Loop over all parameters that need to be set for that node
            # Append the parameter name and ranges to the corresponding lists
                parameter_ranges.append(gridsearch_parameters[node_key])
                parameters_lst.append(node_key)

        # Construct all combinations
        param_space = list(itertools.product(*parameter_ranges))
        if progress:
            iteration = mdp.utils.progressinfo(enumerate(param_space), style='timer', length=len(param_space))
        else:
            iteration = enumerate(param_space)

        # Loop over all points in the parameter space i.e for each parameters combination
        with open(out_csv,'wb+') as csv_file:
            w=csv.writer(csv_file,delimiter=';')
            csv_header=['S.No','RMSE','std. RMSE','Meaning_Error','std. Meaning Error', 'Sentence_Error','std. Sentence Error']
            csv_header+=[param for param in parameters_lst]
            w.writerow(csv_header)

            for paramspace_index_flat, parameter_values in iteration:
                # Set all parameters of all nodes to the correct values
                for parameter_index, parameter in enumerate(parameters_lst):
                    # Add the current node to the set of nodes whose parameters are changed, and which should be re-initialized
                    self.__setattr__(parameter,parameter_values[parameter_index])

                # Re-initialize esn
                self.initialize_esn()

                # Do the validation and get the errors for each paramater combination
                errors = self.execute()

                # Store the current errors in the respective errors arrays for a param combination
                mean_rmse=errors[0]
                std_rmse=errors[1]
                mean_meaning_error =  errors[2]
                std_meaning_error =  errors[3]
                mean_sentence_error =  errors[4]
                std_sentence_error =  errors[5]

                row=[paramspace_index_flat+1,mean_rmse,std_rmse, mean_meaning_error, std_meaning_error,mean_sentence_error,std_sentence_error]
                row+=list(param_space[paramspace_index_flat])
                w.writerow(row)

if __name__=="__main__":
    '''
        learning mode can be:
            SFL : sentence final learning
            SCL : sentence continous learning
    '''
    start_time = time.time()
    learning_mode='SCL' # 'SCL'

    #************************** Corpus 462 ************************************
    corpus='373'
    sub_corpus_per=100
    subset=range(0,373)
    n_folds=10
    iss= 2.5 #2.5 for SCL # 2.3 for SFL
    sr= 2.4 #2.4  for SCL # 2.2 for SFL
    lr= 0.07 #0.07 for SCL # 0.13 for SFL

    #******************* Initialize a Model ***********************************

    model = ThematicRoleModel(corpus=corpus,input_dim=50,reservoir_size=500,input_scaling=iss,spectral_radius=sr,
                            leak_rate=lr,ridge=1e-3,subset=subset,n_folds=n_folds,verbose=True,seed=1,
                            plot_activations=False,save_predictions=False,learning_mode=learning_mode)
    model.initialize_esn()

    #******************* Execute a Model with multiple Reservoir instances ***********************************

    model_instances=1 # No. of instances of reservoir

    if model_instances > 1:
        # Name of file where to save the execution results
        ct=time.strftime("%d-%m_%H:%M")
        out_csv='outputs/'+str(model_instances)+'instances-tra-'+str(model.corpus)+'-'+\
                     str(sub_corpus_per)+'subcorpus-'+\
                     str(model.reservoir_size)+'res-'+\
                     str(model.n_folds)+'folds-'+\
                     str(model.ridge)+'ridge-'+\
                     str(model.input_dim)+'w2vdim-'+learning_mode+''+\
                     ct+'.csv'

        train_indices,test_indices=model.apply_nfold()
        with open(out_csv,'wb+') as csv_file:
            w=csv.writer(csv_file, delimiter=';')
            csv_header=['Instance','RMSE','std. RMSE','Meaning_Error','std. Meaning_Error', 'Sentence_Error','std. Sentence_Error']
            w.writerow(csv_header)
            for instance in range(model_instances):
                print "***************** Instance: "+str(instance+1)+"***********************"
                # re-initialize esn
                model.initialize_esn()
                #execute the model
                rmse_error,std_rmse,me,std_me,se,std_se= model.execute(verbose=True,train_sent_indices=train_indices,test_sent_indices=test_indices)
                row=[instance+1,rmse_error,std_rmse, me, std_me,se,std_se]
                w.writerow(row)
    else:
        model.execute(verbose=True)

    #model.grid_search()
    #model.flow.save('outputs/flow-'+corpus+'.flow')
    end_time = time.time()
    print '\nTotal execution time : %s min '%((end_time-start_time)/60)