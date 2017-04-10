# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:21:23 2016

This script is used to create subplots for activations of given sentences.

@author: fox
"""
import matplotlib.pyplot as plt
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle as pickle

def load_activations(corpus='373'):
    """
        returns:
            sentence_activations: list of arrays where each array is the activation of a sentence
            tokenized_sentences: list of list of tokenizes sentences
    """
    data_file='../outputs/activations/corpus-'+corpus+'.act'
    with open(data_file,'r') as f:
        data=pickle.load(f)
        sentence_activations=data[0]
        tokenized_sentences=data[1]
    return sentence_activations,tokenized_sentences

def plot_outputs(corpus='45',subplots=4,sent_index=None,sw_tick_pos=None):
    """
        subplots: No of subplots to be drawn
        sentence_order:list of sentences numbers in order to be drawn in subplots

    Note: no of subplots should be same as length of sentences_order
    """

    sent_activations,tok_sentences=load_activations(corpus=corpus)
    plt.close('all')

    fig, (ax,ax1) = plt.subplots(2,2, sharey='row')

    sent_idx=sent_index[0]
    sw_ticks=sw_tick_pos[0]
    i=1
    tok_sent=tok_sentences[sent_idx]
    for word_idx, word in enumerate(tok_sent):
        if word_idx in sw_ticks:
            tok_sent[word_idx]='(X'+str(i)+') '+ word
            i+=1

    s_act=sent_activations[sent_idx]
    sl= len(tok_sent)
    sr=s_act.reshape(sl,sw,actions,nr_roles)

    for i in range(actions):
        sr_plot= sr[:,:,i,:].reshape((sl,sw*nr_roles))
        ax[i].plot(sr_plot)
        ax[i].legend(unique_labels[:,i,:].flatten(),fancybox=True,bbox_to_anchor=(0., 1.02, 1.0, 0.102), loc=3,
           ncol=6, mode="expand", borderaxespad=0.).get_frame().set_alpha(0.4)

        if sw_ticks is not None:
            for x_tick in sw_ticks:
                ax[i].get_xticklabels()[x_tick].set_color('red')

        ax[0].set_ylabel(r'\textbf{Activations}')
        ax[i].set_ylim([-1.5,1.5])
        ax[i].set_xticks(range(0,s_act.shape[0]+1))
        tok_sent[0]=""
        tok_sent[-1]=""
        ax[i].set_xticklabels(tok_sent,rotation=45,ha='right')
        ax[i].axhline(y=0, c="brown",ls='--', linewidth=1)
        ax[i].grid(alpha=0.6,c='grey',ls='dotted')
        if i==0:
            ax[i].text(3, 0.2, r'\textbf{Meaning}: \textit{put(cross, right)}', fontsize=12)
        else:
            ax[i].text(3, 0.7, r'\textbf{Meaning}: - (-, -)', fontsize=12)


    sent_idx_2=sent_index[1]
    sw_ticks_2=sw_tick_pos[1]
    tok_sent_2=tok_sentences[sent_idx_2]
    i=1
    for word_idx, word in enumerate(tok_sent_2):
        if word_idx in sw_ticks_2:
            tok_sent_2[word_idx]='(X'+str(i)+') '+ word
            i+=1

    s_act_2=sent_activations[sent_idx_2]
    sl_2= len(tok_sent_2)
    sr_2=s_act_2.reshape(sl_2,sw,actions,nr_roles)

    for i in range(actions):
        sr_plot_2= sr_2[:,:,i,:].reshape((sl_2,sw*nr_roles))
        ax1[i].plot(sr_plot_2)

        ax1[i].set_ylim([-1.5,1.5])
        ax1[i].set_xticks(range(0,s_act_2.shape[0]+1))
        ax1[i].set_xlabel(r'\textbf{Action-'+str(i+1)+'}')
        ax1[0].set_ylabel(r'\textbf{Activations}')

        if sw_ticks_2 is not None:
            for sw_no, sw_tick_index in enumerate(sw_ticks_2):
               ax1[i].get_xticklabels()[sw_tick_index].set_color('red')

        tok_sent_2[0]=""
        tok_sent_2[-1]=""

        ax1[i].set_xticklabels(tok_sent_2,rotation=45,ha='right')
        ax1[i].axhline(y=0, c="brown",ls='--', linewidth=1)
        ax1[i].grid(alpha=0.6,c='grey',ls='dotted')
        if i==0:
            ax1[i].text(5, 0.3, r'\textbf{Meaning}: \textit{push(triangle, left)}', fontsize=12)
        else:
            ax1[i].text(8, 0.3, r'\textbf{Meaning}: \textit{push(cross, left)}', fontsize=12)

    fig.subplots_adjust(left=0.05,right=0.95, top=0.95,bottom=0.15,hspace=0.25, wspace=0.1)
    plt.show()
    fig.set_size_inches(12, 10)
    plt.savefig('/home/fox/thesis_report/src/act_analysis_2.pdf',bbox_inches='tight')

if __name__=="__main__":

    pgf_with_latex = {
        "text.usetex": True,                # use LaTeX to write all text
        "font.family": "serif",
        "font.serif":[],
        'font.size': 14,                   # blank entries should cause plots to inherit fonts from the document
        'axes.titlesize':'medium',
        "axes.labelsize": 'medium',
        "legend.fontsize": 10,               # Make the legend/label fonts a little smaller
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        }

    plt.rcParams.update(pgf_with_latex)

    sw=6
    actions=2
    nr_roles=3
    ToSpan=2*3
    roles=['P','S','L']
    TOSpN=6
    unique_labels=np.array(['X'+str(s+1)+'-'+r+str(a+1) for s in range(sw) for a in range(actions) for r in roles ])
    unique_labels=unique_labels.reshape((6,2,3))

    sent_index=[4,3] # plotting sent 5 and 4 respectively from corpus file

    sw_tick_pos=[[1,3,6],[1,3,6,8,10,13]]

    sent_act=plot_outputs(corpus='373',subplots=4,sent_index=sent_index,sw_tick_pos=sw_tick_pos)