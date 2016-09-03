import pylab as pl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot

class PlotRoles(object):

    def __init__(self,save_pdf=True,nr_nouns=4, nr_verbs=2,file_name="plots/activations-plot",
                window=0, verbose=False, y_lim=[-2.5,2.5], no_ext_fct=True):

            super(PlotRoles,self).__init__()

            self.save_pdf=save_pdf
            self.nr_nouns=nr_nouns
            self.nr_verbs=nr_verbs
            self.file_name=file_name
            self.window=window
            self.verbose=verbose
            self.y_lim=y_lim
            self.no_ext_fct=no_ext_fct

    def get_labels(self,test_sentences_subset,verbose=False):
        """

        """
        tokenized_sentence = [self.sentences[i] for i in test_sentences_subset]
        sent_ticks = len(tokenized_sentence)*[None]

        for sent_index in range(len(tokenized_sentence)):
            sent_ticks[sent_index]=[] # ticks list for each sentence
            for word_index in range(len(tokenized_sentence[sent_index])):
                if word_index==0 or word_index==len(tokenized_sentence[sent_index])-1:
                    sent_ticks[sent_index].append("")
                else:
                    sent_ticks[sent_index].append(tokenized_sentence[sent_index][word_index])
        return (tokenized_sentence, sent_ticks)

    def plot_outputs(self,test_activations,test_sentences_subset,plot_subtitle):

        print "*** Plotting outputs *** "
        from matplotlib.font_manager import FontProperties
        fontP = FontProperties()
        fontP.set_size('small')

        (labels, lab_tick) = self.get_labels(test_sentences_subset)
        TOSpN = 3 * self.nr_verbs  #number of Total Output Signal per Noun (for AOR, it's 3 times the number of verbs)
        outputs=test_activations
        ## Plotting logic
        if self.save_pdf:
            ## Initiate object PdfPages for saving figures
            pp = PdfPages(str(self.file_name)+'_'+str(plot_subtitle)+'.pdf')
        for i in range(len(outputs)):
            ## For each sentence, plot as many graphs as the number of nouns
            for j in range(self.nr_nouns):
                pl.figure()
                pl.plot(outputs[i][:,TOSpN*j:TOSpN*(j+1)])

                #style-1 for legends
                #pl.legend(self.unique_labels[TOSpN*j:TOSpN*(j+1)],ncol=3,fancybox=True,shadow=True,bbox_to_anchor=(0.5, 1.07), loc="upper center")

                #style-1 for legends
                pl.legend(self.unique_labels[TOSpN*j:TOSpN*(j+1)],fancybox=True,prop=fontP,loc="upper left").get_frame().set_alpha(0.4)

                pl.suptitle("Sent-"+str(self.subset[i])+ ": '"+" ".join(labels[i][1:-1])+"'"+"\n"+plot_subtitle)
                pl.xticks(range(0,outputs[i].shape[0]),lab_tick[i],rotation=40)
                pl.axhline(y=0, c="brown", linewidth=1)
                a = matplotlib.pyplot.gca()
                if self.y_lim!=None:
                    a.set_ylim(self.y_lim)
                if self.save_pdf:
                    # Save figure for each plot
                    pp.savefig(bbox_inches="tight")
                pl.close()

        if self.save_pdf:
            ## Close object PdfPages
            pp.close()
        print "*** Plotting finished ***"

    def plot_array_in_file(self,root_file_name, array_, data_subset=None, titles_subset=None, plot_slice=None, title="", subtitle="", legend_=None):
        """

        inputs:
            array_: is the array or matrix to plot
            data_subset: correspond to the subset of the whole data that is treated. array_ is corresponds to this subset. /
                array_ and subset have to have the same length
            titles_subset: list of subtitles
            plot_slice: slice determining the element of array_ that will be plotted.
        """
        import mdp
        if data_subset is None:
            data_subset = range(len(array_))
        if titles_subset is None:
            titles_subset = ['' for _ in range(len(data_subset))]
            nl_titles_sub = ''
        else:
            nl_titles_sub = '\n'
        if array_==[] or array_==mdp.numx.array([]):
            import warnings
            warnings.warn("Warning: array empty. Could not be plotted. Title:"+str(title))
            return
        if plot_slice is None:
            plot_slice = slice(0,len(data_subset))
        else:
            if (plot_slice.stop-1) > len(data_subset):
                raise Exception, "The last element of the slice is out of the subset."
            subtitle = subtitle+"_slice-"+str(plot_slice.start)+"-"+str(plot_slice.stop)+"-"+str(plot_slice.step)
        ppIS = PdfPages(str(root_file_name)+str(title)+'.pdf')

        for i in range(plot_slice.stop)[plot_slice]:
            pl.figure()
            pl.suptitle(title+" "+str(titles_subset[i])+nl_titles_sub+" - seq "+str(data_subset[i])+"\n"+subtitle)
            pl.plot(array_[i])
            if legend_ is not None:
                pl.legend(legend_)
            ppIS.savefig()
            pl.close()
        ppIS.close()