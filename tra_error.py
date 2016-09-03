import mdp
import numpy as np
import Oger
import csv

def keep_max_for_each_time_step_with_default(input_signal, default_min_value=-1):
    # get the maximum for each line (= each time step)
    m_arr = np.max(input_signal, axis=1)
    m_arr = np.atleast_2d(m_arr).T
    m_mat = np.concatenate([m_arr for _ in range(input_signal.shape[1])],axis=1)
    # keep only the maximum in each line / for each time step, rest is 0
    return (input_signal >= m_mat)*input_signal + (input_signal < m_mat)*default_min_value

def threshold_and_take_max_before_error(input_signal, target_signal, error_measure, thresh, default_min_value=-1):
    """
    First keep only the maximum for each line (i.e. keep the maximum for each time step)
    then applies a threshold to input_signal and target_signal,
    finally determines the error using the error_measure function.
    The threshold is estimated as the mean of the target_signal maximum and minimum unless a threshold 'thresh' is specified
    """
    if thresh == None:
        thresh = (max(target_signal) + min(target_signal)) / 2.

    # check if default_min_value is coherent with the threshold
    if default_min_value >= thresh:
        raise Exception, 'the default value applied after the max is taken is equal or superior to the threshold.'

    input_signal_max = keep_max_for_each_time_step_with_default(input_signal, default_min_value=default_min_value)
    return error_measure(input_signal_max>thresh, target_signal>thresh)

class ThematicRoleError(object):
    """
    Specific language error: measure defined for a special language task on thematic role assignment.
    """
    def __init__(self,error_measure=Oger.utils.loss_01, threshold=0, verbose=False):
        """
        Inputs:
            - error_measure: method used to compute the error on the given interval defined by self.time_step_slice
            - threshold: threshold used by the error_measure to discriminate for binary response.
        """
        super(ThematicRoleError,self).__init__()
        self.error_measure = error_measure
        self.threshold = threshold


        self.xa_pairs = [('X1','A1'), ('X1','A2'), ('X2','A1'), ('X2','A2'),
                         ('X3','A1'), ('X3','A2'), ('X4','A1'), ('X4','A2'),
                         ('X5','A1'), ('X5','A2'), ('X6','A1'), ('X6','A2')]

        self.verbose = verbose
        self.__initialize_error_algorithm()

    def __initialize_error_algorithm(self):
        """
        Defining:
            time_step_slice: start and stop slice for evaluate output signal
            max_answers: maximum number of active outputs
        """

        self.time_step_slice = slice(-1,None)
        self.max_answers = self._get_max_answers()

    def _get_max_answers(self):
        """
        Return the maximal number of answer for one sentence.
            It corresponds to the maximal length of elements in 'l_teacher'.
        !!! Warning: to be accurate, 'l_teacher' needs to corresponds to the subset selected, and not the full set of data.
        """
        return max([len(x) for x in self.labels])


    def _get_XAassoc_sliced(self, input_signal, target_signal, verbose=False):
        """
        Output:
            (XAassoc_admiting_anwser, XAassoc_not_present_in_sent)
            Each element of this tuple is a list. Each element of a list is a 3-tuple:
                - 1st: index of the Noun-Verb association in the 'self.xa_pairs' list of tuples
                - 2nd: sub-matrix (sub-numpyarray) of the input_signal that will be used by error_measure
                - 3rd: sub-matrix (sub-numpyarray) of the teacher_signal that will be used by error_measure
        !Warning: this method rely on a specific way of coding the output signal.
            If this coding is changed, you may have to recode most of this method.
        Notation:
            XAa: SW-A association
        NB: some precisions on what is the problem that have to deal the algorithm:
            It should infer which XAa-s (SW-A association) are present in the sentence
            -- this means inferring how many Open Class Words (~Nouns) and how many meanings
             there is by looking to the teacher output signals --,
            In this method we do it by inferring from the target_signal (because the input corpus is not available).
        """
        ## The Noun-Verb associations (i.e. full PSL for a given SW respect to a given verb)
        ##    that admit answer are the XAassoc that have at the same time
        ##    Noun and Verb with one teacher at 1 (one of the AOR for the XAassoc).
        ##    The different XAassoc possible are given by self.xa_pairs.
        if verbose:
            print "<<< Beginning method _get_XAassoc_sliced():"
            print "self.time_step_slice", self.time_step_slice

        ## creating XAassoc
        xa_contributing_and_present_in_sentence = [] # what all sw-action pair are present in the sentence
        XAassoc_contributing_anwser = [] # what all sw-action are above threshold i.e comptetor for final role
        XAassoc_not_contributing_answer = [] # what all sw-action are below threshold and not present in the sentence i.e. not a competetor for role assingment

        # Finding which SW, Action and association of sw-action are present in the sentence.
        # Here 3 at the end represent the no of roles for a sw wrt to action
        for idx in range(0,len(self.unique_labels),3):
            XAindex = int(idx/3)
            XAassoc_tuple = (XAindex, input_signal[self.time_step_slice, idx:idx+3], target_signal[self.time_step_slice, idx:idx+3])

            #check if activatiion of a role wrt to noun and corresoponding verb is above threshold or not
            if mdp.numx.any(target_signal[self.time_step_slice, idx:idx+3] > self.threshold): # the non-1 signal could be 0 or -1 (so np.any() is not sufficient)
                # add the current XAassoc to the list
                XAassoc_contributing_anwser.append(XAassoc_tuple)
                # add the noun and the verb to the list of Noun and Verb present in the sentence (will be used later) there will be duplicate, but this is not an issue
                xa_contributing_and_present_in_sentence.extend(self.xa_pairs[XAindex])
            else:
                XAassoc_not_contributing_answer.append(XAassoc_tuple)

        ## create list of XAassoc_not_contributing_answer but present in the sentence
        XAassoc_not_contributing_answer_but_present = []
        XAassoc_not_present_in_sentence = []

        # for each XAassoc_tuple in XAassoc_not_contributing_answer
        for XAassoc_tuple in XAassoc_not_contributing_answer:
            # if its Noun (i.e. self.xa_pairs[XAassoc_tuple[0]][0]) or its Verb (i.e. self.xa_pairs[XAassoc_tuple[0]][1]) is not present in the sentence
            if xa_contributing_and_present_in_sentence.count(self.xa_pairs[XAassoc_tuple[0]][0])==0 \
                or xa_contributing_and_present_in_sentence.count(self.xa_pairs[XAassoc_tuple[0]][1])==0:
                # put it in a new list containing the XAa not present in sentence
                XAassoc_not_present_in_sentence.append(XAassoc_tuple)
                # if N and V are present in the XAa
            else:
                # add it to the new list
                XAassoc_not_contributing_answer_but_present.append(XAassoc_tuple)

        if (len(XAassoc_contributing_anwser)+len(XAassoc_not_contributing_answer_but_present)+len(XAassoc_not_present_in_sentence)) != len(self.xa_pairs):
            raise Exception, "The number of Noun-Verb association is not correct. Should be "+str(len(self.xa_pairs))

        return (XAassoc_contributing_anwser, XAassoc_not_contributing_answer_but_present, XAassoc_not_present_in_sentence)

    def compute_error(self, input_signal, target_signal):
        """
        Inputs:
            input_signal: readout activity of ESN for a sentence
            target_signal: teacher output used for the supervised training corresponding to sentence
        Outputs:
            (mean of meaning errors, mean of sentence errors,
                number of erroneous Noun/action, number of pertinent Noun/action, list of XAa that are correct, list of XAa that are incorrect)
        The 2nd line gathers results not used in default mode. Use this information to know more on errors.
        """

        ## initialization
        perf_asso_adm_answ = [] # performance of XAa admitting answer
        (XAassoc_contributing_anwser, XAassoc_not_contributing_answer_but_present, XAassoc_not_present_in_sentence) = \
            self._get_XAassoc_sliced(input_signal, target_signal, verbose=False)

        if len(XAassoc_contributing_anwser)==0 and len(XAassoc_not_contributing_answer_but_present)==0:
            return 0, 0

        XAa_correct = []
        XAa_erroneous = []

        ## Computing errors and impossible states for XAa admiting answer
        for XAassoc_tuple in XAassoc_contributing_anwser:
            ## Evaluate fraction of time when the good answer if given for the 3 signal AOR at the same time
            err_answer = threshold_and_take_max_before_error(input_signal=XAassoc_tuple[1],
                                                           target_signal=XAassoc_tuple[2],
                                                           error_measure=self.error_measure,
                                                           thresh=self.threshold)
            perf_asso_adm_answ.append(1 - err_answer)
            if err_answer > 0:
                XAa_erroneous.append(XAassoc_tuple[0])
            else:
                XAa_correct.append(XAassoc_tuple[0])

        ## Computing errors and impossible states for XAa not admiting answer, but present in the sentence
        perf_asso_not_adm_answ_p = [] #performance of XAa not admiting answer, but present
        for XAassoc_tuple in XAassoc_not_contributing_answer_but_present:
            err_answer = threshold_and_take_max_before_error(input_signal=XAassoc_tuple[1],
                                                       target_signal=XAassoc_tuple[2],
                                                       error_measure=self.error_measure,
                                                       thresh=self.threshold)

            perf_asso_not_adm_answ_p.append(1 - err_answer)
            if err_answer > 0:
                XAa_erroneous.append(XAassoc_tuple[0])
            else:
                XAa_correct.append(XAassoc_tuple[0])

        ## Compute means
        if perf_asso_adm_answ != []:
            if perf_asso_not_adm_answ_p != []:
                aa = perf_asso_adm_answ
                naap = perf_asso_not_adm_answ_p
                perf_asso_present = (len(aa)*mdp.numx.mean(aa) + len(naap)*mdp.numx.mean(naap)) / float((len(aa) + len(naap)))
            else:
                perf_asso_present = mdp.numx.mean(perf_asso_adm_answ)
        else:
            perf_asso_present=0
            raise Warning, "There is no answer for this sentence."

        # compute the fraction of time when all the pertinent XAa are correct (for XAa present in the sentence)
        all_output_signal = []
        all_target_signal = []

        for XAassoc_tuple in XAassoc_contributing_anwser:
            all_output_signal.append(keep_max_for_each_time_step_with_default(XAassoc_tuple[1]))
            all_target_signal.append(XAassoc_tuple[2])

        for XAassoc_tuple in XAassoc_not_contributing_answer_but_present:
            all_output_signal.append(keep_max_for_each_time_step_with_default(XAassoc_tuple[1]))
            all_target_signal.append(XAassoc_tuple[2])

        global_out_arr = mdp.numx.concatenate(all_output_signal, axis=1)
        global_target_arr = mdp.numx.concatenate(all_target_signal, axis=1)
        global_err_answer = Oger.utils.threshold_before_error(input_signal=global_out_arr,
                                                   target_signal=global_target_arr,
                                                   error_measure=self.error_measure,
                                                   thresh=self.threshold)

        ## Supplementary computations (not used in default program)
        ## Compute the number of pertinent SW (semantic word) outputs for each verb that is erroneous
        # i.e. number of erroneous XA-assoc
        total_nr_of_pertinent_SW = len(XAassoc_contributing_anwser) + len(XAassoc_not_contributing_answer_but_present)
        nr_of_erroneous_SW = int(round(total_nr_of_pertinent_SW * (1-perf_asso_present)))

        if total_nr_of_pertinent_SW != (len(XAa_erroneous)+len(XAa_correct)):
            raise Exception, "Incoherent total_nr_of_pertinent_SW. total_nr_of_pertinent_SW"+str(total_nr_of_pertinent_SW)+ \
                "\n XAa_correct="+str(XAa_correct)+ \
                "\n XAa_erroneous="+str(XAa_erroneous)
        if nr_of_erroneous_SW != len(XAa_erroneous):
            raise Exception, "Incoherent nr_of_erroneous_SW." \
                +"\nnr_of_erroneous_SW="+str(nr_of_erroneous_SW)+ \
                "\n XAa_correct="+str(XAa_correct)+ \
                "\n len(XAa_erroneous)="+str(len(XAa_erroneous))

        '''return (1 - perf_asso_present, global_err_answer,
                nr_of_erroneous_SW, total_nr_of_pertinent_SW, XAa_correct, XAa_erroneous)'''

        return 1 - perf_asso_present, global_err_answer

