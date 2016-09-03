import mdp
import numpy as np
import Oger

def keep_max_for_each_time_step_with_default(input_signal, default_min_value=0.0):
    # get the maximum for each line (= each time step)
    m_arr = np.max(input_signal, axis=1)
    m_arr = np.atleast_2d(m_arr).T
    m_mat = np.concatenate([m_arr for _ in range(input_signal.shape[1])],axis=1)
    # keep only the maximum in each line / for each time step, rest is 0
    return (input_signal >= m_mat)*input_signal + (input_signal < m_mat)*default_min_value

def threshold_and_take_max_before_error(input_signal, target_signal, error_measure, thresh, default_min_value=0.0):
    """
    First keep only the maximum for each line (i.e. keep the maximum for each time step)
    then applies a threshold to input_signal and target_signal,
    finally determines the error using the error_measure function.
    The threshold is estimated as the mean of the target_signal maximum and minimum unless a threshold 'thresh' is specified
    """
    # check if default_min_value is coherent with the threshold
    if default_min_value >= thresh:
        raise Exception, 'the default value applied after the max is taken is equal or superior to the threshold.'

    if thresh == None:
        thresh = (max(target_signal) + min(target_signal)) / 2.

    input_signal_max = keep_max_for_each_time_step_with_default(input_signal, default_min_value=default_min_value)
    return error_measure(input_signal_max > thresh, target_signal > thresh)

class ThematicRoleError(object):
    """
    Specific language error: measure defined for a special language task on thematic role assignment.
    """
    def __init__(self,error_measure=Oger.utils.loss_01, threshold=0.5, verbose=False):
        """
        Inputs:
            - error_measure: method used to compute the error on the given interval defined by self.time_step_slice
            - threshold: threshold used by the error_measure to discriminate for binary response.
        """
        super(ThematicRoleError,self).__init__()
        self.error_measure = error_measure
        self.threshold = threshold
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

    def _get_NVassoc_sliced(self, input_signal, target_signal, verbose=False):
        """
        Output:
            (NVassoc_admiting_anwser, NVassoc_not_present_in_sent)
            Each element of this tuple is a list. Each element of a list is a 3-tuple:
                - 1st: index of the Noun-Verb association in the 'self.nv_pairs' list of tuples
                - 2nd: sub-matrix (sub-numpyarray) of the input_signal that will be used by error_measure
                - 3rd: sub-matrix (sub-numpyarray) of the teacher_signal that will be used by error_measure
        !Warning: this method rely on a specific way of coding the output signal.
            If this coding is changed, you may have to recode most of this method.
        Notation:
            Nva: Noun-Verb association
        NB: some precisions on what is the problem that have to deal the algorithm:
            It should infer which NVa-s (Noun-Verb association) are present in the sentence
            -- this means inferring how many Open Class Words (~Nouns) and how many meanings
             there is by looking to the teacher output signals --,
            In this method we do it by inferring from the target_signal (because the input corpus is not available).
        """

        ## creating NVassoc
        n_v_contributing_and_present_in_sentence = [] # what all noun-verb pair are present in the sentence
        NVassoc_contributing_anwser = [] # what all noun-verb are above threshold i.e comptetor for final role
        NVassoc_not_contributing_answer = [] # what all noun-verb are below threshold and not present in the sentence i.e. not a competetor for role assingment

        # Finding which Nouns, Verbs and association of Noun-Verb are present in the sentence.
        # Here 3 at the end represent the no of roles for a noun wrt to v
        for idx in range(0,len(self.unique_labels),3):
            NVindex = int(idx/3)
            NVassoc_tuple = (NVindex, input_signal[self.time_step_slice, idx:idx+3], target_signal[self.time_step_slice, idx:idx+3])

            #check if activatiion of a role wrt to noun and corresoponding verb is above threshold or not
            if mdp.numx.any(target_signal[self.time_step_slice, idx:idx+3] > self.threshold): # the non-1 signal could be 0 or -1 (so np.any() is not sufficient)
                ## add the current NVassoc to the list
                NVassoc_contributing_anwser.append(NVassoc_tuple)
                # add the noun and the verb to the list of Noun and Verb present in the sentence (will be used later)
                # there will be duplicate, but this is not an issue
                n_v_contributing_and_present_in_sentence.extend(self.nv_pairs[NVindex])
            else:
                NVassoc_not_contributing_answer.append(NVassoc_tuple)

        ## create list of NVassoc_not_contributing_answer but present in the sentence
        NVassoc_not_contributing_answer_but_present = []
        NVassoc_not_present_in_sentence = []

        # for each NVassoc_tuple in NVassoc_not_contributing_answer
        for NVassoc_tuple in NVassoc_not_contributing_answer:
            # if its Noun (i.e. self.nv_pairs[NVassoc_tuple[0]][0]) or its Verb (i.e. self.nv_pairs[NVassoc_tuple[0]][1]) is not present in the sentence
            if n_v_contributing_and_present_in_sentence.count(self.nv_pairs[NVassoc_tuple[0]][0])==0 \
                or n_v_contributing_and_present_in_sentence.count(self.nv_pairs[NVassoc_tuple[0]][1])==0:
                # put it in a new list containing the NVa not present in sentence
                NVassoc_not_present_in_sentence.append(NVassoc_tuple)
                # if N and V are present in the NVa
            else:
                # add it to the new list
                NVassoc_not_contributing_answer_but_present.append(NVassoc_tuple)

        if (len(NVassoc_contributing_anwser)+len(NVassoc_not_contributing_answer_but_present)+len(NVassoc_not_present_in_sentence)) != len(self.nv_pairs):
            raise Exception, "The number of Noun-Verb association is not correct. Should be "+str(len(self.nv_pairs))

        return (NVassoc_contributing_anwser, NVassoc_not_contributing_answer_but_present, NVassoc_not_present_in_sentence)

    def compute_error(self, input_signal, target_signal):
        """
        Inputs:
            input_signal: readout activity of ESN for a sentence
            target_signal: teacher output used for the supervised training corresponding to sentence
        Outputs:
            (mean of meaning errors, mean of sentence errors,
                number of erroneous Noun/action, number of pertinent Noun/action, list of NVa that are correct, list of NVa that are incorrect)
        The 2nd line gathers results not used in default mode. Use this information to know more on errors.
        """
        input_signal=input_signal.reshape(4,2,3)
        target_signal=target_signal.reshape(4,2,3)

        nouns_present,verbs_present=np.where(target_signal.any(axis=2))
        nouns_verbs_assoc_present=zip(nouns_present,verbs_present)

        nouns_present_or_absent,verbs_absent=np.where(~target_signal.any(axis=2))
        nouns_verbs_assoc_absent=zip(nouns_present_or_absent,verbs_absent)
        noun_present_but_verb_absent=[nvnp for nvnp in nouns_verbs_assoc_absent
                                        if nvnp[0] in dict(nouns_verbs_assoc_present).keys()]

        noun_absent_and_verb_absent=[nvnp for nvnp in nouns_verbs_assoc_absent
                                        if nvnp[0] not in dict(nouns_verbs_assoc_present).keys()]


        input_signal_nvp=input_signal[tuple(np.array(nouns_verbs_assoc_present).T)]
        target_signal_nvp=target_signal[tuple(np.array(nouns_verbs_assoc_present).T)]

        input_signal_npva=input_signal[tuple(np.array(noun_present_but_verb_absent).T)]
        target_signal_npva=target_signal[tuple(np.array(noun_present_but_verb_absent).T)]


        ## initialization
        performance_nvp= [] #performance of NVa admiting answer


        ## Computing errors and impossible states for NVa admiting answer
        for idx,nvp in enumerate(input_signal_nvp):
            ## Evaluate fraction of time when the good answer if given for the 3 signal AOR at the same time
            err_answer = threshold_and_take_max_before_error(input_signal=nvp,
                                                           target_signal=target_signal_nvp[idx],
                                                           error_measure=self.error_measure,
                                                           thresh=self.threshold)
            performance_nvp.append(1 - err_answer)


        ## Computing errors and impossible states for NVa not admiting answer, but present in the sentence
        performance_npva = [] #performance of NVa not admiting answer, but present
        for idx,npva in enumerate(input_signal_npva):
            err_answer = threshold_and_take_max_before_error(input_signal=npva,
                                                       target_signal=target_signal_npva[idx],
                                                       error_measure=self.error_measure,
                                                       thresh=self.threshold)
            performance_npva.append(1 - err_answer)


        ## Compute means
        if performance_nvp != []:
            if performance_npva != []:
                aa = performance_nvp
                naap = performance_npva
                performance_nvp = (len(aa)*mdp.numx.mean(aa) + len(naap)*mdp.numx.mean(naap)) / float((len(aa) + len(naap)))
            else:
                performance_nvp = mdp.numx.mean(performance_nvp)
        '''else:
            raise Exception, "There is no answer for this sentence."'''

        # compute the fraction of time when all the pertinent NVa are correct (for NVa present in the sentence)
        all_output_signal = []
        all_target_signal = []
        for NVassoc_tuple in NVassoc_contributing_anwser:
            all_output_signal.append(keep_max_for_each_time_step_with_default(NVassoc_tuple[1]))
            all_target_signal.append(NVassoc_tuple[2])
        for NVassoc_tuple in NVassoc_not_contributing_answer_but_present:
            all_output_signal.append(keep_max_for_each_time_step_with_default(NVassoc_tuple[1]))
            all_target_signal.append(NVassoc_tuple[2])

        global_out_arr = mdp.numx.concatenate(all_output_signal, axis=1)
        global_target_arr = mdp.numx.concatenate(all_target_signal, axis=1)
        global_err_answer = Oger.utils.threshold_before_error(input_signal=global_out_arr,
                                                   target_signal=global_target_arr,
                                                   error_measure=self.error_measure,
                                                   thresh=self.threshold)

        ## Supplementary computations (not used in default program)
        ## Compute the number of pertinent SW (semantic word) outputs for each verb that is erroneous
        # i.e. number of erroneous NV-assoc
        total_nr_of_pertinent_SW = len(NVassoc_contributing_anwser) + len(NVassoc_not_contributing_answer_but_present)
        nr_of_erroneous_SW = int(round(total_nr_of_pertinent_SW * (1-perf_asso_present)))



        '''return (1 - perf_asso_present, global_err_answer,
                nr_of_erroneous_SW, total_nr_of_pertinent_SW, NVa_correct, NVa_erroneous)'''
        #print 1-perf_asso_present
        return 1 - perf_asso_present


