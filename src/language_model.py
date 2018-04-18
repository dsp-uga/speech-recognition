'''
This is an implementation of using language models in speech recognition.

Largely adapted from Baidu's DeepSpeech here:
https://github.com/PaddlePaddle/DeepSpeech/tree/develop/decoders

However, this implementation does not
depend on PaddlePaddle. Please check the README for detailed requirements and
installation instructions.
'''

import swig_decoders

class Scorer(swig_decoders.Scorer):
    """Wrapper for Scorer.

    :param alpha: Parameter associated with language model. Don't use
                  language model when alpha = 0.
    :type alpha: float
    :param beta: Parameter associated with word count. Don't use word
                 count when beta = 0.
    :type beta: float
    :model_path: Path to load language model.
    :type model_path: basestring
    """

    def __init__(self, alpha, beta, model_path, vocabulary):
        swig_decoders.Scorer.__init__(self, alpha, beta, model_path, vocabulary)


def ctc_beam_search_decoder_batch(probs_split,
                                  vocabulary,
                                  beam_size,
                                  num_processes,
                                  cutoff_prob=1.0,
                                  cutoff_top_n=40,
                                  ext_scoring_func=None):
    """Wrapper for the batched CTC beam search decoder.

    :param probs_seq: 3-D list with each element as an instance of 2-D list
                      of probabilities used by ctc_beam_search_decoder().
    :type probs_seq: 3-D list
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :param beam_size: Width for beam search.
    :type beam_size: int
    :param num_processes: Number of parallel processes.
    :type num_processes: int
    :param cutoff_prob: Cutoff probability in vocabulary pruning,
                        default 1.0, no pruning.
    :type cutoff_prob: float
    :param cutoff_top_n: Cutoff number in pruning, only top cutoff_top_n
                         characters with highest probs in vocabulary will be
                         used in beam search, default 40.
    :type cutoff_top_n: int
    :param num_processes: Number of parallel processes.
    :type num_processes: int
    :param ext_scoring_func: External scoring function for
                             partially decoded sentence, e.g. word count
                             or language model.
    :type external_scoring_function: callable
    :return: List of tuples of log probability and sentence as decoding
             results, in descending order of the probability.
    :rtype: list
    """
    probs_split = [probs_seq.tolist() for probs_seq in probs_split]

    batch_beam_results = swig_decoders.ctc_beam_search_decoder_batch(
        probs_split, vocabulary, beam_size, num_processes, cutoff_prob,
        cutoff_top_n, ext_scoring_func)
    batch_beam_results = [
        [(res[0], res[1]) for res in beam_results]
        for beam_results in batch_beam_results
        ]
    return batch_beam_results


def ctc_beam_search_decoder(probs_seq,
                            vocabulary,
                            beam_size,
                            cutoff_prob=1.0,
                            cutoff_top_n=40,
                            ext_scoring_func=None):
    """Wrapper for the CTC Beam Search Decoder.

    :param probs_seq: 2-D list of probability distributions over each time
                      step, with each element being a list of normalized
                      probabilities over vocabulary and blank.
    :type probs_seq: 2-D list
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :param beam_size: Width for beam search.
    :type beam_size: int
    :param cutoff_prob: Cutoff probability in pruning,
                        default 1.0, no pruning.
    :type cutoff_prob: float
    :param cutoff_top_n: Cutoff number in pruning, only top cutoff_top_n
                         characters with highest probs in vocabulary will be
                         used in beam search, default 40.
    :type cutoff_top_n: int
    :param ext_scoring_func: External scoring function for
                             partially decoded sentence, e.g. word count
                             or language model.
    :type external_scoring_func: callable
    :return: List of tuples of log probability and sentence as decoding
             results, in descending order of the probability.
    :rtype: list
    """
    beam_results = swig_decoders.ctc_beam_search_decoder(
        probs_seq.tolist(), vocabulary, beam_size, cutoff_prob, cutoff_top_n,
        ext_scoring_func)
    beam_results = [(res[0], res[1]) for res in beam_results]
    return beam_results


def init_ext_scorer(language_model_path,
                    vocab_list, beam_alpha=5, beam_beta=1):
    """Initialize the external scorer.

    :param beam_alpha: Parameter associated with language model.
    :type beam_alpha: float
    :param beam_beta: Parameter associated with word count.
    :type beam_beta: float
    :param language_model_path: Filepath for language model. If it is
                                empty, the external scorer will be set to
                                None, and the decoding method will be pure
                                beam search without scorer.
    :type language_model_path: basestring|None
    :param vocab_list: List of tokens in the vocabulary, for decoding.
    :type vocab_list: list
    """
    if language_model_path != '':
        _ext_scorer = Scorer(beam_alpha, beam_beta, language_model_path, vocab_list)
        lm_char_based = _ext_scorer.is_character_based()
        lm_max_order = _ext_scorer.get_max_order()
        lm_dict_size = _ext_scorer.get_dict_size()
    else:
        _ext_scorer = None
