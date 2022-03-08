'''
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall
Updated by: Raeid Saqur <raeidsaqur@cs.toronto.edu>

All of the files in this directory and all subdirectories are:
Copyright (c) 2022 University of Toronto
'''
import numpy as np

'''Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
'''

from math import exp  # exp(x) gives e^x
from typing import List, Sequence, Iterable


def grouper(seq: Sequence[str], n: int) -> List:
    '''Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    '''
    n_grams = []
    if (len(seq) > n):

        for i in range(len(seq)-n+1):
            n_grams.append(seq[i: i+n])

    return n_grams


def n_gram_precision(reference: Sequence[str], candidate: Sequence[str], n: int) -> float:
    '''Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    '''


    ref_ngram_list = grouper(reference, n)
    can_ngram_list = grouper(candidate, n)

    p_n = 0
    if len(candidate) != 0 and n < len(candidate):
        common_grams = [element for element in can_ngram_list if element in ref_ngram_list]
        p_n = len(common_grams)/len(can_ngram_list)


    return p_n


def brevity_penalty(reference: Sequence[str], candidate: Sequence[str]) -> float:
    '''Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    '''

    if len(candidate) == 0:
        BP = 0
    else:
        brevity = len(reference) / len(candidate)
        if brevity < 1:
            BP = 1
        else:
            BP = exp(1 - brevity)

    return BP


def BLEU_score(reference: Sequence[str], candidate: Sequence[str], n) -> float:
    '''Calculate the BLEU score

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    '''

    precision = 1
    for i in range(n):
        precision = precision * n_gram_precision(reference, candidate, i + 1)

    blue_score = brevity_penalty(reference, candidate) * np.power(precision, 1 / n)
    return blue_score


# ids = 0
# reference = '''\
# it is a guide to action that ensures that the military will always heed
# party commands'''.strip().split()
# candidate = '''\
# it is a guide to action which ensures that the military always obeys the
# commands of the party'''.strip().split()
# if ids:
#     # should work with token ids (ints) as well
#     reference = [hash(word) for word in reference]
#     candidate = [hash(word) for word in candidate]
#
# ## Unigram precision
# p1_hat = n_gram_precision(reference, candidate, 1)
# p1 = 15 / 18    # w/o capping
# assert np.isclose(p1_hat, p1)
#
# ## Bi-gram precision
# p2_hat = n_gram_precision(reference, candidate, 2)
# p2 = 8/17
# assert np.isclose(p2_hat, p2)
#
# ## BP
# BP_hat = brevity_penalty(reference, candidate)
# BP = 1.0
# assert np.isclose(BP_hat, BP)
#
# ## BLEU Score
# bleu_score_hat = BLEU_score(reference, candidate, 2)
# bleu_score = BP * (p1 * p2) ** (1 / 2)
# assert np.isclose(bleu_score_hat, bleu_score)
#
