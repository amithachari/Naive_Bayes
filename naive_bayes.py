# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader
import time

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset_main(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def create_word_maps_uni(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: words 
        values: number of times the word appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word appears 
    """
    ##TODO:
    pos_list = []
    neg_list = []

    for i in range(len(y)):
        if y[i] == 1:
            pos_list += X[i]
        else:
            neg_list += X[i]

    pos_vocab = Counter(pos_list)
    print(len(pos_vocab))
    neg_vocab = Counter(neg_list)

    return dict(pos_vocab), dict(neg_vocab)

def create_bigram(elist):
   list_bigrams = []
   bigram_counts = {}

   for i in range(len(elist)-1):
      if i < len(elist) - 1:
         list_bigrams.append((elist[i] +" "+ elist[i + 1]))

         if (elist[i] +" "+ elist[i+1]) in bigram_counts:
            bigram_counts[(elist[i] +" "+ elist[i + 1])] += 1
         else:
            bigram_counts[(elist[i] +" "+ elist[i + 1])] = 1

   return bigram_counts

def add_dicts(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def create_word_maps_bi(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1
        keys: pairs of words
        values: number of times the word pair appears
    neg_vocab:
        In data where labels are 0
        keys: words
        values: number of times the word pair appears
    """

    ##TODO:
    pos_list = []
    neg_list = []
    for i in range(len(y)):
        if y[i] == 1:
            pos_list += X[i]
        else:
            neg_list += X[i]

    pos_vocab = create_bigram(pos_list)
    neg_vocab = create_bigram(neg_list)

    pos_vocab_uni, neg_vocab_uni = create_word_maps_uni(X, y, max_size=None)
    pos_vocab = add_dicts(pos_vocab_uni, pos_vocab)
    neg_vocab = add_dicts(neg_vocab_uni, neg_vocab)

    return dict(pos_vocab), dict(neg_vocab)

# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.001, pos_prior=0.8, silently=False):
    '''
    Compute a naive Bayes unigram model from a training set; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    # Keep this in the provided template

    pos_vocab, neg_vocab = create_word_maps_uni(train_set, train_labels, max_size=None)

    P_N = pos_prior
    P_S = 1 - pos_prior
    devlabels = []

    word_count_pos = (sum(pos_vocab.values()) + laplace * (1 + len(pos_vocab)))
    word_count_neg = (sum(neg_vocab.values()) + laplace * (1 + len(neg_vocab)))
    for sent in dev_set:
        P_ham = np.log(P_N)
        P_spam = np.log(P_S)

        for i in range(len(sent)):
            temp = (pos_vocab.get(sent[i], 0) + laplace) / word_count_pos
            temp = np.log(temp)
            P_ham += temp

            temp1 = (neg_vocab.get(sent[i], 0) + laplace) / word_count_neg
            temp1 = np.log(temp1)
            P_spam += temp1

        if P_spam < P_ham:
            devlabels.append(1)
        else:
            devlabels.append(0)

    return devlabels


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.8,silently=False):
    '''
    Compute a unigram+bigram naive Bayes model; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    unigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    bigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating bigram probs
    bigram_lambda (scalar float) = interpolation weight for the bigram model
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    pos_vocab_uni, neg_vocab_uni = create_word_maps_uni(train_set, train_labels, max_size=None)
    pos_vocab_bi, neg_vocab_bi = create_word_maps_bi(train_set, train_labels, max_size=None)

    P_N = pos_prior
    P_S = 1 - pos_prior
    devlabels = []

    word_count_pos = (sum(pos_vocab_uni.values()) + unigram_laplace * (1 + len(pos_vocab_uni)))
    word_count_neg = (sum(neg_vocab_uni.values()) + unigram_laplace * (1 + len(neg_vocab_uni)))

    word_count_pos_bi = ((sum(pos_vocab_bi.values()) - sum(pos_vocab_uni.values())) +
                      bigram_laplace * (1 + (len(pos_vocab_bi) - len(pos_vocab_uni))) )

    word_count_neg_bi = ((sum(neg_vocab_bi.values()) - sum(neg_vocab_uni.values())) +
                      bigram_laplace * (1 + (len(neg_vocab_bi) - len(neg_vocab_uni))) )


    for sent in dev_set:
        P_ham_one = np.log(P_N)
        P_spam_one = np.log(P_S)
        P_ham_two = np.log(P_N)
        P_spam_two = np.log(P_S)

        for i in range(len(sent)):
            uni_temp = (pos_vocab_uni.get(sent[i], 0) + unigram_laplace) / word_count_pos
            uni_temp = np.log(uni_temp)
            P_ham_one += uni_temp

            uni_temp1 = (neg_vocab_uni.get(sent[i], 0) + unigram_laplace) / word_count_neg
            uni_temp1 = np.log(uni_temp1)
            P_spam_one += uni_temp1

        for i in range(len(sent) - 1):
            bigram = (sent[i] +" "+ sent[i + 1])
            bi_temp = (pos_vocab_bi.get(bigram,0) + bigram_laplace) / word_count_pos_bi
            bi_temp = np.log(bi_temp)
            P_ham_two += bi_temp

            bi_temp1 = (neg_vocab_bi.get(bigram,0) + bigram_laplace) / word_count_neg_bi
            bi_temp1 = np.log(bi_temp1)
            P_spam_two += bi_temp1

        P_ham = ( 1 - bigram_lambda)*P_ham_one + bigram_lambda*P_ham_two
        P_spam = ( 1 - bigram_lambda)*P_spam_one + bigram_lambda*P_spam_two

        if P_spam < P_ham:
            devlabels.append(1)
        else:
            devlabels.append(0)

    return devlabels
