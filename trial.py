import numpy as np

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

    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    pos_vocab, neg_vocab = create_word_maps_uni(train_set, train_labels, max_size=None)
    pos_vocab_bi, neg_vocab_bi = create_word_maps_bi(train_set, train_labels, max_size=None)

    max_vocab_size = None
    print(len(pos_vocab), len(neg_vocab))
    for x in list(pos_vocab_bi)[0:20]:
        print("key {}, value {} ".format(x, pos_vocab_bi[x]))
    P_N = pos_prior
    P_S = 1 - pos_prior
    devlabels = []
    tic = time.perf_counter()
    pos_word_count_uni = (sum(pos_vocab.values()) + unigram_laplace * (1 + len(pos_vocab)))
    pos_word_count_bi = (
            (sum(pos_vocab.values()) + sum(pos_vocab_bi.values())) +
            bigram_laplace * (1 + (len(pos_vocab) + len(pos_vocab_bi))))

    neg_word_count_uni = (
            sum(neg_vocab.values()) + unigram_laplace * (1 + len(neg_vocab)))
    neg_word_count_bi = (
                    (sum(neg_vocab.values()) + sum(neg_vocab_bi.values())) +
                    bigram_laplace * (1 + (len(neg_vocab) + len(neg_vocab_bi))))
    for sent in dev_set:
        P_ham_one = np.log(P_N)
        P_ham_two = np.log(P_N)
        P_spam_one = np.log(P_S)
        P_spam_two = np.log(P_S)

        for i in range(len(sent)):
            uni_temp = (pos_vocab.get(sent[i], 0) + unigram_laplace) / pos_word_count_uni
            bi_temp = (pos_vocab_bi.get(sent[i], 0) + bigram_laplace) / pos_word_count_bi

            P_ham_one += np.log(uni_temp)
            P_ham_two += np.log(bi_temp)
            P_ham = (1 - bigram_lambda )*P_ham_one + (bigram_lambda)*P_ham_two

            uni_temp1 = (neg_vocab.get(sent[i], 0) + unigram_laplace) / neg_word_count_uni
            bi_temp1 = (neg_vocab_bi.get(sent[i], 0) + bigram_laplace) / neg_word_count_bi

            P_spam_one += np.log(uni_temp1)
            P_spam_two += np.log(bi_temp1)
            P_spam = (1 - bigram_lambda) * P_spam_one + (bigram_lambda) * P_spam_two

        if P_spam < P_ham:
            devlabels.append(1)
        else:
            devlabels.append(0)
    # print(devlabels)
    # raise RuntimeError("Replace this line with your code!")
    toc = time.perf_counter()
    print(f"Bigram Bayes Time {toc - tic:0.4f} seconds")
    print(pos_word_count_uni, pos_word_count_bi, neg_word_count_uni, neg_word_count_bi)
    return devlabels


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
    tic = time.perf_counter()
    #print(len(X),'X')
    pos_vocab = {}
    neg_vocab = {}
    ##TODO:
    print(len(X), 'X')
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
    pos_vocab = Merge(pos_vocab_uni, pos_vocab)
    neg_vocab = Merge(neg_vocab_uni, neg_vocab)
    toc = time.perf_counter()
    print(f"Create world maps bi time {toc - tic:0.4f} seconds")
    # raise RuntimeError("Replace this line with your code!")
    return dict(pos_vocab), dict(neg_vocab)





    for doc in dev_set:
        for i in range(len(doc)):
            w_in_doc = doc.count(doc[i])
            len_of_doc = len(doc)
            if len_of_doc == 0:
                print(doc)
            for j in range(len(train_set)):
                count = 0
                if doc[i] in train_set[j]:
                    count += 1

            tf_idf_1 = (w_in_doc/len_of_doc) * np.log(no_of_docs_train_set/(1 + count))
            if tf_idf < tf_idf_1:
                tf_idf = tf_idf_1
                idx = i
        result.append(doc[idx])