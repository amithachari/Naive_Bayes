# Naive Bayes
Implementation of Naive Bayes using Unigram and Bigram models

https://courses.grainger.illinois.edu/ece448/sp2022/mps/mp1/assignment1.html

## Background
Given a dataset consisting of Spam and Ham (not spam) emails. Using the training set, will learn a Naive Bayes classifier that will predict the right class label given an unseen email. Used the development set to test the accuracy of your learned model, with the included function grade.py

## Dataset
The dataset that used consists of 1500 ham and 1500 spam emails, a subset of the Enron-Spam dataset provided by Ion Androutsopoulos. This dataset is split into 2000 training examples and 1000 development examples. This dataset is located in the spam_data folder of the template code provided. You will also find another dataset located under the counter-data folder of the template code - this data is only used by our grade.py file to check whether your model implementation is correct.

## Background
The bag of words model in NLP is a simple unigram model which considers a text to be represented as a bag of independent words. That is, we ignore the position the words appear in, and only pay attention to their frequency in the text. Here, each email consists of a group of words. Using Bayes theorem, you need to compute the probability that the label of an email (Y) should be ham (Y=ham) given the words in the email. Thus you need to estimate the posterior probabilities:

![image](https://github.com/amithachari/Naive_Bayes/assets/64373075/4f993f81-9110-46a8-b6ec-de57f3ecf436)

## Unigram
``python mp1.py --training ../data/spam_data/train --development ../data/spam_data/dev --stemming False --lowercase False --laplace 1.0 --pos_prior 0.8``

## Bigram
``python mp1.py --bigram True --training ../data/spam_data/train --development ../data/spam_data/dev --stemming False --lowercase False --bigram_laplace 1.0 --bigram_lambda 0.5 --pos_prior 0.8``
