# Naive_Bayes
Implementation of Naive Bayes using Unigram and Bigram models

https://courses.grainger.illinois.edu/ece448/sp2022/mps/mp1/assignment1.html

## Unigram
``python mp1.py --training ../data/spam_data/train --development ../data/spam_data/dev --stemming False --lowercase False --laplace 1.0 --pos_prior 0.8``

## Bigram
``python mp1.py --bigram True --training ../data/spam_data/train --development ../data/spam_data/dev --stemming False --lowercase False --bigram_laplace 1.0 --bigram_lambda 0.5 --pos_prior 0.8``
