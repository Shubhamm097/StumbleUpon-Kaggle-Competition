# News Classification: Stumble Upon Kaggle Competition

## Table of Contents
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Approach](#approach)
  * [Technical Aspect](#technical-aspect)
  * [Run](#run)
  * [Bug / Feature Request](#bug---feature-request)
  * [Technology Stack](#technology-stack)
  * [Credits](#credits)


## Overview
News is a daily part of our lives. We all thrive on some kind of news or other. Thus, it becomes relevant for the news channels to assess the human behaviour and predict what sort of news coulg go on forever, in this [Kaggle competition](https://www.kaggle.com/c/stumbleupon), such news is classified as evergreen. A suitable use case for the news classification problem could be for all the big news telecasting company to recommend news that could make an impact on the reader's and might encourage him/her to do something good. Thus, through this project, I have tried to classify the news as evergreen or not, by having the dataset in the form of the news article.

## Motivation
News like climate change is something that one should focus on, its high time that we encourage people to give something to this bountiful nature in some sort, but it is clearly not the case, just by news classification, one can classify all the lame news to non-evergreen and news related to politics or climate change or something that might have an impact on one's life to be evergreen. Thus, in this dataset, a long paragraphs are given that requires to be classified as evergreen new or not.

## Approach
At first, data is cleaned by removing all the stopwords(and, a, the etc), applying contractions, applying lemmatization and stemming techniques so that the words like receiving and received will be transformed in its root form i.e receive. Also, the tweets are coverted into small letters so that words like Python and python would be treated as same word. Any non-alphabetical character is removed.

In order to classify the news, I tried with all the traditional machine learning algorithms. I tried using the feature extraction techniques like tfidf, word2vec and then gave those feature to the classification algorithms like logistic regression, or a basic one layer LSTM architecture, but those could not gave good results.

Therefore, I went for BERT, by fine-tuning the weights in accordance with the given data. The only problem with BERT was that it only allows sequence length of 512, but the sequences that were built using the news article were far more lengthy. In order to eradicate this problem a couple of things can be done.

1. Truncate the article either from the top bottom or from middle, this has the potential causes regarding loss of relvant sentences in the news article, therefore, I did not go ahead with this one.

2. The other thing that could be done is to summarize the news article in a paragraph that will lead to sequence of less than 512. There, are transformers that can do the text summarization task effectively. Thus, I used T5 transformer. I first summarize the news article and then gave it to the BERT for classification task. The problem with this approach is that it takes a lot of time to summarize all the news article in the dataset and it could easily take more than day.

## Technical Aspect
BERT is being used to classify the news as evergreen or not. BERT being the state-of-the-art model to understand the context, could really prove out to be a handy tool. BERT is being fine-tuned over the dataset. Kaggle's [dataset](https://www.kaggle.com/c/stumbleupon/data) has been used for the project. But instead of the news article as it is, first the news article is summarized then given to BERT.

## Run
Open a google colab notebook and run the cells.


## Bug / Feature Request
If you'd like to request a new feature/approach, feel free to do so by opening an issue [here](https://github.com/Shubhamm097/StumbleUpon-Kaggle-Competition/issues/new). Please include the relevant reasons for the feature or approach.

## Technology Stack
1. PyTorch
2. Python
3. BERT Architecture
4. Keras
5. T5 Transformers


## Credits
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

- [BERT repository](https://github.com/huggingface/transformers)

- [Fine-tune BERT for sentence classification](https://colab.research.google.com/github/DerwenAI/spaCy_tuTorial/blob/master/BERT_Fine_Tuning.ipynb)

- [Google's Blog on T5 Transformer](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)

- [Medium Article for T5 text summarization](https://towardsdatascience.com/simple-abstractive-text-summarization-with-pretrained-t5-text-to-text-transfer-transformer-10f6d602c426#:~:text=T5%20is%20a%20new%20transformer,and%20modified%20text%20as%20output.&text=It%20achieves%20state%2Dof%2Dthe,on%20a%20large%20text%20corpus.)
