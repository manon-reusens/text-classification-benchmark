# Experimental evaluation of the state-of-the-art in text classification

## Structure
* util/dataloader.py :  collect the datasets in their raw format and convert the useful columns to a pandas dataframe
* util/datasplitter.py : splits a dataset into its train-(val)-test set components
* evaluator.py : fits a model on the train data and evaluates it on the test set
* fasttext_embeddings.py : loads fast text embeddings and generates sentence embeddings for a corpus
* huggingface_loading_script.py :  collect datasets from the huggingface hub and store them as csv files
* preprocessing.py : preprocess raw text fields


## Datasets

Instructions draft : the datasets are collected by running 'data_collection.py'. The only exception is FakeNewsNet which needs to be downloaded separately by :
1) cloning the [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) repository in a directory datasets/fake_news/FakeNewsNet
2) Following the instructions in the readme of the FakeNewsNet repository

| Dataset | Task |  Classes  | Size |  Split |
| --- | --- |  --- | --- | --- |
| FakeNewsNet - Politifact | Fake News | 2 | 1056  | None (80% Train - 20% Test in paper) |
| FakeNewsNet - GossipCop | Fake News |2 | 18590 | None (80% Train - 20% Test in paper) | 
| LIAR | Fake News | 6 | 12836 | Train-Val-Test |
| 20News | Topic | 20 | 18846 | Train-Test |
| AGNews | Topic | 4 | 127600 | Train-Test |
| Yahoo | Topic | 10 | 1460000 | Train-Test |
| TweetEval Emotion | Emotion | 4 | 5052 | Train-Val-Test |
| CARER | Emotion | 8 | 20000 | Train-Val-Test |
| DailyDialog Act - Silicone | Emotion | 7 | 102979 | Train-Val-Test |
| IMDb | Polarity | 2 | 50000 | Train-Test |
| Stanford Sentiment Treebank | Polarity | 2 | 68403 |  Train-Val-Test |
| YELP | Polarity | 2 | 598000 | Train-Test |
| SemEval Task 3 | Sarcasm | 2(4) | 4601 | Train-Test |
| SARC | Sarcasm | 2 | 1409012 | Train-Test |
|iSarcasm - English | Sarcasm | 2 | 5735 | Train-Test | 


### Links

The datasets can be retrieved with the following links.

* [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)
* [LIAR](https://huggingface.co/datasets/liar)
* [20News](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html)
* [AGNews](https://huggingface.co/datasets/ag_news)
* [Yahoo!Answer](https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU)
* [Tweet Eval : Emotion detection](https://github.com/cardiffnlp/tweeteval)
* [CARER Emotion](https://huggingface.co/datasets/emotion)
* [Daily Dialog Act Corpus (silicone)](https://huggingface.co/datasets/silicone/viewer/dyda_e/train)
* [IMDb](https://huggingface.co/datasets/imdb#dataset-creation)
* [Stanford Sentiment Tree Bank](https://huggingface.co/datasets/sst2)
* [Yelp polarity](https://huggingface.co/datasets/yelp_polarity)
* [SemEval 2018 Task 3](https://github.com/Cyvhee/SemEval2018-Task3)
* [SARC](https://nlp.cs.princeton.edu/SARC/1.0/)
* [SemEval 2022 iSarcasm](https://github.com/iabufarha/iSarcasmEval) 
