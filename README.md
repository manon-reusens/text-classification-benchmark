# Experimental evaluation of the state-of-the-art in text classification

## Structure
* config : contains the yaml file with the hyperparameter configurations for Weight&Biases
* notebooks :  contains example notebooks
* preprocessing/fasttext_embeddings.py : loads fast text embeddings and generates sentence embeddings for a corpus
* preprocessing/preprocessor.py : preprocess raw text fields
* util/dataloader.py :  collect the datasets in their raw format and convert the useful columns to a pandas dataframe
* util/datasplitter.py : splits a dataset into its train-(val)-test set components
* data_collection.py :  collect datasets from the web and store them as csv files
* evaluator.py : fits a model on the train data and evaluates it on the test set
* run.py : main script, conducts an evaluation of all models accross all tasks

## Installation

Experiments are conducted with Python 3.9.
```
$ conda create --name TextBenchmark python=3.9
$ conda activate TextBenchmark
$ pip install -r requirements.txt
```

## How to use
Instructions for datasets collection : 

1. Run 'data_collection.py'. This will download all datasets except FakeNewsNet.
2. Clone the [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) repository in datasets/fake_news/FakeNewsBet
3. Follow the data collection instructions in the readme file of FakeNewsNet
4. Execute 'run.py' to get the trained models and evaluation metrics reported in the paper

## Datasets
### Structure of the datasets

| Dataset | Task |  Classes  | Size |  Split |
| --- | --- |  --- | --- | --- |
| FakeNewsNet - GossipCop | Fake News |2 | 18590 | None (80% Train - 20% Test in paper) | 
| CoAID | Fake News | 2 |  3669     | None (75% Train - 25% Test in paper) |
| LIAR | Fake News | 6 | 12836 | Train-Val-Test |
| 20News | Topic | 20 | 18846 | Train-Test |
| AGNews | Topic | 4 | 127600 | Train-Test |
| Web of Science Dataset | Topic | 7 | 11967 | None |
| TweetEval Emotion | Emotion | 4 | 5052 | Train-Val-Test |
| CARER | Emotion | 8 | 20000 | Train-Val-Test |
| DailyDialog Act - Silicone | Emotion | 7 | 102979 | Train-Val-Test |
| IMDb | Polarity | 2 | 50000 | Train-Test |
| Stanford Sentiment Treebank | Polarity | 2 | 68403 |  Train-Val-Test |
| Movie Review | Polarity | 2 | 10662 | None |
| SemEval Task 3 | Sarcasm | 2(4) | 4601 | Train-Test |
| SARC | Sarcasm | 2 | 1409012 | Train-Test |
|iSarcasm - English | Sarcasm | 2 | 5735 | Train-Test | 
| Sarcasm News Headlines | Sarcasm | 2 | 55328 | Train-Test |


### Links

The datasets can be retrieved with the following links.

* [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)
* [CoAID](https://github.com/cuilimeng/CoAID)
* [LIAR](https://huggingface.co/datasets/liar)
* [20News](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html)
* [AGNews](https://huggingface.co/datasets/ag_news)
* [Web of Science Dataset] (https://huggingface.co/datasets/web_of_science)
* [Tweet Eval : Emotion detection](https://github.com/cardiffnlp/tweeteval)
* [CARER Emotion](https://huggingface.co/datasets/emotion)
* [Daily Dialog Act Corpus (silicone)](https://huggingface.co/datasets/silicone/viewer/dyda_e/train)
* [Stanford Sentiment Tree Bank](https://huggingface.co/datasets/sst2)
* [Movie Review] (https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz)
* [SemEval 2018 Task 3](https://github.com/Cyvhee/SemEval2018-Task3)
* [SARC](https://nlp.cs.princeton.edu/SARC/1.0/)
* [SemEval 2022 iSarcasm](https://github.com/iabufarha/iSarcasmEval) 
* [Sarcasm News Headlines] (https://huggingface.co/datasets/raquiba/Sarcasm_News_Headline/viewer/raquiba--Sarcasm_News_Headline)
