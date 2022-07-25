# Experimental evaluation of the state-of-the-art in text classification

## Structure
* util/dataloader.py :  collect the datasets in their raw format and convert the useful columns to a pandas dataframe
* huggingface_loading_script.py :  collect datasets from the huggingface hub and store them as csv files
* preprocessing.py : preprocess raw text fields


## Datasets

15 datasets are included

| Dataset | Task |  Classes  | Size |  Split |
| --- | --- |  --- | --- | --- |
| FakeNewsNet - Politifact | Fake News | 2 | 1056  | None |
| FakeNewsNet - GossipCop | Fake News |2 | 18590 | None | 
| Fake & Real News | Fake News | 2 | 44898 | None  |
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





### Links

* [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)
* [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
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
* [Tweet Eval : Irony detection](https://github.com/cardiffnlp/tweeteval)
* [SemEval 2018 Task 3](https://github.com/Cyvhee/SemEval2018-Task3)
* [SARC](https://nlp.cs.princeton.edu/SARC/1.0/)
