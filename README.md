# Experimental evaluation of the state-of-the-art in text classification

## Structure
* util/dataloader.py :  collect the datasets in their raw format and convert the useful columns to a pandas dataframe
* huggingface_loading_script.py :  collect datasets from the huggingface hub and store them as csv files
* preprocessing.py : preprocess raw text fields


## Datasets

15 datasets are included

| Dataset | Task |  Classes  | Size |  Split |
| --- | --- |  --- | --- | --- |
| FakeNewsNet - Politifact | Fake News | 2 |  | |
| FakeNewsNet - GossipCop | Fake News |2 |  | | 
| Fake & Real News | Fake News | 2 | |  |
| LIAR | Fake News | 6 | | |
| 20News | Topic | 20 | | |
| AGNews | Topic | 4 |  | |
| Yahoo | Topic | 10 | |  |
| TweetEval Emotion | Emotion | 4 | | |
| CARER | Emotion | 8 | | |
| DailyDialog Act - Silicone | Emotion | 7 | | |
| IMDb | Polarity | 2 | | |
| Stanford Sentiment Treebank | Polarity | 2 | | |
| YELP | Polarity | 2 | | |
| TweetEval Irony | Sarcasm | 2  | | | 
| SemEval Task 3 | Sarcasm | 2(4) | | |
| SARC | Sarcasm | 2 | |  |





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
