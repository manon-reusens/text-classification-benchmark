#Import packages
import pandas as pd
from datasets import load_dataset
import os
import urllib.request
import tarfile


def huggingface_to_csv(name, 
                       subset,
                       output_path):
    '''
    Load a dataset from HuggingFace website and store it as csv files.
    Args:
        name: the name of the dataset to retrieve, as it appears on HuggingFace
        subset: the subset to retrieve
        output_path: the path to the location where the csv file needs to be stored
    '''
    dataset = load_dataset(name,subset)
    pd.DataFrame(dataset['train']).to_csv(output_path + 'train.csv',index=False)
    if 'test' in  dataset.keys():
        pd.DataFrame(dataset['test']).to_csv(output_path + 'test.csv',index=False)
    if 'validation' in  dataset.keys():
        pd.DataFrame(dataset['validation']).to_csv(output_path + 'val.csv',index=False)



if not os.path.exists('datasets'):
    os.makedirs('datasets')
os.chdir('datasets')
if not os.path.exists('fake_news'):
    if not os.path.exists('fake_news/liar'):
        os.makedirs('fake_news/liar')
    if not os.path.exists('fake_news/coAID'):
        os.makedirs('fake_news/CoAID')
    if not os.path.exists('fake_news/Gossipcop'):
        os.makedirs('fake_news/Gossipcop')
    if not os.path.exists('fake_news/Politifact'):
        os.makedirs('fake_news/Politifact')
    if not os.path.exists('fake_news/McIntyre'):
        os.makedirs('fake_news/McIntyre')
if not os.path.exists('topic'):
    os.makedirs('topic/agnews')
    os.makedirs('topic/WOS')
    os.makedirs('topic/bbc')
    os.makedirs('topic/yahoo')
if not os.path.exists('sentiment'):
    if not os.path.exists('sentiment/emotion'):
      os.makedirs('sentiment/emotion/tweetEval')
      os.makedirs('sentiment/emotion/CARER')
      os.makedirs('sentiment/emotion/silicone')
      os.makedirs('sentiment/emotion/MELD')
      #os.makedirs('sentiment/emotion/go_emotions')
    if not os.path.exists('sentiment/polarity'):
      os.makedirs('sentiment/polarity/imdb')
      os.makedirs('sentiment/polarity/movie_review')
      os.makedirs('sentiment/polarity/sst2')
      os.makedirs('sentiment/polarity/CR')
      os.makedirs('sentiment/polarity/STS-Gold')
    if not os.path.exists('sentiment/sarcasm'):
      os.makedirs('sentiment/sarcasm/SemEval')
      os.makedirs('sentiment/sarcasm/iSarcasm')
      os.makedirs('sentiment/sarcasm/SNH')
      os.makedirs('sentiment/sarcasm/sarcasm_v2')
      os.makedirs('sentiment/sarcasm/sarcasm_v1')

#GitHub datasets
#TweetEval Emotion
pd.read_table('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/train_text.txt').to_csv('sentiment/emotion/tweetEval/train_text.csv',index=False)
pd.read_table('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/train_labels.txt').to_csv('sentiment/emotion/tweetEval/train_labels.csv',index=False)
pd.read_table('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/val_text.txt').to_csv('sentiment/emotion/tweetEval/val_text.csv',index=False)
pd.read_table('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/val_labels.txt').to_csv('sentiment/emotion/tweetEval/val_labels.csv',index=False)
pd.read_table('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/test_text.txt').to_csv('sentiment/emotion/tweetEval/test_text.csv',index=False)
pd.read_table('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/test_labels.txt').to_csv('sentiment/emotion/tweetEval/test_labels.csv',index=False)
#MELD
pd.read_csv('https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv').to_csv('sentiment/emotion/MELD/train.csv',index=False)
pd.read_csv('https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv').to_csv('sentiment/emotion/MELD/dev.csv',index=False)
pd.read_csv('https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv').to_csv('sentiment/emotion/MELD/test.csv',index=False)
#GoEmotions
#pd.read_table('https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv', sep='\t').to_csv('sentiment/emotion/go_emotions/train.csv',index=False)
#pd.read_table('https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/dev.tsv', sep='\t').to_csv('sentiment/emotion/go_emotions/dev.csv',index=False)
#pd.read_table('https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/test.tsv', sep='\t').to_csv('sentiment/emotion/go_emotions/test.csv',index=False)
#SemEval
pd.read_table('https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/train/SemEval2018-T3-train-taskA.txt').to_csv('sentiment/sarcasm/SemEval/train-taskA.csv',index=False)
# pd.read_table('https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/train/SemEval2018-T3-train-taskB.txt').to_csv('sentiment/sarcasm/SemEval/train-taskB.csv',index=False)
pd.read_table('https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/test_TaskA/SemEval2018-T3_input_test_taskA.txt').to_csv('sentiment/sarcasm/SemEval/test-taskA.csv',index=False)
# pd.read_table('https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/test_TaskB/SemEval2018-T3_input_test_taskB.txt').to_csv('sentiment/sarcasm/SemEval/test-taskB.csv',index=False)
pd.read_table('https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt').to_csv('sentiment/sarcasm/SemEval/gold_test_taskA_emoji.csv',index=False)
# pd.read_table('https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt').to_csv('sentiment/sarcasm/SemEval/gold_test_taskB_emoji.csv',index=False)
#iSarcasm
pd.read_csv('https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/train/train.En.csv').to_csv('sentiment/sarcasm/iSarcasm/train.En.csv',index=False)
pd.read_csv('https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/test/task_A_En_test.csv').to_csv('sentiment/sarcasm/iSarcasm/task_A_En_test.csv',index=False)
#CoAID
fake1 = pd.read_csv('https://raw.githubusercontent.com/cuilimeng/CoAID/master/05-01-2020/NewsFakeCOVID-19.csv')
real1 = pd.read_csv('https://raw.githubusercontent.com/cuilimeng/CoAID/master/05-01-2020/NewsRealCOVID-19.csv')
fake2 = pd.read_csv('https://raw.githubusercontent.com/cuilimeng/CoAID/master/07-01-2020/NewsFakeCOVID-19.csv')
real2 = pd.read_csv('https://raw.githubusercontent.com/cuilimeng/CoAID/master/07-01-2020/NewsRealCOVID-19.csv')
fake3 = pd.read_csv('https://raw.githubusercontent.com/cuilimeng/CoAID/master/09-01-2020/NewsFakeCOVID-19.csv')
real3 = pd.read_csv('https://raw.githubusercontent.com/cuilimeng/CoAID/master/09-01-2020/NewsRealCOVID-19.csv')
fake4 = pd.read_csv('https://raw.githubusercontent.com/cuilimeng/CoAID/master/11-01-2020/NewsFakeCOVID-19.csv')
real4 = pd.read_csv('https://raw.githubusercontent.com/cuilimeng/CoAID/master/11-01-2020/NewsRealCOVID-19.csv')
fake = pd.concat([fake1,fake2,fake3,fake4],ignore_index=True)
real = pd.concat([real1,real2,real3,real4],ignore_index=True)
fake.to_csv('fake_news/CoAID/Fake.csv',index=False)
real.to_csv('fake_news/CoAID/Real.csv',index=False)
#movie_review
tar_dataset = "https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz"
ftpstream = urllib.request.urlopen(tar_dataset)
tar_dataset = tarfile.open(fileobj=ftpstream, mode="r|gz")
tar_dataset.extractall(path='sentiment/polarity/movie_review')
#Gossipcop
fake=pd.read_csv('https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/gossipcop_fake.csv')
real=pd.read_csv('https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/gossipcop_real.csv')
fake.to_csv('fake_news/Gossipcop/fake.csv',index=False)
real.to_csv('fake_news/Gossipcop/real.csv',index=False)
#Politifact
fake=pd.read_csv('https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_fake.csv')
real=pd.read_csv('https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_real.csv')
fake.to_csv('fake_news/Politifact/fake.csv',index=False)
real.to_csv('fake_news/Politifact/real.csv',index=False)
#mcintyre
pd.read_csv('https://raw.githubusercontent.com/GeorgeMcIntire/fake_real_news_dataset/main/fake_and_real_news_dataset.csv').to_csv('fake_news/McIntire/fake_or_real_news.csv',index=False)



#Archives
#SARC
#pd.read_csv('https://nlp.cs.princeton.edu/SARC/1.0/main/test-balanced.csv.bz2',sep='\t',compression='bz2',header=None,usecols=[0,1]).rename(columns={0:'label',1:'text'}).to_csv('sentiment/sarcasm/sarc/test-balanced.csv',index=False)
#pd.read_csv('https://nlp.cs.princeton.edu/SARC/1.0/main/train-balanced.csv.bz2',sep='\t',compression='bz2',header=None,usecols=[0,1]).rename(columns={0:'label',1:'text'}).to_csv('sentiment/sarcasm/sarc/train-balanced.csv',index=False)


#HuggingFace datasets
huggingface_to_csv('silicone','dyda_e','sentiment/emotion/silicone/')
huggingface_to_csv('liar','default','fake_news/liar/')
huggingface_to_csv('emotion','default','sentiment/emotion/CARER/')
huggingface_to_csv('imdb','plain_text','sentiment/polarity/imdb/')
huggingface_to_csv('SetFit/CR', 'default','sentiment/polarity/CR/')
huggingface_to_csv('ag_news','default','topic/agnews/')
huggingface_to_csv('yahoo_answers_topics','yahoo_answers_topics','topic/yahoo/')
#huggingface_to_csv('yelp_polarity','plain_text','sentiment/polarity/yelp/')
huggingface_to_csv('sst2','default','sentiment/polarity/sst2/')
huggingface_to_csv('raquiba/Sarcasm_News_Headline', 'default','sentiment/sarcasm/SNH/')
huggingface_to_csv('web_of_science',  'WOS11967', 'topic/WOS/')
huggingface_to_csv('SetFit/bbc-news', 'default' , 'topic/bbc/')


#Return to root directory
os.chdir('..')
print('Data collection successfully completed!')
