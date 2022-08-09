#Import packages
import pandas as pd
from datasets import load_dataset
import os

if not os.path.exists('datasets'):
    os.makedirs('datasets')
os.chdir('datasets')
os.makedirs('fake_news/liar')
os.makedirs('fake_news/FakeNewsNet')
os.makedirs('topic/agnews')
os.makedirs('topic/yahoo_answers')
os.makedirs('sentiment/emotion/tweetEval')
os.makedirs('sentiment/emotion/CARER')
os.makedirs('sentiment/emotion/silicone')
os.makedirs('sentiment/polarity/imdb')
os.makedirs('sentiment/polarity/yelp')
os.makedirs('sentiment/polarity/sst2')
os.makedirs('sentiment/sarcasm/SemEval')
os.makedirs('sentiment/sarcasm/iSarcasm')
os.makedirs('sentiment/sarcasm/SARC')

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
    if 'validation' in  dataset.keys():
        pd.DataFrame(dataset['train']).to_csv(output_path + 'train.csv',index=False)
        pd.DataFrame(dataset['validation']).to_csv(output_path + 'val.csv',index=False)
        pd.DataFrame(dataset['test']).to_csv(output_path + 'test.csv',index=False)
    else :
        pd.DataFrame(dataset['train']).to_csv(output_path + 'train.csv',index=False)
        pd.DataFrame(dataset['test']).to_csv(output_path + 'test.csv',index=False)



#HuggingFace datasets
huggingface_to_csv('silicone','dyda_e','sentiment/emotion/silicone/')
huggingface_to_csv('liar','default','fake_news/liar/')
huggingface_to_csv('emotion','default','sentiment/emotion/CARER/')
huggingface_to_csv('imdb','plain_text','sentiment/emotion/IMDb/')
huggingface_to_csv('ag_news','default','topic/ag_news/')
huggingface_to_csv('yahoo_answers_topics','yahoo_answers_topics','topic/yahoo_answers/')
huggingface_to_csv('yelp_polarity','plain_text','sentiment/polarity/yelp/')
huggingface_to_csv('sst2','default','sentiment/polarity/SST2/')


#GitHub datasets
#TweetEval Emotion
pd.read_table('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/train_text.txt').to_csv('tweetEval/train_text.csv',index=False)
pd.read_table('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/train_labels.txt').to_csv('tweetEval/train_labels.csv',index=False)
pd.read_table('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/val_text.txt').to_csv('tweetEval/val_text.csv',index=False)
pd.read_table('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/val_labels.txt').to_csv('tweetEval/val_labels.csv',index=False)
pd.read_table('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/test_text.txt').to_csv('tweetEval/test_text.csv',index=False)
pd.read_table('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/test_labels.txt').to_csv('tweetEval/test_labels.csv',index=False)
#SemEval
pd.read_table('https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/train/SemEval2018-T3-train-taskA.txt').to_csv('sarcasm/SemEval/train-taskA.csv',index=False)
pd.read_table('https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/train/SemEval2018-T3-train-taskB.txt').to_csv('sarcasm/SemEval/train-taskB.csv',index=False)
pd.read_table('https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/test_TaskA/SemEval2018-T3_input_test_taskA.txt').to_csv('sarcasm/SemEval/test-taskA.csv',index=False)
pd.read_table('https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/test_TaskB/SemEval2018-T3_input_test_taskB.txt').to_csv('sarcasm/SemEval/test-taskB.csv',index=False)
pd.read_table('https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt').to_csv('sarcasm/SemEval/gold_test_taskA_emoji.csv',index=False)
pd.read_table('https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt').to_csv('sarcasm/SemEval/gold_test_taskB_emoji.csv',index=False)
#iSarcasm
pd.read_csv('https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/train/train.En.csv').to_csv('sarcasm/iSarcasm/train.En.csv',index=False)
pd.read_csv('https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/test/task_A_En_test.csv').to_csv('sarcasm/iSarcasm/task_A_En_test.csv',index=False)

#Archives
#SARC
pd.read_csv('https://nlp.cs.princeton.edu/SARC/1.0/main/test-balanced.csv.bz2',sep='\t',compression='bz2',header=None,compression=[0,1]).rename(columns={0:'label',1:'text'}).to_csv('sarcasm/SARC/test-balanced.csv',index=False)
pd.read_csv('https://nlp.cs.princeton.edu/SARC/1.0/main/train-balanced.csv.bz2',sep='\t',compression='bz2',header=None,compression=[0,1]).rename(columns={0:'label',1:'text'}).to_csv('sarcasm/SARC/train-balanced.csv',index=False)

#Return to root directory
os.chdir('..')
print('Data collection successfully completed!')