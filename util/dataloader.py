#Import packages
import pandas as pd
import numpy as np
import os 
import json
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups

class DataLoader:

    def __init__(self,
                 subset=['fake_news','topic','emotion','polarity','sarcasm']):
        '''
        Loads all required datasets as pandas dataframes.
        Param : 
            subset (list) : task-related subsets to select
        Output :
            a dictionary with the name of the dataset as key and the corresponding dataframe as value.
        '''
        self.subset = subset


    def retrieve_text_path(self,path):
        #Auxiliary function to load the FakeNewsNet data from the json subfolders
        text = []
        for folder in tqdm(os.listdir(path)):
            if os.path.exists(path+folder+"/news content.json"):
                with open(path+folder+"/news content.json")  as f:
                    text.append(json.load(f)['text'])
            else:              
                text.append(' ')
        return text


    def load(self):
        dataset_dict = {}
        #Fake News datasets
        os.chdir('datasets/fake_news')
        if 'fake_news' in self.subset:
            #Fake and Real News dataset
            real_news = pd.read_csv('fake_and_real_news/True.csv',usecols = ['title','text'])
            fake_news = pd.read_csv('fake_and_real_news/Fake.csv',usecols = ['title','text'])
            real_news['label'] = np.full(shape=(real_news.shape[0],1),fill_value=1)
            fake_news['label'] = np.zeros(shape=(fake_news.shape[0],1))
            dataset_dict['fake_real_news'] = pd.concat([real_news,fake_news])
            #FakeNewsNet Politifact
            real_path = "FakeNewsNet/code/fakenewsnet_dataset/politifact/real/"
            fake_path = "FakeNewsNet/code/fakenewsnet_dataset/politifact/fake/"
            real_text = self.retrieve_text_path(real_path)
            fake_text = self.retrieve_text_path(fake_path)    
            label = [1] * len(real_text) + [0] * len(fake_text)        
            dataset_dict['politifact'] = pd.DataFrame({'text':real_text+fake_text,'label':label})
            #FakeNewsNet GossipCop
            real_path = "FakeNewsNet/code/fakenewsnet_dataset/gossipcop/real/"
            fake_path = "FakeNewsNet/code/fakenewsnet_dataset/gossipcop/fake/"
            real_text = self.retrieve_text_path(real_path)
            fake_text = self.retrieve_text_path(fake_path)    
            label = [1] * len(real_text) + [0] * len(fake_text)        
            dataset_dict['gossipcop'] = pd.DataFrame({'text':real_text+fake_text,'label':label})
            #LIAR
            dataset_dict['liar'] = {}
            dataset_dict['liar']['train'] = pd.read_csv('liar/train.csv',
                         usecols=['label','statement']).rename(columns={'statement':'text'})
            dataset_dict['liar']['val'] = pd.read_csv('liar/val.csv',
                         usecols=['label','statement']).rename(columns={'statement':'text'})
            dataset_dict['liar']['test'] = pd.read_csv('liar/test.csv',
                         usecols=['label','statement']).rename(columns={'statement':'text'})
            
        #Topic Classification
        os.chdir('../topic')
        if 'topic' in self.subset:
            #20NewsGroup
            #https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset
            dataset_dict['twentynews'] = {}
            twentynews_train = fetch_20newsgroups(subset='train',
                           remove=('headers', 'footers', 'quotes') #option to remove metadata 
                          )
            twentynews_test = fetch_20newsgroups(subset='test',
                           remove=('headers', 'footers', 'quotes') #option to remove metadata  
                          )    

            dataset_dict['twentynews']['train'] = pd.DataFrame({'label':twentynews_train['target'],
                                     'text':twentynews_train['data']})
            dataset_dict['twentynews']['test'] = pd.DataFrame({'label':twentynews_test['target'],
                                     'text':twentynews_test['data']})  
            #AG News
            dataset_dict['agnews'] = {}
            dataset_dict['agnews']['train'] = pd.read_csv('ag_news/train.csv')
            dataset_dict['agnews']['test'] = pd.read_csv('ag_news/test.csv')
            #Yahoo answers
            dataset_dict['yahoo'] = {}
            col_dict = {0:'label',1:'title',2:'question',3:'answer'}
            dataset_dict['yahoo']['train'] = pd.read_csv('yahoo_answers/train.csv',
                          header=None).rename(columns=col_dict)
            dataset_dict['yahoo']['train']['text'] = dataset_dict['yahoo']['train']['title'] + ' ' + dataset_dict['yahoo']['train']['question']+ ' ' + dataset_dict['yahoo']['train']['answer']
            dataset_dict['yahoo']['test'] = pd.read_csv('yahoo_answers/test.csv',
                          header=None).rename(columns=col_dict)
            dataset_dict['yahoo']['test']['text'] = dataset_dict['yahoo']['test']['title'] + ' ' + dataset_dict['yahoo']['test']['question']+ ' ' + dataset_dict['yahoo']['test']['answer']
            
        #Sentiment Analysis 1 : Emotion
        os.chdir('../sentiment/emotion')
        if 'emotion' in self.subset:
            #Tweet Eval : Emotion
            dataset_dict['eval_emotion'] = {}
            dataset_dict['eval_emotion']['train'] = pd.DataFrame()
            dataset_dict['eval_emotion']['val'] = pd.DataFrame()
            dataset_dict['eval_emotion']['test'] = pd.DataFrame()
            dataset_dict['eval_emotion']['train']['label'] = pd.read_table('tweetEval/datasets/train_labels.txt',header=None)
            dataset_dict['eval_emotion']['val']['label'] = pd.read_table('tweetEval/datasets/val_labels.txt',header=None)
            dataset_dict['eval_emotion']['test']['label'] = pd.read_table('tweetEval/datasets/test_labels.txt',header=None)
            dataset_dict['eval_emotion']['train']['text']= pd.read_table('tweetEval/datasets/train_text.txt',header=None)
            dataset_dict['eval_emotion']['val']['text']= pd.read_table('tweetEval/datasets/val_text.txt',header=None)
            dataset_dict['eval_emotion']['test']['text'] = pd.read_table('tweetEval/datasets/test_text.txt',header=None)
            #CARER
            dataset_dict['CARER'] = {}
            dataset_dict['CARER']['train'] = pd.read_csv('CARER/train.csv')
            dataset_dict['CARER']['val'] = pd.read_csv('CARER/val.csv')
            dataset_dict['CARER']['test'] = pd.read_csv('CARER/test.csv')
            #silicone (Daily Dialog Act)
            dataset_dict['silicone'] = {}
            dataset_dict['silicone']['train'] = pd.read_csv('silicone/train.csv',
                             usecols=['Utterance','Label']).rename(columns={'Utterance':'text','Label':'label'})
            dataset_dict['silicone']['val'] = pd.read_csv('silicone/val.csv',
                             usecols=['Utterance','Label']).rename(columns={'Utterance':'text','Label':'label'})
            dataset_dict['silicone']['test'] = pd.read_csv('silicone/test.csv',
                             usecols=['Utterance','Label']).rename(columns={'Utterance':'text','Label':'label'})

        # Sentiment Analysis 2: Polarity  
        os.chdir('../polarity')
        if 'polarity' in self.subset:
            #IMDb
            dataset_dict['imdb'] = {}
            dataset_dict['imdb']['train'] = pd.read_csv('IMDb/train.csv')
            dataset_dict['imdb']['test'] = pd.read_csv('IMDb/test.csv')
            #YELP
            dataset_dict['yelp'] = {}
            dataset_dict['yelp']['train']  = pd.read_csv('yelp/train.csv')
            dataset_dict['yelp']['test'] = pd.read_csv('yelp/test.csv')
            #SST2
            dataset_dict['sst2'] = {}
            dataset_dict['sst2']['train'] = pd.read_csv('sst2/train.csv',usecols=['sentence','label']).rename(columns={'sentence':'text'})
            dataset_dict['sst2']['val'] = pd.read_csv('sst2/val.csv',usecols=['sentence','label']).rename(columns={'sentence':'text'})
            dataset_dict['sst2']['test'] = pd.read_csv('sst2/test.csv',usecols=['sentence','label']).rename(columns={'sentence':'text'})

        #Sentiment Analysis 3: Sarcasm
        os.chdir('../sarcasm')
        if 'sarcasm' in self.subset:
            #SemEval 2018 
            dataset_dict['semeval_A'] = {}
            dataset_dict['semeval_B'] = {}
            dataset_dict['semeval_A']['train'] = pd.read_table('SemEval/datasets/train/SemEval2018-T3-train-taskA.txt',
                                    usecols=['Label','Tweet text']).rename(columns = {'Label':'label','Tweet text':'text'})
            dataset_dict['semeval_B']['train'] = pd.read_table('SemEval/datasets/train/SemEval2018-T3-train-taskB.txt',
                                                usecols=['Label','Tweet text']).rename(columns = {'Label':'label','Tweet text':'text'})
            dataset_dict['semeval_A']['test'] = pd.DataFrame()
            dataset_dict['semeval_B']['test'] = pd.DataFrame()
            dataset_dict['semeval_A']['test']['text'] = pd.read_table('SemEval/datasets/test_TaskA/SemEval2018-T3_input_test_taskA.txt',
                                                usecols=['tweet text'])
            dataset_dict['semeval_B']['test']['text'] = pd.read_table('SemEval/datasets/test_TaskB/SemEval2018-T3_input_test_taskB.txt',
                                                usecols=['tweet text'])
            dataset_dict['semeval_A']['test']['label'] = pd.read_table('SemEval/datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt',
                                                usecols=['Label'])
            dataset_dict['semeval_B']['test']['label'] = pd.read_table('SemEval/datasets/goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt',
                                                usecols=['Label'])
            #SARC V1.0 balanced dataset
            dataset_dict['sarc'] = {}
            dataset_dict['sarc']['train'] = pd.read_csv('SARC/train-balanced.csv',
                    sep='\t',
                    header=None,
                    usecols=[0,1]).rename(columns={0:'label',1:'text'})
            dataset_dict['sarc']['test'] = pd.read_csv('SARC/test-balanced.csv',
                    sep='\t',
                    header=None,
                    usecols=[0,1]).rename(columns={0:'label',1:'text'})
        os.chdir('../../..') #Return to home directory
        
        return dataset_dict