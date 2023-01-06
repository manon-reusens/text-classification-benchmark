#Import packages
import pandas as pd
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
            a dictionary with the names of the datasets as keys and the corresponding dataframes as values.
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
            #FakeNewsNet
            if len(os.listdir('FakeNewsNet'))!=0: 
            # The FakeNewsData has been added following the intructions in this repository : https://github.com/KaiDMML/FakeNewsNet 
            #FakeNewsNet Politifact
                # real_path = "FakeNewsNet/code/fakenewsnet_dataset/politifact/real/"
                # fake_path = "FakeNewsNet/code/fakenewsnet_dataset/politifact/fake/"
                # real_text = self.retrieve_text_path(real_path)
                # fake_text = self.retrieve_text_path(fake_path)    
                # politifact_label = [1] * len(real_text) + [0] * len(fake_text)        
                # dataset_dict['politifact'] = pd.DataFrame({'text':real_text+fake_text,'label':politifact_label}).reset_index(drop=True)
                #FakeNewsNet GossipCop
                real_path = "FakeNewsNet/code/fakenewsnet_dataset/gossipcop/real/"
                fake_path = "FakeNewsNet/code/fakenewsnet_dataset/gossipcop/fake/"
                real_text = self.retrieve_text_path(real_path)
                fake_text = self.retrieve_text_path(fake_path)    
                gossipcop_label = [1] * len(real_text) + [0] * len(fake_text)        
                dataset_dict['gossipcop'] = pd.DataFrame({'text':real_text+fake_text,'label':gossipcop_label}).reset_index(drop=True)
            #CoAID
            real = pd.read_csv('CoAID/Real.csv')
            fake = pd.read_csv('CoAID/Fake.csv')
            real_text = real.fillna('').apply(lambda row :  row['content'] if row['content']!='' 
                                     else row['abstract'] if row['abstract']!=''
                                     else row['title'],axis=1).to_list()
            fake_text = fake.fillna('').apply(lambda row :  row['content'] if row['content']!='' 
                                     else row['abstract'] if row['abstract']!=''
                                     else row['title'],axis=1).to_list()
            coaid_label = [1] * len(real_text) + [0] * len(fake_text)  
            dataset_dict['CoAID'] = pd.DataFrame({'text':real_text+fake_text,'label':coaid_label}).reset_index(drop=True)
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
            dataset_dict['agnews']['train'] = pd.read_csv('agnews/train.csv')
            dataset_dict['agnews']['test'] = pd.read_csv('agnews/test.csv')
            #Web Of Science, WOS11967
            dataset_dict['WOS']=pd.read_csv('WOS/WOStrain.csv',usecols=['input_data','label_level_1']).rename(columns={'input_data':'text', 'label_level_1':'label'})
        
        #Sentiment Analysis 1 : Emotion
        os.chdir('../sentiment/emotion')
        if 'emotion' in self.subset:
            #Tweet Eval : Emotion
            dataset_dict['tweetEval'] = {}
            dataset_dict['tweetEval']['train'] = pd.DataFrame()
            dataset_dict['tweetEval']['val'] = pd.DataFrame()
            dataset_dict['tweetEval']['test'] = pd.DataFrame()
            dataset_dict['tweetEval']['train']['label'] = pd.read_csv('tweetEval/train_labels.csv',header=None)
            dataset_dict['tweetEval']['val']['label'] = pd.read_csv('tweetEval/val_labels.csv',header=None)
            dataset_dict['tweetEval']['test']['label'] = pd.read_csv('tweetEval/test_labels.csv',header=None)
            dataset_dict['tweetEval']['train']['text']= pd.read_csv('tweetEval/train_text.csv',header=None)
            dataset_dict['tweetEval']['val']['text']= pd.read_csv('tweetEval/val_text.csv',header=None)
            dataset_dict['tweetEval']['test']['text'] = pd.read_csv('tweetEval/test_text.csv',header=None)
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
            dataset_dict['imdb']['train'] = pd.read_csv('imdb/train.csv')
            dataset_dict['imdb']['test'] = pd.read_csv('imdb/test.csv')
            #movie_review
            dataset_dict['movie_review']={}
            with open('movie_review/rt-polarity.pos') as f:
                positive = [line.rstrip('\n') for line in f]

            with open('movie_review/rt-polarity.neg') as f:
                negative = [line.rstrip('\n') for line in f]
            moviereview_label = [1] * len(positive) + [0] * len(negative)
            dataset_dict['movie_review'] = pd.DataFrame({'text':positive+negative,'label':moviereview_label}).reset_index(drop=True)

            #SST2
            dataset_dict['sst2'] = {}
            dataset_dict['sst2']['train'] = pd.read_csv('sst2/train.csv',usecols=['sentence','label']).rename(columns={'sentence':'text'})
            dataset_dict['sst2']['test'] = pd.read_csv('sst2/val.csv',usecols=['sentence','label']).rename(columns={'sentence':'text'}) #we will use the development set as test set
            #dataset_dict['sst2']['test'] = pd.read_csv('sst2/test.csv',usecols=['sentence','label']).rename(columns={'sentence':'text'})

        #Sentiment Analysis 3: Sarcasm
        os.chdir('../sarcasm')
        if 'sarcasm' in self.subset:
            #SemEval 2018 
            dataset_dict['SemEval_A'] = {}
            # dataset_dict['SemEval_B'] = {}
            dataset_dict['SemEval_A']['train'] = pd.read_csv('SemEval/train-taskA.csv',
                                    usecols=['Label','Tweet text']).rename(columns = {'Label':'label','Tweet text':'text'})
            # dataset_dict['SemEval_B']['train'] = pd.read_csv('SemEval/train-taskB.csv',
                                                # usecols=['Label','Tweet text']).rename(columns = {'Label':'label','Tweet text':'text'})
            dataset_dict['SemEval_A']['test'] = pd.DataFrame()
            # dataset_dict['SemEval_B']['test'] = pd.DataFrame()
            dataset_dict['SemEval_A']['test']['text'] = pd.read_csv('SemEval/test-taskA.csv',
                                                usecols=['tweet text'])
            # dataset_dict['SemEval_B']['test']['text'] = pd.read_csv('SemEval/test-taskB.csv',
                                                # usecols=['tweet text'])

            dataset_dict['SemEval_A']['test']['label'] = pd.read_csv('SemEval/gold_test_taskA_emoji.csv',
                                                usecols=['Label'])
            # dataset_dict['SemEval_B']['test']['label'] = pd.read_csv('SemEval/gold_test_taskB_emoji.csv',
                                                # usecols=['Label'])
            #Sarcasm_news_headline
            dataset_dict['SNH'] = {}
            dataset_dict['SNH']['train']=pd.read_csv('SNH/SNHtrain.csv',usecols=['headline','is_sarcastic']).rename(columns={'headline':'text', 'is_sarcastic':'label'})
            dataset_dict['SNH']['test']=pd.read_csv('SNH/SNHtest.csv',usecols=['headline','is_sarcastic']).rename(columns={'headline':'text', 'is_sarcastic':'label'})
            
            #iSarcasm
            dataset_dict['iSarcasm'] = {}
            dataset_dict['iSarcasm']['train'] = pd.read_csv('iSarcasm/train.En.csv',
                                                             usecols=['tweet','sarcastic']).rename(columns={'tweet':'text','sarcastic':'label'})
            dataset_dict['iSarcasm']['train']['text']= dataset_dict['iSarcasm']['train']['text'].apply(str)
            dataset_dict['iSarcasm']['test'] = pd.read_csv('iSarcasm/task_A_En_test.csv').rename(columns={'sarcastic':'label'})

        #Return to home directory
        os.chdir('../../..') 
        
        return dataset_dict
