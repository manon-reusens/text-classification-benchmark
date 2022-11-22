#Import packages
import pandas as pd
from tqdm import tqdm
import fasttext
import fasttext.util
import os

def download(directory='/fasttext'):
    os.chdir('..') #Return to the root directory
    if not os.path.isdir(directory):
        os.makedirs(directory)
    fasttext.util.download_model('en',if_exists='ignore')

class FastTextEmbeddings:

    def __init__(self,model=None):
        self.model=model

    def load_model(self,file_name='fasttext/cc.en.300.bin'):
        #Load a FastText model
        self.model = fasttext.load_model(file_name)       
    
    def generate_sentence_embeddings(self,corpus):
        '''
        Generate sentence embeddings for a given corpus based on a FastText model
        Args:
            corpus (pandas.Series): the 'text' column of a dataset
        Returns:
            embedded_corpus (pandas.DataFrame): the 300 columns embeddings DataFrame
        '''
        fasttext_text_av=[]
        print('Starting to generate sentence embeddings')
        for i in tqdm(corpus.index):

            if type(corpus[i])== float:
                fasttext_text_emb.append([0] * 300)
            else:
                fasttext_text_emb.append(self.model.get_sentence_vector(corpus[i]))

        embedded_corpus=pd.DataFrame(fasttext_text_emb).fillna(0)
        
        return embedded_corpus
