#Load packages
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import fasttext


class FastTextEmbeddings:

    def __init__(self,model=None):
        self.model=model


    def load_model(self,file_name='fasttext/cc.en.300.bin'):
    
        self.model = fasttext.load_model(file_name)       
    
    def generate_sentence_embeddings(self,corpus):
        '''
        Generate text embeddings for a given corpus based fasttext word embeddings
        '''
        fasttext_text_av=[]
        print('starting to generate sentence embeddings')
        for i in tqdm(corpus.index):
            sentence_emb=[]
            
            if type(corpus[i])== float:
                sentence_emb.append([0] * 300)
            else:
                for word in corpus[i].split():
                        sentence_emb.append(self.model.get_word_vector(word))
            avg = [float(sum(col))/len(col) for col in zip(*sentence_emb)]
            fasttext_text_av.append(avg) 

        embedded_corpus=pd.DataFrame(fasttext_text_av)
        
        return embedded_corpus
    
