#Load packages
import pandas as pd
import io
from tqdm import tqdm
from copy import deepcopy


class FastTextEmbeddings:

    def __init__(self):
        self.data = {}


    def load_vectors(self,file_name='fasttext/wiki-news-300d-1M.vec'):
        '''
        Generate a dictionary with word tokens as keys and vectors as values from a given fasttext file
        Args:
            file_name:the file location path
        Returns:
            fasttext_dict: a dictionary with the tokens as keys and embeddings as values
        '''
        fin = io.open(file_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
        _, _ = map(int, fin.readline().split())

        for line in fin:
            tokens = line.rstrip().split(' ')
            self.data[tokens[0]] = map(float, tokens[1:])        
        # for k in tqdm(data.keys()): 
        #     self.fasttext_dict[k] = list(data[k])
    
    def generate_sentence_embeddings(self,corpus):
        '''
        Generate text embeddings for a given corpus based fasttext word embeddings
        '''
        fasttext_text_av=[]
        print('starting to generate sentence embeddings')
        for i in tqdm(range(len(corpus))):
            sentence_emb=[]
            
            if type(corpus[i])== float:
                sentence_emb.append([0] * 300)
            else:
                for word in corpus[i].split():
                    if word in self.data.keys():
                        sentence_emb.append(list(deepcopy(self.data[word])))
            if len(sentence_emb) ==0: #still no match for words
                sentence_emb.append([0] * 300)
            avg = [float(sum(col))/len(col) for col in zip(*sentence_emb)]
            fasttext_text_av.append(avg) 
            #Question : what to do if it is empty, fill with 0? 

        embedded_corpus=pd.DataFrame(fasttext_text_av)
        
        return embedded_corpus
    
