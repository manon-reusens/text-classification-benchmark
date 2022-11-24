#Import packages
import pandas as pd
from tqdm import tqdm
import fasttext
import fasttext.util
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

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
        fasttext_text_emb=[]
        print('Starting to generate sentence embeddings')
        for i in tqdm(corpus.index):

            if type(corpus[i])== float:
                fasttext_text_emb.append([0] * 300)
            else:
                fasttext_text_emb.append(self.model.get_sentence_vector(corpus[i]))

        embedded_corpus=pd.DataFrame(fasttext_text_emb).fillna(0)
        
        return embedded_corpus

    def one_hot_enc(self,train, validation, test):
        #this function onehot encodes the sentences to make it ready as input for the bidirectional LSTM
        t = Tokenizer()
        t.fit_on_texts(train.text)
        word_index = t.word_index
        #vocab_size = len(word_index) + 1
        
        train_seq_nopad = t.texts_to_sequences(train.text)
        val_seq_nopad  = t.texts_to_sequences(validation.text)
        test_seq_nopad  = t.texts_to_sequences(test.text)

        #the sentences will be padded to the max length, which is the longest sequence of words
        max_length= max(len(item) for item in train_seq_nopad)


        padded_train = pad_sequences(train_seq_nopad, maxlen=max_length, padding='post')
        padded_val = pad_sequences(val_seq_nopad, maxlen=max_length, padding='post')
        padded_test = pad_sequences(test_seq_nopad, maxlen=max_length, padding='post')
        
        return padded_train, padded_val,padded_test, word_index, max_length

    def generate_embedding_matrix(self, train, val, test, emb_dim):
        train, val, test, word_index, max_length = self.one_hot_enc(train, val, test)
        embedding_matrix = np.zeros((len(word_index) + 1, emb_dim)) #embedding dimension is 300
        for word, i in word_index.items():
            embedding_vector = self.model.get_word_vector(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        return train, val, test, word_index, embedding_matrix, max_length