# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:41:10 2022

@author: u0146965
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from wandb.keras import WandbCallback
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from sklearn import metrics
import numpy as np
import tensorflow as tf

#put here the code for a CNN hyperparameter sweep
class CNN():
    
    def __init__(self, config, train_label, train_text, validation_label, validation_text, test_label, test_text):
        """
        The CNN class will build a customized CNN  based on the given parameters in the config.
        Next, the class will also make predictions on the test set or validation set, depending on the given parameter.
        
        config= the config file through wandb
        train= the training set, including the labels and text
        validation= the validation set, including the labels and text
        test= the test set, including the labels and text 
        """
        self.config= config
        self.train_label= train_label
        self.validation_label=validation_label
        self.test_label=test_label
        self.train_text= train_text
        self.validation_text= validation_text
        self.test_text= test_text


    def build_network(self, word_index, emb_dim, emb_matrix, max_len):
        if max_len<32:
            max_len=32 #minmimum length required for the CNN to work
            embedding_layer = Embedding(len(word_index) + 1, emb_dim, weights=[emb_matrix], input_length=max_len, trainable=False)
            model=Sequential()
            model.add(embedding_layer)
            model.add(Conv1D(filters=self.config.filters,kernel_size=3,activation='relu',padding='same',strides=1))
        else:
            
            embedding_layer = Embedding(len(word_index) + 1, emb_dim, weights=[emb_matrix], input_length=max_len, trainable=False)

            #embedding_layer= Embedding(self.vocab_size, output_dim=self.config['embedding_layer_size'], mask_zero=True, input_length=self.padded_length)
            model=Sequential()
            model.add(embedding_layer)
            model.add(Conv1D(filters=self.config.filters,kernel_size=3,activation='relu',padding='valid',strides=1))if max_len<32:

        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=self.config.filters,kernel_size=4,activation='relu',padding='valid',strides=1))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=self.config.filters,kernel_size=5,activation='relu',padding='valid',strides=1))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(self.config.drop_out))
        model.add(tf.keras.layers.Flatten())
        model.add(Dense(units=len(self.train_label.columns), activation='softmax'))
        #the dense layer has the same number of nodes as there are target variables
        return model
    
    def build_optimizer(self,decay=False): #if you want to decay the learning rate, add here extra variable
        if decay==False:
            if self.config['optimizer']=='adam':
                optimizer= Adam(learning_rate=self.config['learning_rate'])
            elif self.config['optimizer']=='sgd':
                optimizer= SGD(learning_rate=self.config['learning_rate'])
            else:
                print('optimizer not in list')
        return optimizer

    def scheduler(self, epoch,lr):
        if epoch < 20:
            return lr
        elif lr <=1e-8:
            lr=1e-8
            return lr
        else:
            return lr * tf.math.exp(-0.1)


    def compile_and_fit_model(self, model, optimizer,path, run,sweep_id,loss= 'categorical_crossentropy', metrics=['accuracy']): # add get_f1 if applicable
        #compile the model
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        #add model checkpoint and earlystopping
        es= EarlyStopping(monitor='val_loss', verbose=1, patience=10, restore_best_weights=True)
        mc = ModelCheckpoint(filepath=path+run.project +'/'+sweep_id+'-model-'+run.name+'.hdf5', monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_freq='epoch')
        LRS = LearningRateScheduler(self.scheduler, verbose=1)
        
        #fit the model
        model.fit(self.train_text, self.train_label,  validation_data=(self.validation_text, self.validation_label), epochs=self.config['epochs'], callbacks=[WandbCallback(),es, LRS,mc])#add mc if you need that
        
        
        return model

    #def one_hot_enc(self):
        #this function onehot encodes the sentences to make it ready as input for the bidirectional CNN
    #    t = Tokenizer()
    #    t.fit_on_texts(self.train.text)
    #    word_index = t.word_index
        #vocab_size = len(word_index) + 1
        
    #    train_seq_nopad = t.texts_to_sequences(self.train.text)
    #    val_seq_nopad  = t.texts_to_sequences(self.validation.text)
    #    test_seq_nopad  = t.texts_to_sequences(self.test.text)

        #the sentences will be padded to the max length, which is the longest sequence of words
    #    max_length= max(len(item) for item in train_seq_nopad)


    #    padded_train = pad_sequences(train_seq_nopad, maxlen=max_length, padding='post')
    #    padded_val = pad_sequences(val_seq_nopad, maxlen=max_length, padding='post')
    #    padded_test = pad_sequences(test_seq_nopad, maxlen=max_length, padding='post')
        
    #    return padded_train, padded_val,padded_test, word_index, max_length

    def make_predictions(self, model, mode='val'): #nclasses higher than 2 toevoegen

        if mode=='val':
            y_text=self.validation_text
            y_label=self.validation_label.idxmax(axis=1)
        elif mode=='test':
            y_text=self.test_text
            y_label= self.test_label.idxmax(axis=1)
        else:
            print('The defined mode is not an option. It should be "val" or "test".')

        predictions=model.predict(y_text)
        pos_class_pred=[pred[1] for pred in predictions ]

        predict_classes=np.argmax(predictions, axis=1)
        accuracy= metrics.accuracy_score(y_label,predict_classes)
        f1_score=metrics.f1_score(y_label,predict_classes ,average='macro')
        precision_macro= metrics.precision_score(y_label,predict_classes,average='macro')
        recall_macro= metrics.recall_score(y_label,predict_classes,average='macro')

        if len(self.train_label.columns) <=2:
            aucpc =  metrics.average_precision_score(y_label,pos_class_pred)
            auc = metrics.roc_auc_score(y_label,pos_class_pred)
        else:
            aucpc = '-'
            auc = '-'
        if mode=='val':
            return accuracy, f1_score, aucpc, auc, precision_macro, recall_macro
        if mode=='test':
            return accuracy, f1_score, aucpc, auc, precision_macro, recall_macro, predict_classes, predictions,y_label, y_text
