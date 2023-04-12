import wandb
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
import os
import sys
sys.path.append(os.getcwd())

import pickle
import warnings
import io
import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker
import yaml
from util.dataloader import DataLoader
from preprocessing.preprocessor import Preprocessor
from util.datasplitter import data_splitter
from preprocessing.fasttext_embeddings import FastTextEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
from datetime import datetime

import yaml


SEED=int(sys.argv[1])
OPT_ITER=7

path= 'results/logistic_regression/'
dl = DataLoader(['emotion'])
data = dl.load()


preprocessor = Preprocessor()

#We are not interested in the test sets for hyperparameter optimization
train_carer, val_carer, _ = data_splitter(data['silicone'],
                                 preprocessor, 
                                 create_val_set=True,   #No validation set is provided
                                 seed=SEED)

fasttext = FastTextEmbeddings()
fasttext.load_model('fastText/cc.en.300.bin')

embedded_train_eval_emotion = fasttext.generate_sentence_embeddings(train_carer['text'])
embedded_val_eval_emotion = fasttext.generate_sentence_embeddings(val_carer['text'])
embedded_train_eval_emotion['label'] = train_carer['label'].to_list()
embedded_val_eval_emotion['label'] = val_carer['label'].to_list()


#Load the template yaml sweep config file for logistic regression
#If the value range for an hyperparameter needs to be changed, better to do it in the .yaml file than in a notebook
with open("config/lr_sweep.yaml", 'r') as stream:
    sweep_config = yaml.safe_load(stream)

#Don't forget to name the sweep instance
name = 'lr_ft_silicone_emotion' #change here
sweep_config['name'] =  name

#Generate a sweep_id
sweep_id = wandb.sweep(sweep_config, project="bayes_lr")

def train_fasttext(config = None,
          train=embedded_train_eval_emotion, #Change here
          val=embedded_val_eval_emotion): #change here
    '''
    Generic WandB function to conduct hyperparameter optimization with tf-idf vectorizer
    '''
    # Initialize a new wandb run
    with wandb.init(config=config,group=name):
        save=path+name+sweep_id+str(datetime.now().time()).replace(':','')+'sav'
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        vec = TfidfVectorizer()
        clf = LogisticRegression(C = config.C,
                                 penalty = config.penalty,
                                 solver = config.solver,
                                 random_state=SEED,
                                 class_weight='balanced') #set the hyperparams here

        #Create the pipeline
        pipe = Pipeline([('clf',clf)])
        #Fit the pipeline
        pipe.fit(train.fillna(0).drop(['label'],axis=1),train['label'])

        #Make predictions
        pred_val = pipe.predict(val.fillna(0).drop(['label'],axis=1))
        pred_prob_val = pipe.predict_proba(val.fillna(0).drop(['label'],axis=1))[:,1]
        accuracy = accuracy_score(val['label'],pred_val)
        f1_macro = f1_score(val['label'],pred_val,average='macro')
        if train['label'].nunique() <=2:
            aucpc =  average_precision_score(val['label'],pred_prob_val)
            auc = roc_auc_score(val['label'],pred_prob_val)
        else:
            aucpc = '-'
            auc = '-'
        #Log metrics on WandB
        wandb.log({"accuracy": accuracy, "f1 macro":f1_macro, "AUC-PC":aucpc, 'AUC':auc,'SEED':SEED })
        pickle.dump(pipe, open(save, 'wb'))

#Track emissions
tracker = EmissionsTracker(project_name=name,log_level='warning', measure_power_secs=300,
                           output_file='output/emissions_logreg.csv')
#Launch the agent
tracker.start()
wandb.agent(sweep_id,  train_fasttext,count=OPT_ITER) #Count : number of iterations
tracker.stop()

