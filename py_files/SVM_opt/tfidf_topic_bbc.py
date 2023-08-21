import wandb
import nltk
nltk.download('stopwords',quiet=True,download_dir='/data/leuven/344/vsc34470/miniconda3/envs/conda_benchmark/nltk_data')
nltk.download('omw-1.4',quiet=True,download_dir='/data/leuven/344/vsc34470/miniconda3/envs/conda_benchmark/nltk_data')
nltk.download('punkt',quiet=True,download_dir='/data/leuven/344/vsc34470/miniconda3/envs/conda_benchmark/nltk_data')
nltk.download('wordnet',quiet=True,download_dir='/data/leuven/344/vsc34470/miniconda3/envs/conda_benchmark/nltk_data');


import os
import sys
sys.path.append(os.getcwd())
os.environ['WANDB_DIR']= '/lustre1/scratch/344/vsc34470/wandb'
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
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
from datetime import datetime

import yaml

SEED=int(sys.argv[1])
OPT_ITER=7

path= '/lustre1/scratch/344/vsc34470/results/svm/'
dl = DataLoader(['topic'])
data = dl.load()

tweet_preprocessor = Preprocessor(is_tweet=True)
preprocessor = Preprocessor()

#We are not interested in the test sets for hyperparameter optimization
train_carer, val_carer, _ =data_splitter(data['bbc'],
                                 preprocessor, 
                                 create_val_set=True,   #No validation set is provided
                                 seed=SEED)


#Load the template yaml sweep config file for logistic regression
#If the value range for an hyperparameter needs to be changed, better to do it in the .yaml file than in a notebook
with open("config/svm_sweep.yaml", 'r') as stream:
    sweep_config = yaml.safe_load(stream)

#Don't forget to name the sweep instance
name = 'svm_tfidf_bbc_topic' #change here
sweep_config['name'] =  name

#Generate a sweep_id
sweep_id = wandb.sweep(sweep_config, project="bayes_svm")

def train_tfidf(config = None,
          train=train_carer, #Change here
          val=val_carer): #change here
    '''
    Generic WandB function to conduct hyperparameter optimization with tf-idf vectorizer
    '''
    # Initialize a new wandb run
    with wandb.init(config=config,group=name) as run:
        save=path+name+sweep_id+'-model-'+run.name+str(datetime.now().time()).replace(':','')+'sav'
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        vec = TfidfVectorizer()
        clf = LinearSVC(C=config.C, 
                  random_state=SEED,
                  class_weight='balanced')        
        #Create the pipeline
        pipe = Pipeline([('vectorizer',vec),('clf',clf)])
        #Fit the pipeline
        pipe.fit(train['text'],train['label'])
        
        #Make predictions
        pred_val = pipe.predict(val['text'])
        #pred_prob_val = pipe.predict_proba(val['text'])[:,1]
        accuracy = accuracy_score(val['label'],pred_val)
        f1_macro = f1_score(val['label'],pred_val,average='macro')
        if train['label'].nunique() <=2:
            aucpc =  average_precision_score(val['label'],pred_prob_val)
            auc = roc_auc_score(val['label'],pred_prob_val)
        else:
            aucpc = '-'
            auc = '-'
        #Log metrics on WandB
        wandb.log({"accuracy": accuracy, 'SEED':SEED, "f1 macro":f1_macro, "AUC-PC":aucpc, 'AUC':auc })
        pickle.dump(pipe, open(save, 'wb'))

#Track emissions
tracker = EmissionsTracker(project_name=name,log_level='warning', measure_power_secs=300,
                           output_file='output/emissions_hyperopt.csv')
#Launch the agent
tracker.start()
wandb.agent(sweep_id, train_tfidf,count=OPT_ITER) #Count : number of iterations
tracker.stop()

