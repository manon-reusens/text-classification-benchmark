import sys
import os

import wandb
from wandb.keras import WandbCallback
import pandas as pd 
sys.path.append(os.getcwd())
os.environ['WANDB_DIR']= '/lustre1/scratch/344/vsc34470/wandb/'

#Load linguistic resources 
from nltk import download
download('stopwords',quiet=True)
download('omw-1.4',quiet=True)
download('punkt',quiet=True)
download('wordnet',quiet=True)

from codecarbon import EmissionsTracker
from util.dataloader import DataLoader
from util.datasplitter import data_splitter
from preprocessing.preprocessor import Preprocessor
from preprocessing.fasttext_embeddings import FastTextEmbeddings
from evaluator import evaluate_classifier
from py_files.cnn import CNN

import warnings
from tensorflow.keras.models import load_model
warnings.filterwarnings('ignore')
import yaml
import tensorflow as tf
from numpy.random import seed


SEED=int(sys.argv[1])
tf.random.set_seed(SEED)
seed(SEED)
path= 'results/CNN/'
project_name = 'bayes_CNN'
dataset_group = 'emotion'
dataset_name ='tweetEval'
name = 'emotion_tweetEval'

dl = DataLoader([dataset_group])
data = dl.load()

silicone=data[dataset_name]
train_emo, val_emo, test_emo = data_splitter(data[dataset_name],0,create_val_set=True,seed=SEED)
emb_dim= 300
train_label= pd.get_dummies(train_emo.label)
val_label=pd.get_dummies(val_emo.label)
test_label=pd.get_dummies(test_emo.label)

fasttext = FastTextEmbeddings()
fasttext.load_model('fastText/cc.en.300.bin') #change to location
train_text, val_text, test_text, word_index, embedding_matrix,max_length= fasttext.generate_embedding_matrix(train_emo,val_emo,test_emo, emb_dim)

with open("config/CNN_sweep.yaml", 'r') as stream:
    sweep_config = yaml.safe_load(stream)

sweep_config['name'] =  name
sweep_id = wandb.sweep(sweep_config, project=project_name, entity="benchmark-nlp")

iterations=30

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config,project=project_name, group=name) as run:
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        print(sweep_id)
        config = wandb.config
        cnn=CNN(config, train_label,train_text,val_label, val_text, test_label, test_text)
        model= cnn.build_network(word_index, emb_dim, embedding_matrix, max_length)
        print(model.summary())
        optimizer=cnn.build_optimizer()
        #compile and fit the model
        model= cnn.compile_and_fit_model(model, optimizer, path, run,sweep_id)

       # artifact=run.use_artifact('benchmark-nlp/'+ run.project +'/model-'+run.name+':latest', type='model')
       # artifact_dir=artifact.download()
        model=tf.keras.models.load_model(path+run.project +'/'+sweep_id+'-model-'+run.name+'.hdf5', compile=False) 

        #make predictions
        accuracy, f1_score, aucpc, auc, prec, recall= cnn.make_predictions(model)
        print('The Accuracy is', accuracy)
        print('The F1 score is', f1_score)
        print('The area under the precision recall curve is', aucpc)
        #log the accuracy etc
        wandb.log({'SEED':SEED,"best_val_accuracy": accuracy, "best_val_f1 macro":f1_score, "best_val_AUC-PC":aucpc, 'best_val_AUC':auc,'best_val_precision': prec, 'best_val_recall':recall  })

wandb.agent(sweep_id, function= train, count=iterations)


