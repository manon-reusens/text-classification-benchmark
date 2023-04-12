import sys
import os
import wandb
from wandb.keras import WandbCallback
import pandas as pd 
sys.path.append(os.getcwd())

from codecarbon import EmissionsTracker
from util.dataloader import DataLoader
from util.datasplitter import data_splitter
from preprocessing.preprocessor import Preprocessor
from preprocessing.fasttext_embeddings import FastTextEmbeddings
from evaluator import evaluate_classifier
from py_files.bidirectional_lstm import Bidirectional_LSTM
from nltk import download
import warnings
from tensorflow.keras.models import load_model
warnings.filterwarnings('ignore')
import yaml
import tensorflow as tf
from numpy.random import seed


#Load linguistic resources 
download('stopwords',quiet=True,download_dir='/data/leuven/344/vsc34470/text-classification-benchmark')
download('omw-1.4',quiet=True,download_dir='/data/leuven/344/vsc34470/text-classification-benchmark')
download('punkt',quiet=True,download_dir='/data/leuven/344/vsc34470/text-classification-benchmark')
download('wordnet',quiet=True,download_dir='/data/leuven/344/vsc34470/text-classification-benchmark');

SEED=int(sys.argv[1])
tf.random.set_seed(SEED)
seed(SEED)
path= 'PATH'

dl = DataLoader(['fake_news'])
data = dl.load()

train_emo, val_emo, test_emo = data_splitter(data['CoAID'],
                                          0,
                                          create_val_set=True,
                                          test_split = 0.25, #Based on informations from the paper
                                          val_split = 0.2, 
                                          seed=SEED)
emb_dim= 300
train_label= pd.get_dummies(train_emo.label)
val_label=pd.get_dummies(val_emo.label)
test_label=pd.get_dummies(test_emo.label)

fasttext = FastTextEmbeddings()
fasttext.load_model('fastText/cc.en.300.bin') #change to location
train_text, val_text, test_text, word_index, embedding_matrix,max_length= fasttext.generate_embedding_matrix(train_emo,val_emo,test_emo, emb_dim)

with open("config/bilstm_sweep.yaml", 'r') as stream:
    sweep_config = yaml.safe_load(stream)

name = 'fakenews_coaid' #change here
sweep_config['name'] =  name
sweep_id = wandb.sweep(sweep_config, project="bayes_lstm")

iterations=30

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config,project='bayes_lstm', group=name) as run:
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        print(sweep_id)
        config = wandb.config
        bidir_lstm=Bidirectional_LSTM(config, train_label,train_text,val_label, val_text, test_label, test_text)
        model=bidir_lstm.build_network(word_index, emb_dim, embedding_matrix, max_length)
        print(model.summary())
        optimizer=bidir_lstm.build_optimizer()
        #compile and fit the model
        model= bidir_lstm.compile_and_fit_model(model, optimizer, path, run,sweep_id)

       # artifact=run.use_artifact('benchmark-nlp/'+ run.project +'/model-'+run.name+':latest', type='model')
       # artifact_dir=artifact.download()
        model=tf.keras.models.load_model(path+run.project +'/'+sweep_id+'-model-'+run.name+'.hdf5', compile=False) 

        #make predictions
        accuracy, f1_score, aucpc, auc= bidir_lstm.make_predictions(model)
        print('The Accuracy is', accuracy)
        print('The F1 score is', f1_score)
        print('The area under the precision recall curve is', aucpc)
        #log the accuracy etc
        wandb.log({'SEED':SEED,"best_val_accuracy": accuracy, "best_val_f1 macro":f1_score, "best_val_AUC-PC":aucpc, 'best_val_AUC':auc })


#Track emissions
tracker = EmissionsTracker(project_name=name,log_level='warning', measure_power_secs=300,
                           output_file='output/emissions_lstm.csv')
#Launch the agent
tracker.start()
wandb.agent(sweep_id, function= train, count=iterations)
tracker.stop()


