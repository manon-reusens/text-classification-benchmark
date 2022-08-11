#Import packages
import pandas as pd
from util.dataloader import DataLoader
from util.datasplitter import data_splitter
from preprocessing.preprocessor import Preprocessor
from preprocessing.fasttext_embeddings import FastTextEmbeddings
from evaluator import evaluate_classifier, get_summary_dataset
from nltk import download
import warnings
warnings.filterwarnings('ignore')

#Load model configurations with best hyperparameters
from config.best_params import best_params

#Set SEED
SEED=42

def run_task(task, 
             save_model=True,
             track_carbon=False):
    #Load data
    dl = DataLoader([task])
    task_data = dl.load()
    #Initialize preprocessor
    preprocessor=Preprocessor()
    tweet_preprocessor=Preprocessor(is_tweet=True)
    #Load fasttext 
    fasttext = FastTextEmbeddings()
    fasttext.load_model('fasttext/cc.en.300.bin')

    for k in task_data.keys():
        if k in ['SemEval_A','SemEval_B','iSarcasm','tweetEval']: #Tweet datasets
            train, _ , test = data_splitter(task_data[k],tweet_preprocessor,
                                            create_val_set=True,seed=SEED)
        else:
            train, _ , test = data_splitter(task_data[k],preprocessor,
                                            create_val_set=True,seed=SEED)
        #Generate sentence embeddings                                   
        embedded_train = fasttext.generate_sentence_embeddings(train['text']).fillna(0)
        embedded_test = fasttext.generate_sentence_embeddings(test['text']).fillna(0)
        embedded_train['label'] =  train['label'].to_list()
        embedded_test['label'] = test['label'].to_list()
        #Evaluate and save the model, save the metrics and predictions on test set as csv
        _, _ = get_summary_dataset(k,train,test, embedded_train,embedded_test,best_params[k], 
                                   save_model=save_model, track_carbon=track_carbon)
        print('Evaluation completed for dataset : ' + k)
        

if __name__=='__main__':

    #Load linguistic resources
    download('stopwords', quiet=True)
    download('omw-1.4', quiet=True)
    download('punkt', quiet=True)
    download('wordnet', quiet=True)

    run_task('fake_news')