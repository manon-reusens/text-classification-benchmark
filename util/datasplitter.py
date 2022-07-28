#Import packages
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessor import Preprocessor



def train_val_test_preparation(dataset,
                               preprocessor=Preprocessor(),
                               val_set=False,
                               test_set=True,
                               val_split=0.8,
                               test_split=0.8,
                               seed=None):
  '''
  Util function to initiate train-(val)-test sets. TO DO  : move it to a separate python script
  '''
  if not test_set: #The dataset has no split yet (gossipcop,politifact)
    train, test = train_test_split(preprocessor.preprocess(dataset),
                                                     test_size=test_split,
                                                     random_state=seed)
    splits = (train,test)
  else:
    train = pd.DataFrame()
    train['text'] = preprocessor.preprocess(dataset['train'])
    train['label'] = dataset['train']['label']
    test = pd.DataFrame()
    test['text'] = preprocessor.preprocess(dataset['test'])
    test['label'] = dataset['test']['label']
    splits = (train,test)
    try:
      if val_set:
        val = pd.DataFrame()
        val['text'] = preprocessor.preprocess(dataset['val'])
        val['label'] = dataset['val']['label']
        splits = (train,val,test)
    except:
      ValueError('Requested to create a validation set but no validation set found in files')
  
  return splits