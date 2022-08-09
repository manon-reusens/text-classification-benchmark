#Import packages
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import Preprocessor



def data_splitter(dataset,
                               preprocessor=Preprocessor(),
                               create_val_set=False,
                               val_split=0.2,
                               test_split=0.2,
                               seed=None):
  '''
  Util function to initiate train-(val)-test sets. TO DO  : move it to a separate python script
  '''
  if type(dataset) != dict : #The dataset has no split yet (gossipcop,politifact)
    dataset['text'] = preprocessor.preprocess(dataset)
    train, test = train_test_split(dataset,
                                                     test_size=test_split,
                                                     random_state=seed)
    if create_val_set:  #An additional validation set split is required
      train, val = train_test_split(train,
                                    test_size=val_split,
                                    random_state=seed)
      splits = (train,val,test)
    else:
      splits = (train,test)
  else: #The dataset is already split into train-(val)-test

    train = pd.DataFrame()
    train['text'] = preprocessor.preprocess(dataset['train'])
    train['label'] = dataset['train']['label']
    test = pd.DataFrame()
    test['text'] = preprocessor.preprocess(dataset['test'])
    test['label'] = dataset['test']['label']
    try:
      if create_val_set: 
        if 'validation' in dataset.keys() or 'val' in dataset.keys(): #The validation set is already given
          val = pd.DataFrame()
          val['text'] = preprocessor.preprocess(dataset['val'])
          val['label'] = dataset['val']['label']
        else:
          train, val = train_test_split(train,test_size=val_split, random_state=seed)
          
        splits = (train,val,test)          
      else:
        splits= (train,test)
    except:
      ValueError('Requested to create a validation set but no validation set found in files')
  return splits