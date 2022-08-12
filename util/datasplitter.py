#Import packages
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing.preprocessor import Preprocessor



def data_splitter(dataset,
                  preprocessor=Preprocessor(),
                  create_val_set=False,
                  val_split=0.2,
                  test_split=0.2,
                  seed=None):
  '''
  Util function to  preprocess and split a dataset into train-(val)-test sets. 
  Args:
    dataset (pandas.DataFrame): the dataset to split
    preprocessor (preprocessor.Preprocessor): the preprocessor object 
    create_val_set (bool): whether to include a validation set. 
    val_split (float): the size of the validation set as a fraction of the train set. Ignored if the validation set is pre-existing in a separate file.
    test_split (float): the size of the test set as a fraction of the whole dataset. Ignored if the test set is pre-existing in a separate file.
    seed (int): seed for reproducibility
  Returns:
    splits (tuple): tuple containing two (train-test) or three (train-val-test) datasets, all of which are pandas.DataFrame.
  '''
  if type(dataset) != dict : #The dataset has no split yet (e.g. gossipcop,politifact)
    dataset['text'] = preprocessor.preprocess(dataset)
    #Call the sklearn train_test_split function
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

  else: #The dataset consists of separate files for train-(val)-test
    train = pd.DataFrame()
    train['text'] = preprocessor.preprocess(dataset['train'])
    train['label'] = dataset['train']['label']
    test = pd.DataFrame()
    test['text'] = preprocessor.preprocess(dataset['test'])
    test['label'] = dataset['test']['label']
    try:
      if create_val_set: 
        #A validation set is requested
        if 'validation' in dataset.keys() or 'val' in dataset.keys():
          #The validation set exists already as a separate file
          val = pd.DataFrame()
          val['text'] = preprocessor.preprocess(dataset['val'])
          val['label'] = dataset['val']['label']
        else:
          #The validation set does not pre-exist and needs to be split from the train set
          train, val = train_test_split(train,test_size=val_split, random_state=seed)          
        splits = (train,val,test)          
      else:
        splits= (train,test)
    except:
      ValueError('Requested to create a validation set but no validation set found in files')
  return splits