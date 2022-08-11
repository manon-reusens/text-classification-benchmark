#Import pakcages
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from codecarbon import EmissionsTracker

def evaluate_classifier(clf,
                        train,
                        test, 
                        tfidf=True,
                        print_metrics=False,
                        track_carbon=False,
                        save_model=False,
                        model_path=None,
                        carbon_path='output/emissions.csv'):
  '''
  Evaluate a model on the test set after fitting it on the train set.
  Args:
    clf (object) : the classifier to use
    train (pandas.DataFrame) : the train set. It can consist of 2 columns (text and label) or more (embeddings dimensions and label)
    test  (pandas.dataFrame) : the test set. It can consist of 2 columns (text and label) or more (embeddings dimensions and label)
    tfidf (bool) : If True, uses Tf-Idf to vectorizer the input text. Else, uses the provided input embeddings.
    print_metrics (bool): If True, prints the evaluation metrics.
    track_carbon (bool) : If True, tracks the carbon emissions while executing the function.
    save_model (bool) : If True, saves the model to the desired folder
    model_path (str) : Path to the desired location to save the model as a pickle file.
    carbon_path (str) : path to the desired location for  the carbon tracker output.
  Returns:
    metrics (dict) : A dictionary of evaluation metrics.
    predictions (array): An array with the integer prediction on the test set.
  '''
  
  if track_carbon:
    tracker = EmissionsTracker(project_name=model_path,log_level='warning', measure_power_secs=300,output_file=carbon_path)
    tracker.start()

  #TfIdf pipeline
  if tfidf:
    pipe = Pipeline([('word_representation',TfidfVectorizer()),('clf',clf)])
    #Fit the pipeline
    pipe.fit(train['text'],train['label'])
    #Make predictions
    predictions = pipe.predict(test['text'])
    if type(clf)!=SVC: #SVC does not support probability prediction by default 
      predictions_proba = pipe.predict_proba(test['text'])[:,1]
  
  #No vectorization needed (e.g. text is already converted to embeddings)
  else:
    try:
      pipe = Pipeline([('clf',clf)])
      pipe.fit(train.drop(columns=['label']),train['label'])
      predictions = pipe.predict(test.drop(columns=['label']))
      if type(clf)!=SVC:
        predictions_proba = pipe.predict_proba(test.drop(columns=['label']))[:,1]
    except:
      raise ValueError('Invalid input format.')

  #Evaluate
  metrics = {}
  metrics['Accuracy'] = accuracy_score(test['label'],predictions)
  metrics['Macro F1'] = f1_score(test['label'],predictions,average='macro')
  if test['label'].nunique() <= 2 and type(clf)!=SVC: #Binary only metrics
    metrics['AUC PC'] =  average_precision_score(test['label'],predictions_proba)
    metrics['AUC ROC'] = roc_auc_score(test['label'],predictions_proba)
  else:
    metrics['AUC PC'] = '-'
    metrics['AUC ROC'] = '-'
  if print_metrics:
    print('Accuracy test set : %s'%metrics['Accuracy'])
    print('F1 score test set: %s'%metrics['Macro F1'])
    print('AUC PC  score test set : %s'%metrics['AUC PC'])
    print('AUC ROC score : %s'%metrics['AUC ROC'])
  #Save model
  if save_model:
    pickle.dump(pipe['clf'],open(model_path,'wb'))
  #Stop the tracker if needed
  if track_carbon:
    tracker.stop()

  return metrics, predictions



def get_summary_dataset(name,
                        train,
                        test,
                        embedded_train,
                        embedded_test,
                        models_dict,
                        to_csv=True,
                        track_carbon=False,
                        save_model=True):
  '''
  Stores all predictions and evaluation metrics for a dataset and a selected number of models
  versions are evaluated.
  Args:
   name (str) : the name of the dataset, used for the output files path
   train (pandas.DataFrame): the train set in non-embedded format
   test (pandas.DataFrame):  the test set in non-embedded format
   embedded_train (pandas.DataFrame): the train set in embedded format. Required if the model uses fasttext
   embedded_test (pandas.DataFrame); the test set in embedded format. Required if the model uses  fasttext
   models_dict (dict): a dictionary with the model acronyme as key (e.g. 'tfidf_lr ,'ft_rf') and the corresponding sklearn model as value          
   to_csv (bool): if True, saves the predictions and metrics DataFrames to csv
   track_carbon (bool) : If True, tracks the carbon emissions while executing the function.
   save_model (bool) : If True, saves the model to the desired folder
  Returns:
    metrics_df (pandas.DataFrame): a DataFrame with metrics as rows and models as columns
    preds_df (pandas.DataFrame) : a DataFrame with the predictions for all the models
  '''

  if not os.path.isdir('models'):
      os.makedirs('models')
  metrics_df=pd.DataFrame()
  preds_df=pd.DataFrame({'label':test['label']})
  for k in models_dict.keys():
    if not os.path.isdir('models/'+name):
        os.makedirs('models/'+name)
    if 'tfidf' in k:
      #The classifier requires tf-idf preprocessing
      metrics_tfidf, preds_df[k] = evaluate_classifier(models_dict[k],train,test, tfidf=True,
                                                      save_model=save_model,track_carbon=track_carbon, model_path='models/'+name+'/'+k+'.sav')
      metrics_df[k] = metrics_tfidf.values()
    else:
      #The classifier does not require tf-idf preprocessing (e.g. fasttext or deep learning)
      metrics_emb, preds_df[k] = evaluate_classifier(models_dict[k],embedded_train,embedded_test, tfidf=False,
                                                save_model=save_model, track_carbon=track_carbon, model_path='models/'+name+'/'+k+'.sav')
      metrics_df[k] = metrics_emb.values()
      
  if not os.path.isdir('output/metrics'):
      os.makedirs('output/metrics')
  if not os.path.isdir('output/predictions'):
      os.makedirs('output/predictions')
  if to_csv:
    metrics_df.rename(index={0:'Accuracy',1:'F1 Macro',2:'AUC PC',3: 'AUC ROC'}).to_csv('output/metrics/'+name+'.csv')
    preds_df.to_csv('output/predictions/'+name+'.csv',index=False) 
  
  return metrics_df, preds_df

