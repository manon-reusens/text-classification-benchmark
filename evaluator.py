#Import pakcages
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from codecarbon import EmissionsTracker

def evaluate_classifier(clf,
                        train,
                        test, 
                        tfidf=True,
                        print_metrics=True,
                        track_carbon=True,
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
    return_carbon (bool) : If True, tracks the carbon emissions while executing the function.
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
    predictions_proba = pipe.predict_proba(test['text'])[:,1]
  
  #No vectorization needed (e.g. text is already converted to embeddings)
  else:
    try:
      pipe = Pipeline([('clf',clf)])
      pipe.fit(train.drop(columns=['label']),train['label'])
      predictions = pipe.predict(test.drop(columns=['label']))
      predictions_proba = pipe.predict_proba(test.drop(columns=['label']))[:,1]
    except:
      raise ValueError('Invalid input format.')


  #Evaluate
  metrics = {}
  metrics['Accuracy'] = accuracy_score(test['label'],predictions)
  metrics['Macro F1'] = f1_score(test['label'],predictions,average='macro')
  if test['label'].nunique() <= 2: #Binary only metrics
    metrics['AUC PC'] =  average_precision_score(test['label'],predictions_proba)
    metrics['AUC ROC'] = roc_auc_score(test['label'],predictions_proba)
  else:
    metrics['AUC PC'] = '-'
    metrics['AUC ROC'] = '_'
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