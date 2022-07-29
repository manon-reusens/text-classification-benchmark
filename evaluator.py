#Import pakcages
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from codecarbon import EmissionsTracker


def evaluate_classifier(clf,
                        train,
                        test, 
                        vectorizer=TfidfVectorizer(),
                        print_metrics=True,
                        return_carbon=True,
                        save_model=False,
                        model_path=None,
                        carbon_path='output/emissions.csv'):
  if return_carbon:
    tracker = EmissionsTracker(project_name=model_path,log_level='warning', measure_power_secs=300,output_file=carbon_path)
    tracker.start()
  #Create a pipeline object
  pipe = Pipeline([('word_representation',vectorizer),('clf',clf)])
  #Fit the pipeline
  pipe.fit(train['text'],train['label'])
  
  #Make predictions
  predictions = pipe.predict(test['text'])
  predictions_proba = pipe.predict_proba(test['text'])[:,1]
  print()
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
  if return_carbon:
    tracker.stop()

  return metrics