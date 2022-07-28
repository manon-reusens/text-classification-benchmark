#Import pakcages
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing.text import TfidfVectorizer
import pickle
from codecarbon import EmissionsTracker



def evaluate_classifier(clf,
                        train,
                        test, 
                        val=None, 
                        word_representation='tf-idf',
                        return_metrics=True,
                        return_carbon=True,
                        save_model=False,
                        model_path=None):
  if return_carbon:
    tracker = EmissionsTracker(project_name=model_path,log_level='warning', measure_power_secs=300,output_file='output/emissions.csv')
    tracker.start()
  if word_representation=='tf-idf':
    vectorizer = TfidfVectorizer()
  #Create a pipeline object
  pipe = Pipeline([('word_representation',vectorizer),('clf',clf)])
  #Fit the pipeline
  pipe.fit(train['text'],train['label'])
  
  #Make predictions
  predictions = pipe.predict(test['text'])
  #Evaluate
  if return_metrics:
    print('Accuracy test set : %s'%accuracy_score(test['label'],predictions))
    print('F1 score test set: %s'%f1_score(test['label'],predictions,average='macro'))
  #Save model
  if save_model:
    pickle.dump(pipe['clf'],open(model_path,'wb'))
  #Stop the tracker if needed
  if return_carbon:
    tracker.stop()

  return predictions