#Import packages
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

#set seed
SEED=42

'''
This python file is used to store the best hyperparameters configurations for all models. It consists in a nested dictionary 
where the first-level keys correspond to the dataset names, and the sub-keys correspond to a machine learning model with specified best params.

Important naming convention: if the model requires tf-idf preprocessing, the character sequence 'tfidf' needs to appear in the subkey.
'''

best_params = {}
best_params['politifact'] = {'tfidf_lr':LogisticRegression(C=3.7,random_state=SEED),
                     'ft_lr':LogisticRegression(C=9.626,random_state=SEED),
                     'tfidf_rf':RandomForestClassifier(random_state=SEED),
                     'ft_rf':RandomForestClassifier(random_state=SEED),
                     'tfidf_svm':SVC(random_state=SEED),
                     'ft_svm':SVC(random_state=SEED),
                     'tfidf_xgb':XGBClassifier(random_state=SEED),
                     'ft_xgb':XGBClassifier(random_state=SEED)}

best_params['gossipcop'] = {'tfidf_lr':LogisticRegression(C=8.959,random_state=SEED),
                     'ft_lr':LogisticRegression(C=7.346,random_state=SEED),
                     'tfidf_rf':RandomForestClassifier(random_state=SEED),
                     'ft_rf':RandomForestClassifier(random_state=SEED)}

best_params['liar'] = {'tfidf_lr':LogisticRegression(C=4.376,random_state=SEED),
                     'ft_lr':LogisticRegression(C=7.82,random_state=SEED),
                     'tfidf_rf':RandomForestClassifier(random_state=SEED),
                     'ft_rf':RandomForestClassifier(random_state=SEED)}

