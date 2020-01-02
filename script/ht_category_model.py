import numpy as np
import optuna
import pandas as pd
from random import randint
import lightgbm as lgb
from catboost import CatBoost
from catboost import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tqdm import tqdm

def get_cat_model(train, valid, hp):  
    c_text = ['keyword', 'location']
    X_train, X_valid, = train[c_text].values, valid[c_text].values
    y_train, y_valid = train.target, valid.target
    
    train_pool = Pool(X_train, label=y_train)
    valid_pool = Pool(X_valid, label=y_valid)

    model = CatBoost(hp)
    model.fit(train_pool, logging_level='Silent')
    
    y_pred = model.predict(valid_pool, prediction_type='Class')
    return f1_score(y_valid, y_pred, average=None)[0]

def objective(trial):    
    hp = {
        'used_ram_limit': '3gb',
        'eval_metric': 'F1',
        'verbose': None,
        'silent': False,
        'learning_rate': 0.3,
        'objective': trial.suggest_categorical('objective', ['Logloss', 'CrossEntropy']),
        'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 1, 12),
        'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
        'random_state': trial.suggest_int('random_state', 1, 100)
    }

    if hp['bootstrap_type'] == 'Bayesian':
        hp['bagging_temperature'] = trial.suggest_uniform('bagging_temperature', 0, 10)
    elif hp['bootstrap_type'] == 'Bernoulli':
        hp['subsample'] = trial.suggest_uniform('subsample', 0.1, 1)
    
    return cross_validation(hp)

def cross_validation(hp):
    eval_text_valid = []
    
    for i in tqdm(range(1, 6)):
        trainfile = './fact/train_cv_{}.csv'.format(i)
        validfile = './fact/valid_cv_{}.csv'.format(i) 
        df_train = pd.read_csv(trainfile)
        df_valid = pd.read_csv(validfile)
        
        eval_text_valid.append(get_cat_model(df_train, df_valid, hp))
        
    return np.mean(eval_text_valid)

def main():
    # optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
    
    hist_df = study.trials_dataframe()
    hist_df.to_csv("optuna_merge.csv")

if __name__ == '__main__':
    main()