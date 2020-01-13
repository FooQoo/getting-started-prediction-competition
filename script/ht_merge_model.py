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

def get_merge_model(train, valid, hp):  
    c_merge = ['text', 'category', 'bert']
    X_train, X_valid, = train[c_merge].values, valid[c_merge].values
    y_train, y_valid = train.target, valid.target

    model = LogisticRegression(**hp)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_valid)
    return f1_score(y_valid, y_pred, average=None)[0]

def objective(trial):    
    hp = {
        'max_iter': 100,
        'verbose': 0,
        'n_jobs': 1,
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', 'none']),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'warm_start': trial.suggest_categorical('warm_start', [True, False]),
        'random_state': trial.suggest_int('random_state', 1, 100),
        'multi_class': trial.suggest_categorical('multi_class', ['ovr', 'auto'])
    }
    # solver
    if hp['penalty'] == 'elasticnet':
        hp['solver'] = 'saga'
        hp['l1_ratio'] = trial.suggest_uniform('l1_ratio', 0.0, 1.0)
    elif hp['penalty'] == 'none':
        hp['solver'] = 'saga'
    else:
        hp['solver'] = trial.suggest_categorical('solver', ['saga', 'liblinear'])
        
    if hp['penalty'] != 'none':
        hp['C'] =  trial.suggest_loguniform('C', 1e-8, 10.0)
        
    if hp['solver'] == 'liblinear':
        hp['intercept_scaling'] = trial.suggest_uniform('intercept_scaling', 0.0, 1)
        if hp['penalty'] == 'l2':
            hp['dual'] = trial.suggest_categorical('dual', [True, False])
    
    return cross_validation(hp)

def cross_validation(hp):
    eval_text_valid = []
    
    for i in tqdm(range(1, 6)):
        trainfile = './fact/train_cv_merge_{}.csv'.format(i)
        validfile = './fact/valid_cv_merge_{}.csv'.format(i) 
        df_train = pd.read_csv(trainfile)
        df_valid = pd.read_csv(validfile)
        
        eval_text_valid.append(get_merge_model(df_train, df_valid, hp))
        
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
