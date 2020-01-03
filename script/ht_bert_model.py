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

def get_text_model(train, valid, hp):  
    def get_c_with_prefix(train, prefix):
        return [column for column in train.columns.tolist() if prefix == column[:len(prefix)]]

    c_vecs = get_c_with_prefix(train, 'vecs')
    
    X_train, X_valid = train[c_vecs], valid[c_vecs]
    y_train, y_valid = train.target.values, valid.target.values
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    
    def lgb_f1_score(y_hat, data):
        y_true = data.get_label()
        y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
        return 'f1', f1_score(y_true, y_hat), True

    model = lgb.train(
        hp, 
        lgb_train, 
        valid_sets=lgb_valid,
        verbose_eval=False,
        feval=lgb_f1_score,
        num_boost_round=100
    )
    
    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    y_pred_cls = y_pred >= 0.5
    return f1_score(y_valid, y_pred_cls, average=None)[0]

def objective(trial):    
    hp = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'random_state': trial.suggest_int('random_state', 1, 100)
    }
    
    return cross_validation(hp)

def cross_validation(hp):
    eval_text_valid = []
    
    vecs_by_bert = pd.read_csv('./fact/train_vecs.csv')
    
    for i in tqdm(range(1, 6)):
        trainfile = './fact/train_cv_{}.csv'.format(i)
        validfile = './fact/valid_cv_{}.csv'.format(i) 
        df_train = pd.merge(pd.read_csv(trainfile), vecs_by_bert, on='id')
        df_valid = pd.merge(pd.read_csv(validfile), vecs_by_bert, on='id')
        
        eval_text_valid.append(get_text_model(df_train, df_valid, hp))
        
    return np.mean(eval_text_valid)

def main():
    # optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
    
    hist_df = study.trials_dataframe()
    hist_df.to_csv("optuna_bert.csv")

if __name__ == '__main__':
    main()