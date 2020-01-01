import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoost
from catboost import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tqdm import tqdm

class Report:
    def __init__(self, y_train_proba, y_valid_proba, y_train_eval, y_valid_eval):
        self.y_train_proba = y_train_proba
        self.y_valid_proba = y_valid_proba
        self.y_train_eval = y_train_eval
        self.y_valid_eval = y_valid_eval
        
    def get_proba(self, is_train=True):
        return self.y_train_proba if is_train else self.y_valid_proba
    
    def get_eval(self, is_train=True):
        return self.y_train_eval if is_train else self.y_valid_eval
            
def get_text_model(train, valid):  
    def get_c_with_prefix(prefix):
        return [column for column in train.columns.tolist() if prefix == column[:len(prefix)]]
    
    c_word, c_url, c_hashtag = get_c_with_prefix('word_'), get_c_with_prefix('url_'), get_c_with_prefix('hashtag_')
    
    train.fillna(0, inplace=True)
    
    X_train, X_valid = train[c_word+c_url+c_hashtag].values, valid[c_word+c_url+c_hashtag].values
    y_train, y_valid = train.target.values, valid.target.values
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    
    def lgb_f1_score(y_hat, data):
        y_true = data.get_label()
        y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
        return 'f1', f1_score(y_true, y_hat), True
    
    lgbm_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'lambda_l1': 3.642434329823594, 
        'lambda_l2': 1.0401748765492007e-08, 
        'num_leaves': 172, 
        'feature_fraction': 0.8251431673667773, 
        'bagging_fraction': 0.9755605959841563, 
        'bagging_freq': 2, 
        'min_child_samples': 5, 
        'random_state': 68
    }

    model = lgb.train(
        lgbm_params, 
        lgb_train, 
        valid_sets=lgb_valid,
        verbose_eval=False,
        feval=lgb_f1_score,
        num_boost_round=300,
    )
    
    def get_pred_f1(X, y):
        y_pred = model.predict(X, num_iteration=model.best_iteration)
        y_pred_cls = y_pred >= 0.5
        return y_pred, f1_score(y, y_pred_cls, average=None)[0]
    
    y_train_proba, train_f1 = get_pred_f1(X_train, y_train)
    y_valid_proba, valid_f1 = get_pred_f1(X_valid, y_valid)
    
    report = Report(y_train_proba, y_valid_proba, train_f1, valid_f1)
    
    return report

def get_category_model(train, valid):
    c_text = ['keyword', 'location']
    X_train, X_valid, = train[c_text].values, valid[c_text].values
    y_train, y_valid = train.target, valid.target
    
    train_pool = Pool(X_train, label=y_train)
    valid_pool = Pool(X_valid, label=y_valid)

    params = {
        'used_ram_limit': '3gb',
        'eval_metric': 'F1',
        'verbose': None,
        'silent': False,
        'learning_rate': 0.3,
        'objective': 'Logloss', 
        'colsample_bylevel': 0.010190258052147128, 
        'depth': 7, 
        'boosting_type': 'Plain', 
        'bootstrap_type': 'MVS', 
        'random_state': 3
    }

    model = CatBoost(params)
    model.fit(train_pool, logging_level='Silent')
    
    def get_pred_f1(X_pool, y):
        y_pred = model.predict(X_pool, prediction_type='Class')
        y_pred_proba = model.predict(X_pool, prediction_type='Probability')[:, 1]
        return y_pred_proba, f1_score(y, y_pred, average=None)[0]
    
    y_train_proba, train_f1 = get_pred_f1(train_pool, y_train)
    y_valid_proba, valid_f1 = get_pred_f1(valid_pool, y_valid)
    
    report = Report(y_train_proba, y_valid_proba, train_f1, valid_f1)
    
    return report

def get_merge_model(train, valid):
    c_merge = ['text', 'category']
    X_train, X_valid, = train[c_merge].values, valid[c_merge].values
    y_train, y_valid = train.target, valid.target
    
    params = {
        'solver': 'saga',
        'max_iter': 100,
        'verbose': 0,
        'n_jobs': -1,
        'penalty': 'elasticnet', 
        'class_weight': None, 
        'warm_start': False, 
        'random_state': 25, 
        'multi_class': 'ovr', 
        'l1_ratio': 0.05374396158679226, 
        'C': 0.0022149471434149024
    }
    
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    def get_pred_f1(X, y):
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X, )[:, 1]
        return y_pred_proba, f1_score(y, y_pred, average=None)[0]
    
    y_train_proba, train_f1 = get_pred_f1(X_train, y_train)
    y_valid_proba, valid_f1 = get_pred_f1(X_valid, y_valid)
    
    report = Report(y_train_proba, y_valid_proba, train_f1, valid_f1)
    
    return report

def cross_validation():
    eval_text_train, eval_text_valid = [], []
    eval_cat_train, eval_cat_valid = [], []
    eval_merge_train, eval_merge_valid = [], []
    
    for i in tqdm(range(1, 6)):
        trainfile = './fact/train_cv_{}.csv'.format(i)
        validfile = './fact/valid_cv_{}.csv'.format(i) 
        df_train = pd.read_csv(trainfile)
        df_valid = pd.read_csv(validfile)
        
        # text model
        report_text = get_text_model(df_train, df_valid)
        eval_text_train.append(report_text.get_eval(True))
        eval_text_valid.append(report_text.get_eval(False))
        
        # category model
        report_cat = get_category_model(df_train, df_valid)
        eval_cat_train.append(report_cat.get_eval(True))
        eval_cat_valid.append(report_cat.get_eval(False))
        
        # merge model
        df_train_merge = pd.DataFrame(
            {'text': report_text.get_proba(True), 
             'category': report_cat.get_proba(True),
             'target': df_train.target})
        
        df_train_merge.to_csv('./fact/train_cv_merge_{}.csv'.format(i), index=None)
        
        df_valid_merge = pd.DataFrame(
            {'text': report_text.get_proba(False), 
             'category': report_cat.get_proba(False),
             'target': df_valid.target})
        
        df_valid_merge.to_csv('./fact/valid_cv_merge_{}.csv'.format(i), index=None)
        
        report_merge = get_merge_model(df_train_merge, df_valid_merge)
        eval_merge_train.append(report_merge.get_eval(True))
        eval_merge_valid.append(report_merge.get_eval(False))
        
    print('f1 of train for text: {0:.3f} +- {1:.3f} in ({2})'.format(
        np.mean(eval_text_train), 
        np.std(eval_text_train), 
        ', '.join(['{:.3f}'.format(eva) for eva in eval_text_train])
    ))
    print('f1 of valid for text: {0:.3f} +- {1:.3f} in ({2})'.format(
        np.mean(eval_text_valid), 
        np.std(eval_text_valid), 
        ', '.join(['{:.3f}'.format(eva) for eva in eval_text_valid])
    ))
    print('f1 of train for category: {0:.3f} +- {1:.3f} in ({2})'.format(
        np.mean(eval_cat_train), 
        np.std(eval_cat_train), 
        ', '.join(['{:.3f}'.format(eva) for eva in eval_cat_train])
    ))
    print('f1 of valid for category: {0:.3f} +- {1:.3f} in ({2})'.format(
        np.mean(eval_cat_valid), 
        np.std(eval_cat_valid), 
        ', '.join(['{:.3f}'.format(eva) for eva in eval_cat_valid])
    ))
    print('f1 of train for merge: {0:.3f} +- {1:.3f} in ({2})'.format(
        np.mean(eval_cat_train), 
        np.std(eval_cat_train), 
        ', '.join(['{:.3f}'.format(eva) for eva in eval_cat_train])
    ))
    print('f1 of valid for merge: {0:.3f} +- {1:.3f} in ({2})'.format(
        np.mean(eval_merge_valid), 
        np.std(eval_merge_valid), 
        ', '.join(['{:.3f}'.format(eva) for eva in eval_merge_valid])
    ))
    
def test():
    trainfile = './fact/train.csv'
    testfile = './fact/test.csv'
    df_train = pd.read_csv(trainfile)
    df_test = pd.read_csv(testfile)
    
    # text
    report_text = get_text_model(df_train, df_test)
    print('f1 of train for text: {:.3f}'.format(report_text.get_eval(True)))
    print('f1 of test  for text: {:.3f}'.format(report_text.get_eval(False)))
    df_text_submit = pd.DataFrame(
        {'id': df_test.id,
        'target': (report_text.get_proba(False) >= 0.5).astype(int)})
    df_text_submit.to_csv('./output/submit_text.csv', index=None)
    
    # category
    report_cat = get_category_model(df_train, df_test)
    print('f1 of train for category: {:.3f}'.format(report_cat.get_eval(True)))
    print('f1 of test  for category: {:.3f}'.format(report_cat.get_eval(False)))
    df_cat_submit = pd.DataFrame(
        {'id': df_test.id,
        'target': (report_cat.get_proba(False) >= 0.5).astype(int)})
    df_cat_submit.to_csv('./output/submit_cat.csv', index=None)
    
    # merge
    df_train_merge = pd.DataFrame(
        {'text': report_text.get_proba(True), 
         'category': report_cat.get_proba(True),
         'target': df_train.target})
        
    df_test_merge = pd.DataFrame(
        {'text': report_text.get_proba(False), 
         'category': report_cat.get_proba(False),
         'target': df_test.target})
        
    report_merge = get_merge_model(df_train_merge, df_test_merge)
    print('f1 of train for merge: {:.3f}'.format(report_merge.get_eval(True)))
    print('f1 of test  for merge: {:.3f}'.format(report_merge.get_eval(False)))
    df_merge_submit = pd.DataFrame(
        {'id': df_test.id,
        'target': (report_merge.get_proba(False) >= 0.5).astype(int)})
    df_merge_submit.to_csv('./output/submit_merge.csv', index=None)

if __name__ == '__main__':
    cross_validation()
    test()