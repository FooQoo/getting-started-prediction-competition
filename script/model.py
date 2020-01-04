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
        'num_boost_round': 1000,
        'objective': 'CrossEntropy', 
        'colsample_bylevel': 0.010419852115438836, 
        'depth': 7, 
        'boosting_type': 'Ordered', 
        'bootstrap_type': 'Bayesian', 
        'random_state': 19, 
        'bagging_temperature': 9.096903904222094
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

def get_bert_model(train, valid):
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
        return 'f1', f1_score(y_true, y_hat, average=None)[0], True

    lgbm_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'lambda_l1': 0.0003543420203502818, 
        'lambda_l2': 4.468466658809475, 
        'num_leaves': 169, 
        'feature_fraction': 0.8390907205934592, 
        'bagging_fraction': 0.8070674146918868, 
        'bagging_freq': 5, 
        'min_child_samples': 65, 
        'random_state': 5
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

def get_merge_model(train, valid):
    c_merge = ['text', 'category', 'bert']
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
    eval_bert_train, eval_bert_valid = [], []
    eval_merge_train, eval_merge_valid = [], []
    
    vecs_by_bert = pd.read_csv('./fact/train_vecs.csv')
    
    for i in tqdm(range(1, 6)):
        trainfile = './fact/train_cv_{}.csv'.format(i)
        validfile = './fact/valid_cv_{}.csv'.format(i) 
        df_train = pd.merge(pd.read_csv(trainfile), vecs_by_bert, on='id')
        df_valid = pd.merge(pd.read_csv(validfile), vecs_by_bert, on='id')
        
        # text model
        report_text = get_text_model(df_train, df_valid)
        eval_text_train.append(report_text.get_eval(True))
        eval_text_valid.append(report_text.get_eval(False))
        
        # category model
        report_cat = get_category_model(df_train, df_valid)
        eval_cat_train.append(report_cat.get_eval(True))
        eval_cat_valid.append(report_cat.get_eval(False))
        
        # bert model
        report_bert = get_bert_model(df_train, df_valid)
        eval_bert_train.append(report_bert.get_eval(True))
        eval_bert_valid.append(report_bert.get_eval(False))
        
        # merge model
        df_train_merge = pd.DataFrame(
            {'text': report_text.get_proba(True), 
             'category': report_cat.get_proba(True),
             'bert': report_bert.get_proba(True),
             'target': df_train.target})
        
        df_train_merge.to_csv('./fact/train_cv_merge_{}.csv'.format(i), index=None)
        
        df_valid_merge = pd.DataFrame(
            {'text': report_text.get_proba(False), 
             'category': report_cat.get_proba(False),
             'bert': report_bert.get_proba(False),
             'target': df_valid.target})
        
        df_valid_merge.to_csv('./fact/valid_cv_merge_{}.csv'.format(i), index=None)
        
        report_merge = get_merge_model(df_train_merge, df_valid_merge)
        eval_merge_train.append(report_merge.get_eval(True))
        eval_merge_valid.append(report_merge.get_eval(False))
        
        pd.DataFrame(
            {'id': df_valid.id, 
             'proba': report_merge.get_proba(False),
             'target': df_valid.target}).to_csv('./fact/valid_merge_{}.csv'.format(i), index=None)
    
    def print_f1(evals, modelname):
        print('f1 of train for {0}: {1:.3f} +- {2:.3f} in ({3})'.format(
            modelname,
            np.mean(evals), 
            np.std(evals), 
            ', '.join(['{:.3f}'.format(eva) for eva in evals])
        ))
        
    print_f1(eval_text_train, 'text')
    print_f1(eval_text_valid, 'text')
    print_f1(eval_cat_train, 'category')
    print_f1(eval_cat_valid, 'category')
    print_f1(eval_bert_train, 'bert')
    print_f1(eval_bert_valid, 'bert')
    print_f1(eval_merge_train, 'merge')
    print_f1(eval_merge_valid, 'merge')
    
def test():
    trainfile = './fact/train.csv'
    testfile = './fact/test.csv'
    
    vecs_train_by_bert = pd.read_csv('./fact/train_vecs.csv')
    vecs_test_by_bert = pd.read_csv('./fact/test_vecs.csv')
    df_train = pd.merge(pd.read_csv(trainfile), vecs_train_by_bert, on='id')
    df_test = pd.merge(pd.read_csv(testfile), vecs_test_by_bert, on='id')
    
    def get_report(train, test, get_model, model_name):
        report = get_model(train, test)
        print('f1 of train for {0}: {1:.3f}'.format(model_name, report.get_eval(True)))
        print('f1 of test  for {0}: {1:.3f}'.format(model_name, report.get_eval(False)))
        df_submit = pd.DataFrame(
            {'id': test.id,
            'target': (report.get_proba(False) >= 0.5).astype(int)})
        df_submit.to_csv('./output/submit_{}.csv'.format(model_name), index=None)
        return report
    
    report_text = get_report(df_train, df_test, get_text_model, 'text')
    report_cat = get_report(df_train, df_test, get_category_model, 'category')
    report_bert = get_report(df_train, df_test, get_bert_model, 'bert')
    
    # merge
    df_train_merge = pd.DataFrame(
        {
         'id': df_train.id,
         'text': report_text.get_proba(True), 
         'category': report_cat.get_proba(True),
         'bert': report_bert.get_proba(True),
         'target': df_train.target})
        
    df_test_merge = pd.DataFrame(
        {
         'id': df_test.id,
         'text': report_text.get_proba(False), 
         'category': report_cat.get_proba(False),
         'bert': report_bert.get_proba(False),
         'target': df_test.target})
        
    report_merge = get_report(df_train_merge, df_test_merge, get_merge_model, 'merge')

if __name__ == '__main__':
    cross_validation()
    test()
