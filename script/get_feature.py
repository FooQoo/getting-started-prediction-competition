import argparse
import category_encoders as ce
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from itertools import chain, combinations
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

if __name__ == '__main__':
    
    # read args
    parser = argparse.ArgumentParser(description='train model.')
    parser.add_argument('-i', help='input filename for train', default='./data/train.csv')
    parser.add_argument('-t', help='input filename for test', default='./data/test_leak.csv')
    parser.add_argument('--splitsize', help='ratio for training and validation data', default=0.1, type=float)
    parser.add_argument('--seed', help='random seed for split train and valid', default=0, type=int)
    args = parser.parse_args()
    
    trainfile, testfile, split_size, seed = args.i, args.t, args.splitsize, args.seed
    
    # input dataframe
    df = pd.read_csv(trainfile)
    df = df.sample(df.shape[0], random_state=seed).reset_index(drop=True) # shuffle
    df_test = pd.read_csv(testfile)
    df_test = df_test.sample(df_test.shape[0], random_state=seed).reset_index(drop=True)
    
    pos, neg = df[df.target==1].reset_index(drop=True), df[df.target==0].reset_index(drop=True)
    b_pos, b_neg = list(np.array_split(range(pos.shape[0]), 5)), list(np.array_split(range(neg.shape[0]), 5))
    
    def fit_transform(df_train, df_vt):
        ppt = PreprocessText()
        X_train_text = ppt.fit_transform(df_train.text.values.tolist())
        X_test_text = ppt.transform(df_vt.text.values.tolist())
        columns = ['keyword', 'location']
        encoder = ce.CatBoostEncoder()
        encoder.fit(df_train[columns], df_train.target)
        X_train_cat = encoder.transform(df_train[columns])
        X_test_cat = encoder.transform(df_vt[columns])
        
        df_train_output = pd.concat(
        [
            df_train.loc[:, ['id']], 
            X_train_text, 
            X_train_cat, 
            df_train.loc[:, ['target']]
        ], 
        axis=1 
        )
        
        df_test_output = pd.concat(
            [
                df_vt.loc[:, ['id']], 
                X_test_text, 
                X_test_cat, 
                df_vt.loc[:, ['target']]
            ], 
            axis=1
        )
        return df_train_output, df_test_output
    
    
    ## for cross validation 
    for i in range(5):
        b_pos_ex_i = list(chain.from_iterable([b_pos[j] for j in range(5) if i!=j]))
        b_neg_ex_i = list(chain.from_iterable([b_neg[j] for j in range(5) if i!=j]))
        df_train = pd.concat([pos.loc[b_pos_ex_i], neg.loc[b_neg_ex_i]], axis=0).reset_index(drop=True)
        df_valid = pd.concat([pos.loc[b_pos[i]], neg.loc[b_neg[i]]], axis=0).reset_index(drop=True)

        df_train_cv_output, df_valid_cv_output = fit_transform(df_train, df_valid)

        print('Start export cv Step.{}'.format(i+1))
        df_train_cv_output.to_csv('./fact/train_cv_{}.csv'.format(i+1), index=None)
        print('Success export train_cv_{}.csv'.format(i+1))
        df_valid_cv_output.to_csv('./fact/valid_cv_{}.csv'.format(i+1), index=None)
        print('Success export valid_cv_{}.csv\nFinish export cv. '.format(i+1))
    
    ## for split
    df_train_output, df_test_output = fit_transform(df, df_test)
    
    print('Start export.')
    df_train_output.to_csv('./fact/train.csv', index=None)
    print('Success export train.csv')
    df_test_output.to_csv('./fact/test.csv', index=None)
    print('Success export test.csv\nFinish export')
    
"""
def over_sampling(df):
    neg, pos = df.target.value_counts().values
    num_samples = pos if neg < pos else neg
    df_neg = df[df.target==0].sample(num_samples, replace=True)
    df_pos = df[df.target==1].sample(num_samples, replace=True)
    return pd.concat([df_neg, df_pos], axis=0)
"""