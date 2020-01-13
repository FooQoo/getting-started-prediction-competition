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

class BitermConverter:
    def __init__(self, include_docs=False, min_df=1, max_df=1.0):
        self.biterms = Counter()
        self.include_docs = include_docs
        self.min_df = min_df
        self.max_df = max_df
        
    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)
        
    def fit(self, docs):
        docs_add_biterm = []
        
        for doc in docs:
            self.biterms.update(
                ['_'.join(sorted(biterm)) 
                 for biterm in combinations(doc.split(), 2) if biterm[0] != biterm[1]
                ]
            )
        
        sum_freq = sum(self.biterms.values())
        
        for b, f in list(self.biterms.items()):
            if f < self.min_df or f/sum_freq > self.max_df:
                del self.biterms[b]
    
    def transform(self, docs):
        docs_add_biterm = []
        
        if self.include_docs:
            get_doc = lambda doc, biterms: '{0} {1}'.format(doc, ' '.join(biterms))
        else:
            get_doc = lambda doc, biterms: ' '.join(biterms)
        
        for doc in docs:
            biterms = [biterm for biterm in [
                '_'.join(sorted(biterm)) for biterm in combinations(doc.split(), 2) if biterm[0] != biterm[1]
            ] if biterm in self.biterms]
            
            docs_add_biterm.append(get_doc(doc, biterms))
            
        return docs_add_biterm

class PreprocessText:
    def __init__(self):
        self.bc = BitermConverter(include_docs=True, min_df=10, max_df=1.0)
        self.cv_word = TfidfVectorizer(min_df=5, token_pattern='\S+')
        self.cv_url = CountVectorizer(min_df=2, token_pattern='\S+')
        self.cv_hashtag = CountVectorizer(min_df=2, token_pattern='\S+')
        self.stopwords = set(stopwords.words('english'))
        self.wordnet_lemmatizer = WordNetLemmatizer()
    
    def fit(self, docs):
        docs_hashtag = [' '.join(re.findall(r'(?<=[\s^])#\w+(?=[\s$])', doc)) for doc in docs]
        docs_url = [' '.join(re.findall(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', doc)) for doc in docs]
        docs_word = self.pipe_process(docs)
        docs_word = self.bc.fit_transform(docs_word)
        self.cv_word.fit(docs_word)
        self.cv_url.fit(docs_url)
        self.cv_hashtag.fit(docs_hashtag)
        
    def transform(self, docs):
        docs_hashtag = [' '.join(re.findall(r'(?<=[\s^])#\w+(?=[\s$])', doc)) for doc in docs]
        docs_url = [' '.join(re.findall(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', doc)) for doc in docs]
        docs_word = self.pipe_process(docs)
        docs_word = self.bc.transform(docs_word)
        
        # get dataframe
        df_word = pd.DataFrame(
            self.cv_word.transform(docs_word).todense(), 
            columns=['word_%s' % column for column in self.cv_word.get_feature_names()]
        ).replace(0, np.nan)
        df_url = pd.DataFrame(
            self.cv_url.transform(docs_url).todense(), 
            columns=['url_%s' % column for column in self.cv_url.get_feature_names()]
        ).replace(0, np.nan)
        df_hashtag = pd.DataFrame(
            self.cv_hashtag.transform(docs_hashtag).todense(), 
            columns=['hashtag_%s' % column for column in self.cv_hashtag.get_feature_names()]
        ).replace(0, np.nan)
        df_docs = pd.concat([df_word, df_url, df_hashtag], axis=1)
        return df_docs
        
    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)
        
    def format_text(self, text):
        text=re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
        text=re.sub('(?<=[\s^])#\w+(?=[\s$])', '', text)
        text=re.sub('RT', '', text)
        text=re.sub(r'[!-/?:@]', '', text)#半角記号,数字,英字
        text=re.sub(r'[︰-＠]', '', text)#全角記号
        text=re.sub('\n', ' ', text)#改行文字
        return text 
    
    def pipe_process(self, docs):
        docs = [self.format_text(text) for text in docs]
        docs = [[word for word in word_tokenize(text) if word not in self.stopwords] for text in docs]
        docs = [' '.join([self.wordnet_lemmatizer.lemmatize(word.lower()) for word in doc]) for doc in docs]
        return docs

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