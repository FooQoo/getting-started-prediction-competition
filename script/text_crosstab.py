import argparse
import pandas as pd
import sys
from itertools import chain, combinations
from collections import Counter
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess(series):
    wordnet_lemmatizer = WordNetLemmatizer()
    return series.apply(lambda doc: ' '.join([wordnet_lemmatizer.lemmatize(word.lower()) for word in doc.split()]))

def get_words(df):
    corpus_for_words = Counter(
        chain.from_iterable(
            [text.split() for text in df.text.tolist()]
        )
    )
    return corpus_for_words

def get_biterms(df):
    def get_biterm(x):
        return [sorted(biterm) for biterm in combinations(x, 2) if biterm[0] != biterm[1]]

    corpus_for_biterms = Counter(
        chain.from_iterable(
            [['-'.join(biterm) for biterm in get_biterm(text.split())] for text in df.text.tolist()]
        )
    )
    
    return corpus_for_biterms

def get_dataframe_from_neg_pos_corpus(neg, pos):
    words = list(set(list(neg.keys())) & set(list(pos.keys())))
    df = pd.DataFrame(
        {
            'term': words, 
            'neg': [neg[word] for word in words], 
            'pos': [pos[word] for word in words],
            'neg_and_pos': [neg[word]+pos[word] for word in words],
            'latio_pos': [pos[word]/(neg[word]+pos[word]) for word in words]
        }
    )
    return df
    
if __name__ == '__main__':
    
    # read args
    parser = argparse.ArgumentParser(description='extract words and biterms from text.')
    parser.add_argument('-i', help='input filename', default='./data/train.csv')
    args = parser.parse_args()
    
    # set file path
    inputfile = args.i

    # read train.csv
    df = pd.read_csv(inputfile)
    
    # get corpus 
    df['text'] = preprocess(df.copy().text)
    
    df_neg, df_pos = df[df.target == 0], df[df.target == 1]
    df_words = get_dataframe_from_neg_pos_corpus(get_words(df_neg), get_words(df_pos))
    df_biterms = get_dataframe_from_neg_pos_corpus(get_biterms(df_neg), get_biterms(df_pos))
        
    df_words.to_csv('./fact/words.csv', index=None)
    df_biterms.to_csv('./fact/biterms.csv', index=None)
    
    sys.exit()
        