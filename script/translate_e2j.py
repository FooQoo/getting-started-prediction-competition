import argparse
import numpy as np
import pandas as pd
import sys
from googletrans import Translator
from json.decoder import JSONDecodeError
from time import sleep

def trans_en_ja(array_key: list, num_text: int, sleep_time: float):
    
    en_delimiter, ja_delimiter = '. ', '。'
    len_arr = len(array_key)
    array_key_ja, array_key_en  = [], []
    
    # preprocess
    array_key = [key.replace('.', ' ') for key in array_key]
    
    for i in range(int(np.ceil(len_arr/num_text))):
        
        translator = Translator()
        # progress 
        print_progress((i+1)/int(np.ceil(len_arr/num_text)))
        
        try: 
            array_key_ja += translator.translate(
                en_delimiter.join(array_key[num_text*i:num_text*(i+1)]), 
                src='en', 
                dest='ja'
            ).text.split(ja_delimiter)
            
            array_key_en += array_key[num_text*i:num_text*(i+1)]
            
        except JSONDecodeError:
            pass
        
        del translator
        
        sleep(sleep_time)
    
    pd.DataFrame({'ja': array_key_ja}).to_csv('./fact/fp_j_wip.csv')
    
    if len(array_key_en) != len(array_key_ja):
        raise Exception('Invalid length of both array.')
    
    print('\nrate of translation: {0:.3f}'.format(len(array_key_en)/len(array_key)*100))
        
    return {w_en: w_ja for w_en, w_ja in zip(array_key_en, array_key_ja)}
 
def to_csv_e2j(filename: str, e2j: dict):
    e2j_csv = pd.DataFrame({'en': list(e2j.keys()), 'ja': list(e2j.values())})
    e2j_csv.to_csv(filename, index=None)

def print_progress(t: float):
    print('\rProgress: {0:3.3f}%'.format(t*100), end='')
    
if __name__ == '__main__':
    
    # read args
    parser = argparse.ArgumentParser(description='translate from Engilish to Japan.')
    parser.add_argument('-c', '--column', help='csv column', required=True)
    parser.add_argument('-i', help='input filename', default='./data/train.csv')
    parser.add_argument('-o', help='output filename', default='./fact/e2j.csv')
    parser.add_argument('-n', '--num', help='number of text each iteration', default=10, type=int)
    parser.add_argument('-s', '--sleep', help='sleep itme each iteration', default=0, type=float)
    args = parser.parse_args()
    
    try:
        # set file path
        inputfile, outputfile = args.i, args.o

        # read train.csv
        df = pd.read_csv(inputfile)

        if args.column not in df.columns:
            raise Exception('Not found "%s" in csv.' % args.column)

        # set paramter
        num_text, column, sleep_time = args.num, args.column, args.sleep

        # translate
        keywords_en = df[df[column].notna()][column].unique().tolist()
        print('length: %s ' % len(keywords_en))
        
        e2j = trans_en_ja(keywords_en, num_text, sleep_time)
        to_csv_e2j(outputfile, e2j)
        
    except Exception as e:
        sys.stderr.write('Error:%s' % e)
        sys.exit(1)
    
    sys.exit()
        