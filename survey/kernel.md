# kernel
## [Basic EDA,Cleaning and GloVe](https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove)
- Basic EDA
- Data Cleaning
- Baseline Model

- score 
    - 0.80764

### Basic EDA
- targetの分布
- ツイート中の文字列長の分布
    - 120~140で最も多い
- ツイート中の単語長の分布
- Average word length in each tweet
- ストップワードの頻度分布
    - target=0 or 1に分けて考える
    - theが頻出している
- punctuationsの分布
    - target=0 or 1に分けて考える
- 単語頻度
- bigramの頻度

### Data Cleaning
- ノイズ除去
    - url
    - HTMLタグ
    - 絵文字
    - punctuation
- スペル修正

### Baseline Model
- Groveによるベクトル化
    - 100次元
- NN
    - emmbeding layerは固定
    - SpatialDropout1D
        - ある次元を丸ごとDropoutする
    - LSTM 
        - dropout
            - 0.2
        - recurrent dropout
            - 0.2
        - dense
            - sigmoid
        - binary
            - cross entropy

## [NLP Getting Started Tutorial](https://www.kaggle.com/philculliton/nlp-getting-started-tutorial)
- NLPのチュートリアル

### 内容
- CountVectorizer
- リッジ回帰


## [NLP with Disaster Tweets - EDA & Full Cleaning](https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-full-cleaning)
### 概要
- なんかする

#### Keyword & location
- 欠損率
    - keyword
        - 0.8%
    - locaiton
        - 33%
- 頻度分析
    - keyword
        - ユニーク数
            - train : 222
            - test : 222
        - 結構聴いている
    - location
        - ユニーク数
            - train : 3342
            - test : 1603
        - 人の手で入力されたため非常に汚く、素性に使うべきでない
        
#### Meta Features
- 災害に関係するツイートはニュースエージェントによるものが多く、単語数も多いと考えられる
- 一方非災害ツイートは、個別のユーザによるため、タイポが多い
- フォーカスした統計量は以下
    - word_count number of words in text
    - unique_word_count number of unique words in text
    - stop_word_count number of stop words in text
    - url_count number of urls in text
    - mean_word_length average character count in words
    - char_count number of characters in text
    - punctuation_count number of punctuations in text
    - hashtag_count number of hashtags (#) in text
    - mention_count number of mentions (@) in text
    
- trainとtestで似たような分布をしている
    - 同じサンプルから得られた考えて良い
- url_count, hashtag_count, mention_countは不足している
- 以下はターゲットに大して分布が異なる
    - word_count
    - unique_word_count
    - stop_word_count
    - mean_word_length
    - char_count
    - punctuation_count
    
#### Target & N-grams
- unigramではpunctuations, stop words, numbersはクリーニングすべき
- 災害に関係する単語は、他の文脈で出にくいと考えられる
- 非災害に関係するツイートに多く含まれる単語は同士が多い
    - unformalな構造を持っており、個々のユーザが使用していることが推測される
- bigrams
    - 災害・非災害で共通するbigramは存在しない
    - 頻度の高いbigramはunigramと比較して災害に関する情報をより多く提供する
    - 句読点は削除したほうがよい
    - 非災害ツイートに多いのは、 redditやyoutubeのものが占める
    - 非災害ツイートのbigramには句読点を含むものが多い

- trigram
    - bigramより有益な情報は少ない

#### Embeddings & Text Cleaning
- pre-trained modelを使う場合、一般的な前処理はよくないかもしれない
    - 情報が欠落するから
    - emmbeddingにできるだけ近い語彙を獲得すべき
- 以下のemmbeddingを使用する
    - GloVe-300d-840B
    - FastText-Crawl-300d-2M
    
- embeddingに含まれる単語とテキストの割合を出す
    - クリーニングなしで50％以上の語彙と80％のテキストをカバーしている
    - GloVeとFastTextのカバレッジは非常に近いが、GloVeのカバレッジはわずかに高くなっている
    - oov : 未定義語
        - oovに含まれる単語にはpunctuationsが前後に付与されるものが多い
        - 特殊な文字は置換するか、消去する
        - タイプミスとスラングは修正される
        - 略文字は戻す
        - ハッシュタグとユーザーネームは展開される 
            - PantherAttack => Panther Attack

#### Mislabeled Samples
- 18種類のツイートは内容が同一にもかかわらず、ラベルが異なる
    - 内容が曖昧なため、発生したと考えられる
   
- これらを揃える

#### Test Set Labels
- コンテストラベルはリーク済み

## [AutoML Getting Started Notebook](https://www.kaggle.com/yufengg/automl-getting-started-notebook)
- gcpでautomlを使うためのkernel

## [Learning BERT for the first time](https://www.kaggle.com/basu369victor/learning-bert-for-the-first-time)
- はじめてbertを使った人のメモ

## [Disaster NLP: Keras BERT using TFHub](https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub)
- BERTを用いた様々なNNを試す
    - No pooling
        - 原論文では、感情分析で\[CLS\]トークンを出力埋め込みとして使う。
        - テンソルの形状が、(batch_size, max_len, hidden_dim) -> (batch_size, hidden_dim)
    - No Dense layer
        - シグモイド
    
    - Fixed learning rate, batch size, epochs, optimizer
        - Optimizer : Adam
            - 2e-5 \- 5e-5
        - epochs : 3
        - batch size : 32
        