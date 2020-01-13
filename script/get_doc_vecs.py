import numpy as np
from tqdm import tqdm_notebook
import transformers as ppb
import torch
import pandas as pd

def get_word_vecs(docs):
    # For DistilBERT:
    model_class = ppb.DistilBertModel
    tokenizer_class = ppb.DistilBertTokenizer
    pretrained_weights = 'distilbert-base-uncased'
    
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    DEVICE = torch.device("cuda")
    model.to(DEVICE)

    def chunks(l, n):
        #Yield successive n-sized chunks from l.
        for i in range(0, len(l), n):
            yield l[i:i + n]

    batch_size = 10

    fin_features = []
    for data in tqdm_notebook(chunks(docs, batch_size)):
        tokenized = []
        for x in data:
            x = " ".join(x.strip().split()[:200])
            tok = tokenizer.encode(x, add_special_tokens=True)
            tokenized.append(tok[:512])
        
        max_len = 512
        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
        attention_mask = np.where(padded != 0, 1, 0)
        input_ids = torch.tensor(padded)
        attention_mask = torch.tensor(attention_mask)
        input_ids = torch.tensor(padded).to(DEVICE)
        attention_mask = torch.tensor(attention_mask).to(DEVICE)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)
      
        features = last_hidden_states[0][:, 0, :].cpu().numpy()
        fin_features.append(features)
        
    return np.vstack(fin_features)

df = pd.read_csv('./data/train.csv')
docs = df.text.tolist()
vecs = get_word_vecs(docs)
print(vecs)