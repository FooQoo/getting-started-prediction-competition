import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import BertModel
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import os
import time
import pdb
import glob

class DisasterClassifier(nn.Module):
    def __init__(self, freeze_bert = True, hidden_size=768, num_class=2):
        super(DisasterClassifier, self).__init__()
        #Instantiating BERT model obeject
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
    
        #Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
                
        #Classification Layer
        self.weights = nn.Parameter(torch.rand(13, 1))
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        self.fc = nn.Linear(hidden_size, num_class-1) 

    def forward(self, seq, attn_masks):
        '''
        Inputs:
          -seq: Tensor of shape [B,T] containing token ids of sequences
          -attn_masks: Tensor of shape [B, T] conatining attention masks to be used to avoid contribution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representation
        cont_reps, _ = self.bert_layer(seq, attention_mask = attn_masks)[-2:]

        batch_size = seq.shape[0]
        ht_cls = torch.cat(cont_reps)[:, :1, :].view(13, batch_size, 1, 768)
        #Obtaining the representation of [CLS} head
        atten = torch.sum(ht_cls * self.weights.view(13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])
        
        for i, dropout in enumerate(self.dropouts):
            if i==0:
                h = self.fc(dropout(feature))
            else:
                h += self.fc(dropout(feature))
                
            h = h / len(self.dropouts)
        return h

class TweetDataset(Dataset):
  
  def __init__(self, filename, maxlen):
    
    #Store the contents of the file in a pandas dataframe
    self.df = pd.read_csv(filename)

    #Initialize the BERT tokenizer
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    self.maxlen= maxlen
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
    
        #Selecting the sentence and label at the specified index in the dataframe
        sentence = self.df.loc[index, 'clean']
        label = self.df.loc[index, 'target']

        #Preprocessing the text to be suitable for BERT 
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]']

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)

        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, label

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args: 
          patience (int): How long to wait after last time validation loss improved.
                          Default: 7
          verbose (bool): If True, prints a message for each validation loss improvement.
                          Default: False
          delta (float): Minimun change in the monitored quantity to qualify as an improvement.
                          Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter; {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss descreased ({self.val_loss_min:.6f}) --> {val_loss:.6f}. Saving model ...')
            torch.save(model.state_dict(), 'checkpoint.pt')
            self.val_loss_min = val_loss

def train(net, criterion, opti, train_loader, val_loader, args):
    best_acc = 0
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for ep in range(args.max_eps):
        for it, (seq, attn_masks, labels) in enumerate(train_loader):
            #Clear gradients
            opti.zero_grad()
            #Converting these to cuda tensors
            seq, attn_masks, labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu)

            #Obtaining the logits from the model
            logits = net(seq, attn_masks)

            #Computing loss
            loss = criterion(logits.squeeze(-1), labels.float())

            #Back propagating the gradients
            loss.backward()

            #Optimization step
            opti.step()

        if (it+1) % args.print_every == 0:
            acc = get_accuracy_from_logits(logits, labels)
            print("Iteration {} of epochs {} complete. Loss : {} Accuracy : {}".format(it+1, ep+1, loss.item(), acc))

    val_acc, val_loss = evaluate(net, criterion, val_loader, args)
    print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(ep, val_acc, val_loss))

    early_stopping(val_loss, net)

    if early_stopping.early_stop:
        print("Early stopping")
        break
        
    if val_acc > best_acc:
        print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))
        best_acc = val_acc

def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc

def evaluate(net, criterion, dataloader, args):
    net.eval()
    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for seq, attn_masks, labels in dataloader:
            seq, attn_masks, labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu)
            logits = net(seq, attn_masks)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1
            
    return mean_acc / count, mean_loss / count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type = int, default = 1)
    parser.add_argument('-freeze_bert', action='store_true')
    parser.add_argument('-maxlen', type = int, default= 128)
    parser.add_argument('-batch_size', type = int, default= 32)
    parser.add_argument('-lr', type = float, default = 1e-5)
    parser.add_argument('-print_every', type = int, default= 300)
    parser.add_argument('-max_eps', type = int, default= 100)
    parser.add_argument('-patience', type = int, default= 5)
    
    args = parser.parse_args()
    
    #Creating instances of training and validation set
    train_set = TweetDataset(filename = './fact/train_clean.csv', maxlen = args.maxlen)      
    valid_set = TweetDataset(filename = './fact/test_clean.csv', maxlen = args.maxlen) 

    #Creating intsances of training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size = args.batch_size, num_workers = 5)
    valid_loader = DataLoader(valid_set, batch_size = args.batch_size, num_workers = 5)

    net = DisasterClassifier(freeze_bert = args.freeze_bert)

    criterion = nn.BCEWithLogitsLoss()
    opti = optim.Adam(net.parameters(), lr = args.lr)
    
    net.cuda(args.gpu)
    
    train(net, criterion, opti, train_loader, valid_loader, args)
    
