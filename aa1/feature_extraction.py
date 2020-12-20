import pandas as pd
import torch
import torch.nn as nn
from nltk.data import load
from nltk import pos_tag
tagdict = load('help/tagsets/upenn_tagset.pickle')
device = torch.device('cuda:3')


def get_tags(sentence, i, pad_id, new_id, last_w_id):
    # gets preceding and succeeding pos tags for word at index i in sentence
    
    if i != 0: 
        if sentence[i-1][0] != '<pad>': # whenever preceeding word is not '<pad>'
            prec_tag = torch.tensor([float(sentence[i-1][1])])
        else:
            prec_tag = pad_id
    else: # if its the 1st word, it wont have a preceding word but gets a dummy id
        prec_tag = new_id # new id to indicate no preceding word
    if i != last_w_id:
        if sentence[i+1][0] != '<pad>': # whenever succeeding word is not '<pad>'
            succ_tag = torch.tensor([float(sentence[i+1][1])]) # take the 
        else: 
            succ_tag = pad_id
    else: # if its the last word, it wont have a succeeding word but gets a dummy id
        succ_tag = new_id
        
    return prec_tag, succ_tag

def get_tensors(df, max_sample_length, id2word, tag2id, new_id, pad_id):
    # constructs the feature tensors
    # helper to extract_features
    
    tokens = list(df['token_id'])
    sentences = [tokens[x:x+(max_sample_length)] for x in range(0, len(tokens),(max_sample_length))] # splitting the entire token list into a nested list where each inner list represents a sentence
    feat_sentences = [] # will be nested list where each inner list represents a sentence; each inner list contains feature tensors for each word
    
    for sentence in sentences:
        sentence = [id2word[i] for i in sentences[0]]
        sentence = [(word, tag2id[tag]) for word, tag in pos_tag(sentence)]
        feat_sentence = [] # will contain features for one sentence
        last_w_id = len(sentence) - 1
        for i, word_tpl in enumerate(sentence): 
            word = word_tpl[0]
            
            # pos tag feature
            tag = torch.tensor([float(word_tpl[1])]) # making id of pos tag into tensor
            if word == '<pad>':
                tag = torch.tensor([float(pad_id)])
            
            # getting 1 or 0 depending on whether the word contains any non-alphabetical characters
            if word.isalpha():
                alpha = torch.tensor([1.])
            else:
                alpha = torch.tensor([0.])
            # word length feature
            w_len = torch.tensor([float(len(word))])
            
            # preceding + succeeding pos tag feature
            prec_tag, succ_tag = get_tags(sentence, i, pad_id, new_id, last_w_id)
            
            features = torch.cat((alpha, w_len, prec_tag, tag, succ_tag))

            feat_sentence.append(features)
        feat_sentences.append(torch.stack(feat_sentence))   
    return torch.stack(feat_sentences)
    
def extract_features(data:pd.DataFrame, max_sample_length:int, id2word:dict):
    
    # gets feature tensor for each word in each split
    # feature tensor consists of word length and whether the word contains only 
    # alphabetical characters (1) or not (0), pos tag id,  and preceding and succeeding pos tag id's
    
    # getting all pos tags, create dict mapping pos tag to numbers
    tags = tagdict.keys()
    nums = list(range(len(tagdict.keys())))
    tag2id = {tag:num for num, tag in enumerate(tags)} # pos tag to ids to convert into tensors
    
    new_id = torch.tensor([float(max(tag2id.values()) + 1)]) # id for when there is no preceding or succeeding word
    pad_id = torch.tensor([float(max(tag2id.values()) + 2)]) # id for when the token is '<pad>'
    
    # splitting the data into train, test, val
    train_df = data.loc[data.split == 'train']
    test_df = data.loc[data.split == 'test']
    val_df = data.loc[data.split == 'val']
    
    # retrieving feature tensors with the help of get_tensors
    train_X = get_tensors(train_df, max_sample_length, id2word, tag2id, new_id, pad_id)
    print("Extracted features for train")
    test_X = get_tensors(test_df, max_sample_length, id2word, tag2id, new_id, pad_id)
    print("Extracted features for test")
    val_X = get_tensors(val_df, max_sample_length, id2word, tag2id, new_id, pad_id)
    print("Extracted features for val")
    
    return train_X.to(device), val_X.to(device), test_X.to(device)