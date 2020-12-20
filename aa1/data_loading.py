
#basics
import random
import pandas as pd
import torch


from nltk.tokenize import RegexpTokenizer
import os
import xml.etree.ElementTree as ET
import re
from random import choice
from collections import Counter
import matplotlib.pyplot as plt
from venn import venn

pd.options.mode.chained_assignment = None

device = torch.device("cuda:3")

class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!

    def __init__(self, data_dir:str, device=None):
        self._parse_data(data_dir)
        assert list(self.data_df.columns) == [
                                                "sentence_id",
                                                "token_id",
                                                "char_start_id",
                                                "char_end_id",
                                                "split"
                                                ]

        assert list(self.ner_df.columns) == [
                                                "sentence_id",
                                                "ner_id",
                                                "char_start_id",
                                                "char_end_id",
                                                ]
        self.device = device
        

    def get_random_sample(self):
        # DO NOT TOUCH THIS
        # simply picks a random sample from the dataset, labels and formats it.
        # Meant to be used as a naive check to see if the data looks ok
        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
        sample_ners = self.ner_df[self.ner_df["sentence_id"]==sentence_id]
        sample_tokens = self.data_df[self.data_df["sentence_id"]==sentence_id]

        decode_word = lambda x: self.id2word[x]
        sample_tokens["token"] = sample_tokens.loc[:,"token_id"].apply(decode_word)

        sample = ""
        for i,t_row in sample_tokens.iterrows():

            is_ner = False
            for i, l_row in sample_ners.iterrows():
                 if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                    sample += f'{self.id2ner[l_row["ner_id"]].upper()}:{t_row["token"]} '
                    is_ner = True
            
            if not is_ner:
                sample += t_row["token"] + " "

        return sample.rstrip()



class DataLoader(DataLoaderBase):


    def __init__(self, data_dir:str, device=None):
        super().__init__(data_dir=data_dir, device=device)

    
    def get_paths(self, rootdir):
        # fetches a list of absolute paths to all xml files in subdirectories in rootdir
        # helper to parse_xmls
        file_paths = []
        for folder, _, files in os.walk(rootdir):
            for filename in files:
                if filename.endswith('xml'):
                    file_paths.append(os.path.abspath(os.path.join(folder, filename)))
        return file_paths
    
    def string_to_span(self, s):
        # creates a tokenized version and a span version of a string
        # helper to parse_xmls
        punctuation = "-,.?!:;"
        tokenizer = RegexpTokenizer("\s|:|;", gaps=True)
        tokenized = tokenizer.tokenize(s.lower())
        tokenized = [word.strip(punctuation) if word[-1] in punctuation else word for word in tokenized] # removing punctuation if it's the last char in a word
        span = list(tokenizer.span_tokenize(s)) # gets the pythonic span i e (start, stop_but_not_including)
        new_span = []
        for tpl in span:
            new_span.append((tpl[0], (tpl[1]-1))) # to get non-pythonic span i e (start,last_char)
        return new_span, tokenized

    def pad_sentences(self, sentences, max_length):
        # pads sentences to be of length max_length
        # helper to parse_xmls
        data_df_list = []
        for sent in sentences:
            split = sent[0][4]
            pad_len = max_length - len(sent) # how many padding token are needed to make len(sent) == max_length
            pad_rows = pad_len * [(0, 0, 0, 0, split)] # list of padding rows made to fit dataframe ie four 0's are for 'sent_id', 'token_id', 'char_start', 'char_end'
            sent.extend(pad_rows)                      # if sent_id is specified, get_random_sample doesn't work properly
            data_df_list.extend(sent)
        return data_df_list
    
    def parse_xmls(self, fileList):

        data_df_list = [] 
        ner_df_list = []
        all_sentences = [] # will contain a list of tuples where each list represents a sentence and each tuple inside the lists represents a word
        self.ner2id = {
            'other/pad' : 0,
            'drug'      : 1,
            'drug_n'    : 2,
            'group'     : 3, 
            'brand'     : 4
        }
        self.word2id = {}

        for file in fileList:
            tree = ET.parse(file)
            root = tree.getroot()
            for sentence in root:
                one_sentence = []
                sent_id = sentence.attrib['id']
                sent_txt = sentence.attrib['text']
                if sent_txt == "": # to exclude completely empty sentences i e DDI-DrugBank.d228.s4 in Train/DrugBank/Fomepizole_ddi.xml
                    continue
                if 'test' in file.lower():
                    split = 'test'
                else:
                    split = choice(["train", "train", "train", "train", "val"]) # making it a 20% chance that it's val and 80% chance that it's train
                char_ids, tokenized = self.string_to_span(sent_txt)
                for i, word in enumerate(tokenized): # creating data_df_list, one_sentence
                    if word in self.word2id.keys(): # creating word2id, vocab
                        word = self.word2id[word]
                    else:
                        w_id = 1 + len(self.word2id) # zero is pad
                        self.word2id[word] = w_id
                        word = w_id
                    word_tpl = (sent_id, word, int(char_ids[i][0]), int(char_ids[i][1]), split) # one row in data_df 
                    one_sentence.append(word_tpl)
                for entity in sentence: # creating the ner_df_list
                    if entity.tag == 'entity':
                        ent_type = (entity.attrib['type']).lower()
                        ent_type = self.ner2id[ent_type]
                        char_offset = entity.attrib['charOffset']
                        char_span = (re.sub(r"[^0-9]+",' ', char_offset)).split(' ') # substituting everything in char_offset that is not a number with a space
                                                                                     # to be able to split on spaces 
                        if len(char_span) > 2: # for multi-word entities 
                            char_pairs = (list(zip(char_span[::2], char_span[1::2])))
                            for pair in char_pairs:
                                entity_tpl = (sent_id, ent_type, int(pair[0]), int(pair[1]))
                                ner_df_list.append(entity_tpl)
                        else:
                            ent_start_id, ent_end_id = char_span    
                            entity_tpl = (sent_id, ent_type, int(ent_start_id), int(ent_end_id))
                            ner_df_list.append(entity_tpl)
                all_sentences.append(one_sentence)

        
        self.max_sample_length = max([len(x) for x in all_sentences])
        data_df_list = self.pad_sentences(all_sentences, self.max_sample_length)
        
        return data_df_list, ner_df_list
    
    def _parse_data(self, data_dir):
        
        allFiles = self.get_paths(data_dir)
        data_df_list, ner_df_list = self.parse_xmls(allFiles)
        
        # creating dataframes
        self.data_df = pd.DataFrame(data_df_list, columns=['sentence_id', 'token_id', 'char_start_id', 'char_end_id', 'split'])
        self.ner_df = pd.DataFrame(ner_df_list, columns=['sentence_id', 'ner_id', 'char_start_id', 'char_end_id'])
        
        # adding 'pad' to word2id, reversing dictionaries, getting vocab
        self.word2id['<pad>'] = 0 
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2ner = {v:k for k, v in self.ner2id.items()}
        self.vocab = list(self.word2id.keys())


    def get_ners(self, df):
        # constructs label tensors with the help of ner_id
        # returns labellist: a list of all labels, and nested_lists: a list of list where each inner list represents a sentence. Each inner list contains labels for that particular sentence
        #helper to get_y
        
        data_sentence_ids = list(df.sentence_id)
        data_start = list(df.char_start_id)
        data_end = list(df.char_end_id)
        data_token = list(df.token_id)
        data_tpls = [(data_sentence_ids[i], data_token[i], data_start[i], data_end[i]) for i, elem in enumerate(data_sentence_ids)]

        labellist = []
        for tpl in data_tpls: # for every word in data_df, give it a label
            data_sent_id, data_token, data_char_start, data_char_end = tpl
            tpl = (data_sent_id, data_char_start, data_char_end)
            if data_token == 0: # if it's padding
                label = 0 # add 0 label for padding/other
                labellist.append(label)
                continue
            for i, ner in enumerate(self.ner_tpls): # enumerate ensures that we get correct label for row
                ner_sent_id, ner_char_start, ner_char_end = ner
                if tpl == ner: 
                    label = self.ner_id[i]
                    continue
                if data_sent_id == ner_sent_id: # if the two tuples (tpls and ner) aren't exactly the same ie when the ner contains multiple words
                    if (data_char_start >= ner_char_start) and (data_char_end <= ner_char_end): # if the word's start character is greater or equal to the ner start AND word end character is smaller or equal to the ner end, then it counts as part of that ner
                        label = self.ner_id[i]
                    else:
                        label = 0
            labellist.append(label)
        nested_lists = [labellist[x:x+(self.max_sample_length)] for x in range(0, len(labellist), (self.max_sample_length))] # same as labellist but divided into number of sentences
        return labellist, nested_lists
    
        
        
    def get_y(self):
        # returns a tensor containing the ner labels for all samples in each split.
        
        # constructing ner tuples (sentence_id, ner_start, ner_end) and list of ner_ids
        ner_sentence_ids = list(self.ner_df.sentence_id)
        ner_start = list(self.ner_df.char_start_id)
        ner_end = list(self.ner_df.char_end_id)
        self.ner_id = list(self.ner_df.ner_id)
        self.ner_tpls = [(ner_sentence_ids[i], ner_start[i], ner_end[i]) for i, elem in enumerate(ner_sentence_ids)]
        
        # splitting dataframes into train, test, val 
        train_df = self.data_df.loc[self.data_df.split == 'train']
        test_df = self.data_df.loc[self.data_df.split == 'test']
        val_df = self.data_df.loc[self.data_df.split == 'val']
        
        # getting list of labels and list of labels divided into sentences for each split
        self.train_labels, self.train_get_y = self.get_ners(train_df)
        self.test_labels, self.test_get_y = self.get_ners(test_df)
        self.val_labels, self.val_get_y = self.get_ners(val_df)
        
        # putting list of labels divided into sentences on the gpu
        self.train_y = torch.Tensor(self.train_get_y).to(device)
        self.test_y = torch.Tensor(self.test_get_y).to(device)
        self.val_y = torch.Tensor(self.val_get_y).to(device)
        
        print('get_y done')
        return self.train_y, self.val_y, self.test_y
        

    def plot_split_ner_distribution(self):
        # plots a histogram displaying ner label counts for each split
        self.get_y()
        
        # counting label excluding 0 since keeping it hides distribution of other labels
        train_counts = Counter([self.id2ner[l] for l in self.train_labels if l != 0])
        test_counts = Counter([self.id2ner[l] for l in self.test_labels if l != 0])
        val_counts = Counter([self.id2ner[l] for l in self.val_labels if l != 0])
        
        pd.DataFrame([train_counts, test_counts, val_counts], index=['train', 'test', 'val']).plot(kind='bar')
        
        pass
        


    def plot_sample_length_distribution(self):
        #plots a histogram displaying the distribution of sample lengths in number tokens
        
        # removing 0's (padding), converting to series with index 'sentence_id' and values 'token_id' (which second column doesn't matter), group by 'sentence_id' and count, convert counts to list. converting to series is done to enable converting to list of counts
        sentence_length = self.data_df.loc[self.data_df.token_id != 0].set_index('sentence_id')['token_id'].groupby('sentence_id').count().to_list()

        #plotting
        plt.style.use('ggplot') # plot style
        plt.rcParams['figure.figsize'] = [20/2.54, 16/2.54] # making plot bigger
        plt.hist(sentence_length, bins=45, color='#C25A7C')
        plt.xlabel('Length of sentence')
        plt.ylabel('No. of sentences')
        plt.show()
        
        


    def plot_ner_per_sample_distribution(self):        
        # plots a histogram displaying the distribution of number of NERs in sentences
        
        ners_per_sentence = self.ner_df.set_index('sentence_id')['ner_id'].groupby(['sentence_id']).count().to_list() # similar to plot_sample_length_distribution, but not removing 0's (since they don't exist as labels in ner_df) 
        no_ners = (self.data_df['sentence_id'].nunique()) - (self.ner_df['sentence_id'].nunique()) # number of sentences with no ners
        no_ners = [0] * no_ners 
        ners_per_sentence.extend(no_ners) # adding the same number 0's as sentences that don't contain ners
        
        #plotting
        plt.style.use('ggplot')
        plt.rcParams['figure.figsize'] = [20/2.54, 16/2.54]
        plt.hist(ners_per_sentence, bins=45, color='#5ac26c')
        plt.xlabel('No. of ners')
        plt.ylabel('No. of sentences')
        plt.show()
        



    def plot_ner_cooccurence_venndiagram(self):
        # plots a ven-diagram displaying how the ner labels co-occur
        
        df_dict = self.ner_df.groupby('ner_id').apply(lambda x: set(x['sentence_id'])).to_dict() # makes a dictionary of {label1 : set(sentence1, sentence2, ...)}
        for i, ner in self.id2ner.items(): # changing the label id's to label names
            if i in df_dict:
                df_dict[self.id2ner[i]] = df_dict.pop(i)
        
        venn(df_dict)



