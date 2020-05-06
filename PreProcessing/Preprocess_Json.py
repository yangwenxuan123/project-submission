#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import re
import os
import csv


# In[2]:


# Preprocessing functions
def replace_digit(input_txt):
    digit_replace = "<digit>"
    # replace big numbers with <digit>
    tokens = [x if not re.match('^\d{%d,}$' % 2, x) else digit_replace for x in input_txt]
    return tokens

def tokenize(input_txt):
    input_txt = re.sub(r'[\r\n\t]', ' ', input_txt)
    # pad spaces to the left and right of special punctuations
    input_txt = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', input_txt)
    # tokenize by non-letters (new-added + # & *, but don't pad spaces, to make them as one whole word)
    tokens = list(filter(lambda x: len(x) > 0, re.split(r'[^a-zA-Z0-9_<>,#&\+\*\(\)\.\']', input_txt)))
    return tokens

def filter_keywords(input_txt,token_str,min_len,max_len):
    filtered_tokens = []
    for token, keyphrase in zip(input_txt, token_str):
        
        keywords = tokenize(keyphrase)
        
        # Removing keyphrase of longer sequences
        if (len(keywords) < min_len) or (len(key_words) > max_len):
            continue
        # Ignore keyphrases with strange punctuations
        punc = re.findall(r'[,_\"<>\(\){}\[\]\?~`!@$%\^=]', keyphrase)
        if len(punc) > 0:
            continue

        #Ignore keyphrases with more than 5 repeated words
        remove = False
        if len(keywords) > 5:
            keyset = set(keywords)
            if len(keyset) * 2 < len(keywords):
                remove = True
        if remove:
            continue
        
        #Ignore keyphrases with strange format such as primary 75v05;secondary 76m10;65n30
        if (len(keywords) > 0 and re.match(r'\d\d[a-zA-Z\-]\d\d', keywords[0].strip()))                 or (len(keywords) > 1 and re.match(r'\d\d\w\d\d', key_words[1].strip())):
            continue
        filtered_tokens.append(token)
    return filtered_tokens


# In[3]:


min_tgt = 2
max_tgt = 10
file = 'train.json'
#file = 'test.json'
#file = 'valid.json'
batch = 1000
to_lower = True
item_dict = []

for num, line in enumerate(open(file, 'r')):
    if (num + 1) % batch == 0:
        print("line %d" % num)
        
    json_dict = json.loads(line)
    if 'id' in json_dict:
        id = json_dict['id']
    else:
        id = str(num)
    keywords = json_dict['keywords']  
    if isinstance(keywords, str):
        keywords = keywords.split(';')
        json_dict['keywords'] = keywords
        
    
    # remove abbreviations/acronyms in parentheses in keyphrases
    keywords = [re.sub(r'\(.*?\)|\[.*?\]|\{.*?\}', '', x) for x in keywords]
    title = json_dict['title']
    abstract = json_dict['abstract']
    if to_lower:
        title = title.lower()
        abstract = abstract.lower()
        keywords = [x.lower() for x in keywords]
        
    title_token = title.split(' ')
    abstract_token = abstract.split(' ')
    keywords_token = [x.split(' ') for x in keywords]
    
    #Ignore tokens with less than 2 keyphrases or more than 10 key phrases
    if len(keywords_token) < min_tgt or len(keywords_token) > max_tgt:
        continue
        
    source_token = title_token+["."]+abstract_token
    target_token = keywords_token
    target_token = filter_keywords(target_token, keywords, 8, 1)
    #target_token = replace_digit(target_token)
    if len(target_token) == 0:
        continue
    tmp = {}
    tmp.update({
                'id': id,
                'src': ' '.join(source_token),
                'tgt': [' '.join(target) for target in target_token],
            })
    item_dict.append(tmp)
    


# In[10]:


len(item_dict)


# In[30]:


keys = item_dict[0].keys()

file_name = 'cleaned_train.csv'
file_name = 'cleaned_test.csv'
file_name = 'cleaned_valid.csv'

with open(file_name, 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(item_dict)


# In[ ]:




