#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import sklearn as sk
import nltk
import pickle
import joblib
import os


# In[4]:


os.chdir('/Users/sk/Desktop/kp20k')


# In[9]:


train_data = pd.read_json('kp20k_train.src', lines = True)


# In[3]:


nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_english = list(stopwords.words('english'))


# In[4]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import re


# In[5]:


#Removing stop words and conerting to tf-idf matrix form
pipeline = pipeline = Pipeline([
    ('vect', CountVectorizer(max_df=0.7, stop_words= stopwords_english, ngram_range = (1,2), max_features= 10000)),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True))])

pipeline.fit_transform(train_data['src'])


# In[17]:


#save pipeline
joblib.dump(pipeline, 'model.pkl')


# In[5]:


#load pipeline
pipeline = joblib.load('model.pkl')


# In[6]:


#Sorting based on tf_idf values in the vector and taking top 10 features
def sort_matrix(matrix):
    col_data = zip(matrix.col, matrix.data)
    return sorted(col_data, key= lambda x: (x[1], x[0]), reverse=True)

def take_topn(features, col_data, n):
    top_col_data = col_data[:n]
    result = {}
    for col, tfidf in top_col_data:
        feature = features[col]
        result[feature] = tfidf
    return result


# In[7]:


features = pipeline['vect'].get_feature_names()


# In[22]:


#checking sort and top features function on a sample
sample_vector = pipeline.transform([train_data['src'][50]])
vec = sort_matrix(sample_vector.tocoo())
result = take_topn(features, vec, n = 10)
print(result)


# In[11]:


vectors = pipeline.transform(train_data['src'].tolist())


# In[12]:


# Implementing sort and top features function on train data
preds = []
for i in range(vectors.shape[0]):
    vec = vectors[i]
    vec = sort_matrix(vec.tocoo())
    preds.append(take_topn(features, vec, 10))


# In[8]:


#Reading test dataset
test_data = pd.read_json('kp20k_test.src', lines = True)
test_vectors = pipeline.transform(test_data['src'].tolist())


# train_data['predictions'] = preds
# train_data.to_csv('result.csv')


# In[9]:


# Implementing sort and top features function on test data
test_preds = []
for i in range(test_vectors.shape[0]):
    vec = test_vectors[i]
    vec = sort_matrix(vec.tocoo())
    test_preds.append(list(take_topn(features, vec, 10).keys()))


# In[10]:


print(len(test_preds))
#checking a sample prediction
print(test_preds[0])


# In[11]:


#Reading actual train and test targets
train_tgts = pd.read_json('kp20k_train.tgt', lines = True)
test_tgts = pd.read_json('kp20k_test.tgt', lines = True)
#checking sample's actual prediction
print(test_tgts['tgt'][0])


# In[12]:


#Functions get_match_result and run_metrics are reused from OpenNMT package
""" Title: OpenNMT-kpg-release source code
Author: Rui Meng,Eric Yuan,Tong Wang,Khushboo Thaker
Year: 2020
Availability: https://github.com/memray/OpenNMT-kpg-release/blob/6725b530d52b756db7b60c69b261f9a8c372ce88/kp_evaluate.py """
#Metrics generation precision, recall, f1
import numpy as np
from nltk.stem.porter import *

def stem_word_list(word_list):
    return [stemmer.stem(w.strip()) for w in word_list]
stemmer = PorterStemmer()

def get_match_result(true_seqs, pred_seqs, do_stem=True, type='exact'):
    '''
    If type='exact', returns a list of booleans indicating if a pred has a matching tgt
    If type='partial', returns a 2D matrix, each value v_ij is a float in range of [0,1]
        indicating the (jaccard) similarity between pred_i and tgt_j
    :param true_seqs:
    :param pred_seqs:
    :param do_stem:
    :param topn:
    :param type: 'exact' or 'partial'
    :return:
    '''
    # do processing to baseline predictions
    if type == "exact":
        match_score = np.zeros(shape=(len(pred_seqs)), dtype='float32')
    else:
        match_score = np.zeros(shape=(len(pred_seqs), len(true_seqs)), dtype='float32')

    target_number = len(true_seqs)
    predicted_number = len(pred_seqs)

    metric_dict = {'target_number': target_number, 'prediction_number': predicted_number, 'correct_number': match_score}

    # convert target index into string
    if do_stem:
        true_seqs = [stem_word_list(seq) for seq in true_seqs]
        pred_seqs = [stem_word_list(seq) for seq in pred_seqs]

    for pred_id, pred_seq in enumerate(pred_seqs):
        if type == 'exact':
            match_score[pred_id] = 0
            for true_id, true_seq in enumerate(true_seqs):
                match = True
                if len(pred_seq) != len(true_seq):
                    continue
                for pred_w, true_w in zip(pred_seq, true_seq):
                    # if one two words are not same, match fails
                    if pred_w != true_w:
                        match = False
                        break
                # if every word in pred_seq matches one true_seq exactly, match succeeds
                if match:
                    match_score[pred_id] = 1
                    break
        elif type == 'ngram':
            # use jaccard coefficient as the similarity of partial match (1+2 grams)
            pred_seq_set = set(pred_seq)
            pred_seq_set.update(set([pred_seq[i]+'_'+pred_seq[i+1] for i in range(len(pred_seq)-1)]))
            for true_id, true_seq in enumerate(true_seqs):
                true_seq_set = set(true_seq)
                true_seq_set.update(set([true_seq[i]+'_'+true_seq[i+1] for i in range(len(true_seq)-1)]))
                if float(len(set.union(*[set(true_seq_set), set(pred_seq_set)]))) > 0:
                    similarity = len(set.intersection(*[set(true_seq_set), set(pred_seq_set)]))                               / float(len(set.union(*[set(true_seq_set), set(pred_seq_set)])))
                else:
                    similarity = 0.0
                match_score[pred_id, true_id] = similarity

    return match_score

import scipy
def run_metrics(match_list, pred_list, tgt_list, score_names, topk_range, type='exact'):
    """
    Return a dict of scores containing len(score_names) * len(topk_range) items
    score_names and topk_range actually only define the names of each score in score_dict.
    :param match_list:
    :param pred_list:
    :param tgt_list:
    :param score_names:
    :param topk_range:
    :return:
    """
    score_dict = {}
    if len(tgt_list) == 0:
        for topk in topk_range:
            for score_name in score_names:
                score_dict['{}@{}'.format(score_name, topk)] = 0.0
        return score_dict

    assert len(match_list) == len(pred_list)
    for topk in topk_range:
        if topk == 'k':
            cutoff = len(tgt_list)
        elif topk == 'M':
            cutoff = len(pred_list)
        else:
            cutoff = topk

        if len(pred_list) > cutoff:
            pred_list_k = np.asarray(pred_list[:cutoff])
            match_list_k = match_list[:cutoff]
        else:
            pred_list_k = np.asarray(pred_list)
            match_list_k = match_list

        if type == 'partial':
            cost_matrix = np.asarray(match_list_k, dtype=float)
            if len(match_list_k) > 0:
                # convert to a negative matrix because linear_sum_assignment() looks for minimal assignment
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(-cost_matrix)
                match_list_k = cost_matrix[row_ind, col_ind]
                overall_cost = cost_matrix[row_ind, col_ind].sum()
            '''
            print("\n%d" % topk)
            print(row_ind, col_ind)
            print("Pred" + str(np.asarray(pred_list)[row_ind].tolist()))
            print("Target" + str(tgt_list))
            print("Maximum Score: %f" % overall_cost)

            print("Pred list")
            for p_id, (pred, cost) in enumerate(zip(pred_list, cost_matrix)):
                print("\t%d \t %s - %s" % (p_id, pred, str(cost)))
            '''

        # Micro-Averaged Method
        correct_num = int(sum(match_list_k))
        # Precision, Recall and F-score, with flexible cutoff (if number of pred is smaller)
        micro_p = float(sum(match_list_k)) / float(len(pred_list_k)) if len(pred_list_k) > 0 else 0.0
        micro_r = float(sum(match_list_k)) / float(len(tgt_list)) if len(tgt_list) > 0 else 0.0

        if micro_p + micro_r > 0:
            micro_f1 = float(2 * (micro_p * micro_r)) / (micro_p + micro_r)
        else:
            micro_f1 = 0.0
        # F-score, with a hard cutoff on precision, offset the favor towards fewer preds
        micro_p_hard = float(sum(match_list_k)) / cutoff if len(pred_list_k) > 0 else 0.0
        if micro_p_hard + micro_r > 0:
            micro_f1_hard = float(2 * (micro_p_hard * micro_r)) / (micro_p_hard + micro_r)
        else:
            micro_f1_hard = 0.0

        for score_name, v in zip(score_names, [correct_num, micro_p, micro_r, micro_f1, micro_p_hard, micro_f1_hard]):
            score_dict['{}@{}'.format(score_name, topk)] = v

    return score_dict


# In[15]:


# scores on test when considering exact match
score_names  = ['correct', 'precision', 'recall', 'f_score', 'precision_hard', 'f_score_hard']
precision = []
recall = []
f1 = []
for test, tgt in zip(test_preds, test_tgts['tgt'].tolist()):
    match_scores = get_match_result( tgt, test, type='exact')
    score_dict = run_metrics(match_scores, test, tgt, score_names, [5], type='exact')
    precision.append(score_dict['precision@5']*100)
    recall.append(score_dict['recall@5']*100)
    f1.append(score_dict['f_score@5']*100)


# In[16]:


print(round(np.average(recall),2), round(np.average(precision),2), round(np.average(f1),0))


# In[17]:


# scores on test when considering partial match
score_names  = ['correct', 'precision', 'recall', 'f_score', 'precision_hard', 'f_score_hard']
precision = []
recall = []
f1 = []
for test, tgt in zip(test_preds, test_tgts['tgt'].tolist()):
    match_scores = get_match_result( tgt, test, type='ngram')
    score_dict = run_metrics(match_scores, test, tgt, score_names, [5], type='partial')
    precision.append(score_dict['precision@5']*100)
    recall.append(score_dict['recall@5']*100)
    f1.append(score_dict['f_score@5']*100)


# In[19]:


print(round(np.average(recall),2), round(np.average(precision),2), round(np.average(f1),0))


# In[ ]:




