'''
Code adapted from https://github.com/CRIPAC-DIG/GRACE
Linear evaluation on learned node embeddings
'''

import numpy as np
import functools

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder

from sklearn.metrics import roc_auc_score
import pickle as pk
import argparse
import pandas as pd
import os
import json 

from dataset import SENSITIVE_ATTR_DICT  # predefined sensitive attributes for different datasets
from dataset import DATA_FOLDER, RAW_FOLDER

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size != k:
        raise ValueError('Ranking List length < k')    
    return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    sort_r = sorted(r,reverse = True)
    idcg = dcg_at_k(sort_r, k)
    if not idcg:
        print('.', end=' ')
        return 0.
    return dcg_at_k(r, k) / idcg


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics

        return wrapper

    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


@repeat(3)
def label_classification(embeddings, y, train_mask, test_mask, split='random', ratio=0.1):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    X = normalize(X, norm='l2')

    if split == 'random':
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1 - ratio)
    elif split == 'public':
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = Y[train_mask]
        y_test = Y[test_mask]

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")

    return {
        'F1Mi': micro,
        'F1Ma': macro
    }


# load sensitive attributes for movielens
# dat format: user_idx, gender, age, occupation
def load_user_attributes_movielens(file, M):
    #(gender, age, occupation)
    user_attributes = {}
    gender_dist = {'F':0, 'M':0}
    age_dist = {1:0, 18:0, 25:0, 35:0, 45:0, 50:0, 56:0}
    occupation_dist = {occup:0 for occup in range(21)}

    with open(file, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            eachline = line.strip().split('::')
            user_idx = int(eachline[0])
            gender = eachline[1]
            age = int(eachline[2])
            occupation = int(eachline[3])
            user_attributes[user_idx] = (gender, age, occupation)

    return user_attributes


# load edges for movielens
# dat format: user_idx, item_idx, rating
def load_rating_matrix_movielens(file, M, N):
    over_rating_sparse_mtx = {}
    over_rating_mtx = np.zeros((M,N))
    #load the overall rating matrices of size MxN of training or testing set
    print('loading data ...')
    with open(file, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            eachline = line.strip().split('::')
            user_idx = int(eachline[0])
            item_idx = int(eachline[1])
            rating = int(eachline[2])
            over_rating_mtx[user_idx, item_idx] = rating
            over_rating_sparse_mtx[(user_idx, item_idx)] = rating
    
    return over_rating_sparse_mtx, over_rating_mtx



def eval_unbiasedness_movielens(data_name, embeddings=None):
    
    M = 6040 + 1
    N = 3952 + 1
        
    rating_sparse_mtx, rating_mtx = load_rating_matrix_movielens('{}/ml-1m/ratings.dat'.format(RAW_FOLDER), M, N)
    user_attributes = load_user_attributes_movielens('{}/ml-1m/users.dat'.format(RAW_FOLDER), M)

    genders = np.array([int(user_attributes[i][0]=='M') for i in range(1, M)])
    ages = np.array([int(user_attributes[i][1]) for i in range(1, M)])
    occupations = np.array([int(user_attributes[i][2]) for i in range(1, M)])

    attribute_labels = {'gender': genders, 'age': ages, 'occupation': occupations}

    rating_mtx = rating_mtx[1:]
    rating_mtx = rating_mtx[:,1:]
    
    # if embed_file:
    if embeddings != None:
        unbiased_embedding = embeddings
        X = unbiased_embedding[:M-1]  # users
        Y = unbiased_embedding[M-1:]  # items
    else:
        X, Y = np.random.rand(*(M-1,16)), np.random.rand(*(N-1,16))
    
    results = {
        'unbiasedness': {
            'gender': 0.0, 
            'age': 0.0, 
            'region': 0.0
        },
        # 'fairness-DP':{
        #     'gender': 0.0, 
        #     'age': 0.0, 
        #     'region': 0.0
        # },
        # 'fairness-EO':{
        #     'gender': 0.0, 
        #     'age': 0.0, 
        #     'region': 0.0
        # },
        'utility': 0.0
    }

    # eval micro-f1 for attribute prediction (unbiasedness)
    print('Unbiasedness evaluation (predicting attribute)')
    for evaluate_attr in ['gender', 'age', 'occupation']:
        
        lgreg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000).fit(
            X[:5000], attribute_labels[evaluate_attr][:5000])
        pred = lgreg.predict(X[5000:])

        # rating_lgreg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000).fit(
        #     rating_mtx[:5000], attribute_labels[evaluate_attr][:5000])
        # rating_pred = rating_lgreg.predict(rating_mtx[5000:])
        
        score = f1_score(attribute_labels[evaluate_attr][5000:], pred, average='micro')
        
        print(f'-- micro-f1 when predicting {evaluate_attr}: {score}')
        results['unbiasedness'][evaluate_attr] = score
        
        # score = f1_score(attribute_labels[evaluate_attr][5000:], rating_pred, average='micro')
        # print('-- raw rating micro-f1: ', score)


    #evaluate NDCG
    k = 10
    accum_ndcg = 0
    
    print('Utility evaluation (link prediction)')
    for user_id in range(1, M):
        user = user_id - 1
        user_ratings = rating_mtx[user] # (rating_mtx[user] > 0).astype(int)
        
        pred_ratings = np.dot(X[user], Y.T)
        rank_pred_keys = np.argsort(pred_ratings)[::-1]
        ranked_user_ratings = user_ratings[rank_pred_keys]
        ndcg = ndcg_at_k(ranked_user_ratings, k)
        accum_ndcg += ndcg
        
    score = accum_ndcg/M
    print(f'-- ndcg of link prediction: {score}')
    results['utility'] = score

    return results