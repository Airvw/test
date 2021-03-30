# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from konlpy.tag import Mecab
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn import datasets
from textrank import KeywordSummarizer
import operator
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity

mecab = Mecab()
models = [['Final_FastText', 17], ['Final_FastText', 8]]
TEXT_REVIEW_COLUMN_NAME = 'text'
SENTENCE_IDX_COLUME_NAME = 'rawSentenceIdx'


def init(filename, model):
    sentence_result = keyword_cluster_processing(model, filename)
    #ft_sentence_result = keyword_cluster_processing(models[0], filename)

    save_result(sentence_result, model)
    #save_result(ft_sentence_result, 'FastText')

def WordEmbedding_tokenizer(sents):
  word = []
  for sent in sents:
    words = mecab.pos(sent, join=True)
    words = [w.split('/')[0] for w in words if ('/NNG' in w or '/XR' in w or '/NNR' in w)]
    tt = []
    for w in words:
      if len(w) == 1:
        continue
      tt.append(w)
    word.append(tt)
  return word

def mecab_tokenizer(sent):
    words = mecab.pos(sent, join=True)
    words = [w for w in words if ('/NNG' in w or '/XR' in w or '/NNR' in w)]
    return words

def data_load(filename):
    df = pd.read_csv(filename, encoding ='utf-8', sep='\t')
    df = df.dropna(axis=0)
    #df = df['text']
    rawData = df[TEXT_REVIEW_COLUMN_NAME].values.tolist()
    rawSentenceIdx = df[SENTENCE_IDX_COLUME_NAME].values.tolist()
    tokenData = WordEmbedding_tokenizer(rawData)
    return rawData, tokenData, rawSentenceIdx

def Word2Vec_Embedding(model):
    word_vectors = model.wv
    vocabs = word_vectors.vocab.keys()
    word_vectors_list = [word_vectors[v] for v in vocabs]
    return word_vectors, vocabs, word_vectors_list

def FastText_Embedding(model):
    word_vectors = model.wv
    vocabs = word_vectors.vocab.keys()
    word_vectors_list = [word_vectors[v] for v in vocabs]
    return word_vectors, vocabs, word_vectors_list

def SentenceMapping(rawData, tokenData, word_vectors, _rawSentenceIdx):
    sentenceData = []
    preRawData = []
    rawSentenceIdx = []
    for idx, i in enumerate(tokenData):
        if len(i) == 0:
            continue
        sentenceData.append(i)
        preRawData.append(rawData[idx])
        rawSentenceIdx.append(_rawSentenceIdx[idx])

    sentenceToken = []
    for sentence in sentenceData:
        n = len(sentence)
        np_arr = np.zeros(shape = word_vectors["화장실"].shape)
        for word in sentence:
            try:
                np_arr += word_vectors[word]
            except:
                pass
        if np_arr.all() == np.zeros:
            continue
        sentenceToken.append(np_arr/n)
    return sentenceToken, preRawData, rawSentenceIdx
def cosine_similarity_measure(key_vector, centroid):
    sim_list = []
    for idx, center in enumerate(centroid):
        sim_list.append(cosine_similarity([key_vector], [center]))
    return np.argmax(sim_list)

def k_means_cluster(sentenceToken, preRawData, centroid, rawSentenceIdx, K):
    k_cluster_sentence_result = []

    for sentenceIdx, sentenceVector in enumerate(sentenceToken):
        index = cosine_similarity_measure(sentenceVector, centroid)
        k_cluster_sentence_result.append([rawSentenceIdx[sentenceIdx], index, preRawData[sentenceIdx]])

    return k_cluster_sentence_result

def cluster_result_mapping(data, cluster_estimator, k):
    results = [[] for _ in range(K)]
    for a,b in zip(data, cluster_estimator.labels_):
        results[b].append(a)
    return results

def cluster_keyword_summarizer(results, k):
    keyword_extractor = KeywordSummarizer(
      tokenize = mecab_tokenizer,
      min_count=2,
      window=-1,                     # cooccurrence within a sentence
      min_cooccurrence=2,
      vocab_to_idx=None,             # you can specify vocabulary to build word graph
      df=0.85,                       # PageRank damping factor
      max_iter=30,                   # PageRank maximum iteration
      verbose=False
    )
    keyword_result = []
    for i in range(k):
        _tmp = keyword_extractor.summarize(results[i], topk=20)
        keyword_result.append(_tmp)
    return keyword_result


def cluster_based_word_counting_in_cluster(embedding_model, cluster_keyword_result, cluster_sentence_result):
    item_list = []
    for cluster, sentences in zip(cluster_keyword_result, cluster_sentence_result):
        item_dic = {}
        cnt = 0
        for item in cluster:
            if cnt == 10:
                break
            try:
                _item = item[0].split('/')[0]
                embedding_model.most_similar(_item, topn=10)
                cnt+=1
                for sentence in sentences:
                    item_dic[_item] = item_dic.get(_item, 0) + sentence.count(_item)
            except:
                pass
        item_list.append(item_dic)

    sorted_item_top3 = []
    for item in item_list:
        sort_dict = sorted(item.items(), key=operator.itemgetter(1), reverse=True)
        sorted_item_top3.append(sort_dict[:3])
    return sorted_item_top3

def keyword_cluster_processing(models, filename):
    K = models[1]
    embedding_model, category_name, cluster_centroid = load_model(models[0], K)

    rawData, tokenData, rawSentenceIdx = data_load(filename)

    word_vectors, vocabs, word_vectors_list = FastText_Embedding(embedding_model)

    sentenceToken, preRawData, rawSentenceIdx = SentenceMapping(rawData, tokenData, word_vectors, rawSentenceIdx)
    k_cluster_sentence_result = k_means_cluster(sentenceToken, preRawData, cluster_centroid, rawSentenceIdx, K)

    return k_cluster_sentence_result

def load_model(name, k):
    folder_name = 'data/keyword_save/'

    model_filename = folder_name + name + '_' + str(k) + '.model'
    model = FastText.load(model_filename)
    
    npoutfile = folder_name + name + '_' + str(k) + '_vector.npy'
    centroid = np.load(npoutfile)

    wordoutfile = folder_name + name + '_' + str(k) + '_category.txt'
    category = pd.read_csv(wordoutfile, sep= ',')
    category = category['0']
    category = category.tolist()
    print(category)
    return model, category, centroid


def save_result(result, model):
    folder_name = 'data/results/'
    resultOutFile = folder_name + model[0] + '_' + str(model[1]) + '_results.txt'
    result_df = pd.DataFrame(result)
    result_df.to_csv(resultOutFile, index = False)

for model in models:
  init("data/small_test.txt", model)

