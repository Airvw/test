# -*- coding: utf-8 -*-

import re
from pykospacing import spacing
from soynlp.normalizer import repeat_normalize
from kss import kss
import pandas as pd
import numpy as np

######### 임시 #########
# init 함수 호출을 통해 사용 #
# init 인자 
# 1. fileName: 원본 raw data 파일 위치+이름
# 2. saveName : 전처리 작업을 진행 후의 파일 위치 + 이름
# 3. labelList : 원본 raw data 파일의 columns 이름들 list, 
#                인자 없을 시 BASE_LABEL_LIST 대로 감
# 4. nes_labeList : 작업 결과로 받고자 하는 columns의 원본 이름
#                    인자 없을 시 NECESSARY_LABEL_LIST 대로 감

# 데이터 포멧 통일 작업 #
# 추후 전처리 프로세스 통일을 위해 #
# 컬럼 1개(text) #

BASE_LABEL_LIST = ['HotelName','HotelAddress','HotelRating','Date','ReviewRating','ReviewTitle','ReviewText']
NECESSARY_LABEL_LIST = ['ReviewText']
TEXT_REVIEW_COLUMN_NAME = 'text'
SENTENCE_IDX_COLUME_NAME = 'rawSentenceIdx'

def init(fileName, saveName = 'hotel.txt', labelList = None, nes_labelList = None):
    global BASE_LABEL_LIST
    global NECESSARY_LABEL_LIST
    
    if labelList:
        BASE_LABEL_LIST = labelList
    if nes_labelList:
        NECESSARY_LABEL_LIST = nes_labelList
    data = data_format_setting(fileName)
    data = preprocessingPipeLine(data)
    data.to_csv(saveName, sep = '\t', index = False)

def data_format_setting(fileName):
    data = data_format_setting_hotel(fileName)
    return data
    
def remove_columns(data):
    remove_columns_list = list(set(BASE_LABEL_LIST) -  set(NECESSARY_LABEL_LIST))
    data.drop(columns = remove_columns_list, inplace=True)
    return data

def data_format_setting_hotel(dataFileName):
    data = pd.read_csv(dataFileName)
    data = remove_columns(data)
    data.columns = [TEXT_REVIEW_COLUMN_NAME]
    return data

def regex_spacing_normalization(data):
    del_filter1 = re.compile(r'[!?,.ㅋㅎㅜㅠ가-힣0-9]+')
    data[SENTENCE_IDX_COLUME_NAME] = 0
    df = pd.DataFrame(columns = [TEXT_REVIEW_COLUMN_NAME, SENTENCE_IDX_COLUME_NAME])
    for idx, item in enumerate(data[TEXT_REVIEW_COLUMN_NAME]):
        tmp = str(item)
        if tmp == 'nan':
            continue
        tmp = ' '.join(del_filter1.findall(item))
        tmp = spacing(tmp)
        tmp = repeat_normalize(tmp, num_repeats=2)

        df = df.append({TEXT_REVIEW_COLUMN_NAME:tmp, SENTENCE_IDX_COLUME_NAME:idx}, ignore_index=True)
    return df

def regex_specialChar(data):
    del_filter2 = re.compile(r'[ㅋㅎㅜㅠ가-힣0-9]+')
    for idx, item in enumerate(data[TEXT_REVIEW_COLUMN_NAME]):
        tmp = ' '.join(del_filter2.findall(item))
        data.at[idx, TEXT_REVIEW_COLUMN_NAME] = tmp
    return data
        
def split_sentence(data):
    sentenceData = []
    sentenceIdx = []
    for sentence, idx in zip(data[TEXT_REVIEW_COLUMN_NAME], data[SENTENCE_IDX_COLUME_NAME]):
        for s in kss.split_sentences(sentence):
            sentenceData.append(s)
            sentenceIdx.append(idx)
    return sentenceData, sentenceIdx

def preprocessingPipeLine(data):
    data = regex_spacing_normalization(data)
    data = data.dropna(axis=0)
    
    split_text, split_idx = split_sentence(data)
    data = pd.DataFrame({TEXT_REVIEW_COLUMN_NAME: split_text, SENTENCE_IDX_COLUME_NAME: split_idx})
    
    data = regex_specialChar(data)
    
    data = data.dropna(axis=0)
    data.drop_duplicates(subset=[TEXT_REVIEW_COLUMN_NAME], inplace=True)
    
    
    data["len_text"] = [len(t.split()) for t in data[TEXT_REVIEW_COLUMN_NAME]]
    data = data[data["len_text"] > 1]
    data.drop(columns = ["len_text"], inplace=True)
    data = data.astype({SENTENCE_IDX_COLUME_NAME:'int'})
    return data

#init('data/raw/small_test.csv','data/small_test.txt')
#init('data/key_raw/big_test.csv','data/key_pre/big_test.txt')
#init('data/key_raw/train.csv','data/key_pre/train.txt')

