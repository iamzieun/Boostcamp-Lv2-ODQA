import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer


# prepare dataframe

def prepare_dataframe(df):
    """
    주어진 데이터 프레임의 문자열들을 tokenize하고, 각각의 token 길이를 컬럼으로 추가하여 새로운 데이터프레임을 생성합니다.
    Args:
        df (pandas.DataFrame): 원본 데이터 프레임.
    Returns:
        pandas.DataFrame: 토큰 길이 정보가 추가된 데이터 프레임.
    """
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large', use_fast=True)
    tokenize_fn = tokenizer.tokenize

    df['original_context_len'] = [len(tokenize_fn(text)) for text in df['original_context']]
    df['context_len'] = [len(tokenize_fn(text)) for text in df['context']]
    df['original_answers_len'] = [len(tokenize_fn(text)) for text in df['original_answers']]
    df['answers_len'] = [len(tokenize_fn(text)) for text in df['answers']]
    df['question_len'] = [len(tokenize_fn(text)) for text in df['question']]
    
    return df


# quantitative analysis

def retriever_accuracy(df):
    """
    Retriever의 정확도를 계산합니다. 
    원본 context가 새로운 context 내에 포함되는 경우를 정확하게 예측한 것으로 간주합니다.
    Args:
        df (pandas.DataFrame): 데이터 프레임.
    Returns:
        float: Retriever의 정확도 (퍼센트).
    """
    count = 0
    for idx in range(df.shape[0]):
        if df.loc[idx, "original_context"] in df.loc[idx, "context"]:
            count += 1
    return round(count / df.shape[0] * 100, 2)

def reader_accuracy(df):
    """
    Reader의 정확도를 계산합니다.
    원본 답변이 새로운 답변과 동일한 경우를 정확하게 예측한 것으로 간주합니다.
    Args:
        df (pandas.DataFrame): 데이터 프레임.
    Returns:
        float: Reader의 정확도 (퍼센트).
    """
    count = 0
    for idx in range(df.shape[0]):
        if df.loc[idx, "original_answers"] == df.loc[idx, "answers"]:
            count += 1
    return round(count / df.shape[0] * 100, 2)


# qualitative analysis

def retriever_x(df):
    """
    Retriever가 잘못 예측한 경우의 인덱스를 찾아 해당 데이터를 반환합니다.
    원본 문맥이 새로운 문맥 내에 포함되지 않는 경우를 잘못 예측한 것으로 간주합니다.
    Args:
        df (pandas.DataFrame): 데이터 프레임.
    Returns:
        pandas.DataFrame: 잘못 예측한 데이터의 데이터 프레임.
    """
    idx_list = []
    for idx in range(df.shape[0]):
        if df.loc[idx, 'original_context'] not in df.loc[idx, 'context']:
            idx_list.append(idx)
    print(f"전체 {df.shape[0]}개의 데이터 중 {len(idx_list)}개의 데이터에 대한 golden passage를 잘못 예측했습니다.")
    return df.loc[df.index.isin(idx_list)]

def reader_x(df):
    """
    Reader가 잘못 예측한 경우의 인덱스를 찾아 해당 데이터를 반환합니다.
    원본 문맥이 새로운 문맥 내에 포함되지 않는 경우를 잘못 예측한 것으로 간주합니다.
    Args:
        df (pandas.DataFrame): 데이터 프레임.
    Returns:
        pandas.DataFrame: 잘못 예측한 데이터의 데이터 프레임.
    """
    idx_list = []
    for idx in range(df.shape[0]):
        if df.loc[idx, 'original_answers'] not in df.loc[idx, 'answers']:
            idx_list.append(idx)
    print(f"전체 {df.shape[0]}개의 데이터 중 {len(idx_list)}개의 데이터에 대한 정답을 잘못 예측했습니다.")
    return df.loc[df.index.isin(idx_list)]

def retriever_x_reader_x(df):
    """
    Retriever와 Reader가 모두 잘못 예측한 경우의 인덱스를 찾아 해당 데이터를 반환합니다.
    원본 문맥이 새로운 문맥 내에 포함되지 않는 경우를 잘못 예측한 것으로 간주합니다.
    Args:
        df (pandas.DataFrame): 데이터 프레임.
    Returns:
        pandas.DataFrame: 잘못 예측한 데이터의 데이터 프레임.
    """
    idx_list = []
    for idx in range(df.shape[0]):
        if df.loc[idx, 'original_context'] not in df.loc[idx, 'context']:
            if df.loc[idx, 'original_answers'] not in df.loc[idx, 'answers']:
                idx_list.append(idx)
    print(f"Retriever 오답 & Reader 오답: 전체 {df.shape[0]}개의 데이터 중 {len(idx_list)}개")
    return df.loc[df.index.isin(idx_list)]

def retriever_o_reader_x(df):
    """
    Retriever는 맞았으나 Reader가 잘못 예측한 경우의 인덱스를 찾아 해당 데이터를 반환합니다.
    원본 문맥이 새로운 문맥 내에 포함되지 않는 경우를 잘못 예측한 것으로 간주합니다.
    Args:
        df (pandas.DataFrame): 데이터 프레임.
    Returns:
        pandas.DataFrame: 잘못 예측한 데이터의 데이터 프레임.
    """
    idx_list = []
    for idx in range(df.shape[0]):
        if df.loc[idx, 'original_context'] in df.loc[idx, 'context']:
            if df.loc[idx, 'original_answers'] not in df.loc[idx, 'answers']:
                idx_list.append(idx)
    print(f"Retriever 정답 & Reader 오답: 전체 {df.shape[0]}개의 데이터 중 {len(idx_list)}개")
    return df.loc[df.index.isin(idx_list)]

def retriever_x_reader_o(df):
    """
    Reader는 맞았으나 Retriever가 잘못 예측한 경우의 인덱스를 찾아 해당 데이터를 반환합니다.
    원본 문맥이 새로운 문맥 내에 포함되지 않는 경우를 잘못 예측한 것으로 간주합니다.
    Args:
        df (pandas.DataFrame): 데이터 프레임.
    Returns:
        pandas.DataFrame: 잘못 예측한 데이터의 데이터 프레임.
    """
    idx_list = []
    for idx in range(df.shape[0]):
        if df.loc[idx, 'original_context'] not in df.loc[idx, 'context']:
            if df.loc[idx, 'original_answers'] in df.loc[idx, 'answers']:
                idx_list.append(idx)
    print(f"Retriever 오답 & Reader 정답: 전체 {df.shape[0]}개의 데이터 중 {len(idx_list)}개")
    return df.loc[df.index.isin(idx_list)]