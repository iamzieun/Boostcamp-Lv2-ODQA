import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from transformers import AutoTokenizer


def prepare_dataframe(df: pd.DataFrame):
    """
    주어진 데이터프레임을 전처리고, 토큰화된 question, context, answer의 길이를 새로운 열로 추가합니다.
    Args: 
        df (pd.DataFrame): 전처리할 데이터프레임.
    Returns: 
        df (pd.DataFrame): 전처리가 완료된 데이터프레임.
    """
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large', use_fast=True)
    tokenize_fn = tokenizer.tokenize
    df = df[['id', 'question', 'answers', 'context', 'title', 'document_id']]
    df['answer_start'] = [answers['answer_start'][0] for answers in df['answers']]
    df['answer_text'] = [answers['text'][0] for answers in df['answers']]
    df = df[['id', 'question', 'answers', 'answer_start', 'answer_text', 'context', 'title', 'document_id']]
    df['question_length'] = [len(tokenize_fn(text)) for text in df['question']]
    df['context_length'] = [len(tokenize_fn(text)) for text in df['context']]
    df['answer_length'] = [len(text) for text in df['answer_text']]
    return df


def barplot(series, xlabel, title, xmin=None, xmax=None):
    """
    주어진 시리즈를 막대 그래프로 표시합니다.
    Args:
        series (Series): 그래프로 표시할 Series type의 데이터.
        xlabel (str): x축 라벨.
        title (str): 그래프 제목.
        xmin (int, optional): x축의 최소값.
        xmax (int, optional): x축의 최대값.
    Returns:
        None
    """
    series.value_counts().sort_index().plot(kind='bar', color='royalblue')
    plt.xlabel(xlabel)
    plt.ylabel('count')
    plt.title(title)
    if xmin is not None and xmax is not None:
        plt.xlim(xmin, xmax)
    plt.show()


def barplot_binning(series, xlabel, title, bins):
    """
    주어진 시리즈를 막대 그래프로 표시합니다.
    Args:
        series (Series): 그래프로 표시할 Series type의 데이터.
        xlabel (str): x축 라벨.
        title (str): 그래프 제목.
        bins (int or sequence): 데이터를 나눌 구간. 정수를 전달하면 해당 수만큼 균등한 구간으로 나눕니다.
    Returns:
        None
    """
    binned_series = pd.cut(series, bins=bins, right=False, include_lowest=True)
    binned_series = binned_series.apply(lambda x: x.left).value_counts().sort_index()
    
    binned_series.index = binned_series.index.astype(str)
    binned_series.index = binned_series.index.where(binned_series.index != binned_series.index[-1], binned_series.index[-1]+' 이상')  # Append "이상" to the last label
    
    binned_series.plot(kind='bar', color='royalblue')
    plt.xlabel(xlabel)
    plt.ylabel('count')
    plt.title(title)
    plt.show()


def filter_special_characters(df, column_name):
    """
    데이터프레임의 지정된 열에서 한글과 공백을 제외한 모든 문자를 포함하는 행을 필터링하여 반환합니다.
    Args:
        df (pd.DataFrame): 필터링할 데이터프레임.
        column_name (str): 필터링 대상이 될 열의 이름.
    Returns:
        filtered_df (pd.DataFrame): 한글과 공백을 제외한 모든 문자를 포함하는 행을 필터링하여 추출된 데이터프레임.
    """
    pattern = r'[^\s가-힣]'
    mask = df[column_name].str.contains(pattern, na=False, regex=True)
    filtered_df = df[mask]
    return filtered_df