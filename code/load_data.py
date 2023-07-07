import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer


def add_data(origin_dataset: DatasetDict, 
             new_dataset_name: str = 'squad_kor_v1', 
             add_valid: bool = False,
             drop_context_duplicate: bool = True):
    """
    원본 DatasetDict 객체에 새로운 데이터를 추가한 DatasetDict 객체를 반환합니다.

    Arguments:
        origin_dataset (DatasetDict): 원본 DatasetDict 객체입니다.
        new_dataset_name (str, optional): 추가로 로드할 Dataset의 이름입니다. 기본값은 'squad_kor_v1'입니다.
        add_valid (bool, optional): 검증 데이터셋도 추가할지 여부를 결정합니다. 기본값은 False입니다.
        drop_context_duplicate (bool, optional): 추가하는 데이터 중 context가 겹치는 데이터를 제거할지 여부를 결정합니다. 기본값은 True입니다.

    Returns:
        full_ds (DatasetDict): 원본 DatasetDict 객체에 새 데이터가 합쳐진, 중복이 제거된 이후 랜덤으로 섞인 DatasetDict 객체입니다.
    """
    # train dataset
    origin_train_df = pd.DataFrame(origin_dataset['train'])
    new_train_df = pd.DataFrame(load_dataset(new_dataset_name)['train'])
    if drop_context_duplicate:
        new_train_df = new_train_df.drop_duplicates(['context'], keep='first')
        
    full_train_df = (pd.concat([origin_train_df, new_train_df])
                     .drop_duplicates(subset=['title', 'context', 'question'])
                     .sample(frac=1, random_state=42)
                     .reset_index(drop=True))
    full_train_ds = Dataset.from_pandas(full_train_df)

    # validation dataset
    if not add_valid:
        full_valid_ds = origin_dataset['validation']
    else:
        origin_valid_df = pd.DataFrame(origin_dataset['validation'])
        new_valid_df = pd.DataFrame(load_dataset(new_dataset_name)['validation'])
        full_valid_df = (pd.concat([origin_valid_df, new_valid_df])
                         .drop_duplicates(subset=['title', 'context', 'question'])
                         .sample(frac=1, random_state=42)
                         .reset_index(drop=True))
        full_valid_ds = Dataset.from_pandas(full_valid_df)


    full_ds = DatasetDict({
        'train': full_train_ds,
        'validation': full_valid_ds
    })

    return full_ds


def sort_train_datasets(datasets: DatasetDict):
    """
    훈련 데이터셋을 answers과 context의 길이에 따라 정렬합니다.

    Args:
        datasets (DatasetDict): 정렬할 데이터셋. 'train' 키로 훈련 데이터셋을 포함해야 합니다.

    Returns:
        full_datasets (DatasetDict): 정렬된 훈련 데이터셋과 원본 검증 데이터셋을 포함한 DatasetDict 객체.
    """
    df = pd.DataFrame(datasets["train"])

    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large', use_fast=True)
    tokenize_fn = tokenizer.tokenize

    df['context_len'] = [len(tokenize_fn(text)) for text in df['context']]
    df['answers_len'] = [len(tokenize_fn(text['text'][0])) for text in df['answers']]

    df = df.sort_values(['answers_len', 'context_len'])
    df.drop(['context_len', 'answers_len', '__index_level_0__'], axis=1, inplace=True)

    sorted_datasets = Dataset.from_pandas(df)

    full_datasets = DatasetDict({
            'train': sorted_datasets,
            'validation': datasets['validation']
        })

    return full_datasets


def make_retrieved_context():
    """
    CSV 파일로부터 훈련 및 검증 데이터셋을 로드하고, 이를 DatasetDict 형태로 반환합니다.

    Returns:
        full_ds (DatasetDict): 'train'과 'validation' 키를 갖는 DatasetDict 객체.
    """
    train_df = pd.read_csv("../data/retrieved_context_dataset/train_3.csv")
    valid_df = pd.read_csv("../data/retrieved_context_dataset/train_3.csv")

    train_df['answers'] = train_df['answers'].apply(str).apply(eval)
    valid_df['answers'] = valid_df['answers'].apply(str).apply(eval)
    
    train_ds = Dataset.from_pandas(train_df)
    valid_ds = Dataset.from_pandas(valid_df)

    full_ds = DatasetDict({
        'train': train_ds,
        'validation': valid_ds
    })

    return full_ds