import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


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