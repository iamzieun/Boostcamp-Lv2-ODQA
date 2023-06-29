import os
import re
import json
import pandas as pd
from tqdm.auto import tqdm

from datasets import Dataset
from elasticsearch import Elasticsearch
from typing import List, Optional, Tuple, Union

def corpus_loader(dataset_path: str) -> List[dict]:
    """
    JSON 파일로부터 wikipedia documents를 불러온다. 추가적인 전처리도 수행한다. 
    
    Args:
        dataset_path: wikipedia documents 파일 경로
    
    Return:
        ditionary의 list, 각각은 전처리된 document를 가진다. 
    """
    with open(dataset_path, "r") as f:
        wiki = json.load(f)

    wiki_texts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    wiki_texts = [preprocess(text) for text in wiki_texts]
    wiki_corpus = [{"document_text": wiki_texts[i]} for i in range(len(wiki_texts))]
    return wiki_corpus

# 삽입할 데이터 전처리
def preprocess(text: str) -> str:
    """
    개행문자, 특수문자 등을 제거하는 전처리 함수
    
    Args:
        text: 전처리할 document
    
    Return:
        전처리된 text
    """
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^A-Za-z0-9가-힣.?!,()~‘’“”"":%&《》〈〉''㈜·\-\'+\s一-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def es_search(es: Elasticsearch, index_name: str, question: str, topk: int) -> dict:
    """
    질문에 맞는 document를 Elasticsearch 인덱스를 통해 검색
    
    Args:
        es -- Elasticsearch object.
        index_name -- 검색할 인덱스 이름 
        question -- 검색 쿼리 
        topk -- return할 top-k개 document 개수 
    
    Return:
        검색 결과를 담은 dictionary
    """
    query = {"query": 
                {"bool": 
                    {"must": 
                        [
                            {"match": 
                                {"document_text": question}
                            }
                        ]
                    }
                }
            }
    res = es.search(index=index_name, body=query, size=topk)
    return res


class ElasticRetrieval:
    def __init__(
        self,
        data_path: Optional[str] = None,  
        context_path: Optional[str] = None, 
        setting_path: Optional[str] = None, 
        index_name: Optional[str] = None, 
    ) -> None:
        """       
        Args:
            data_path: 데이터 경로
            context_path: wikipedia document 경로
            setting_path: elastic search setting 파일 경로
            index_name: 검색에 사용할 인덱스 이름
        """

        # Declare ElasticSearch class
        self.es = Elasticsearch(
            "http://localhost:9200", timeout=30, max_retries=10, retry_on_timeout=True
        )

        # ElasticSearch params
        self.data_path = "./data" if data_path is None else data_path
        self.context_path = "wikipedia_documents.json" if context_path is None else context_path
        self.setting_path = "./retriever/elastic_setting.json" if setting_path is None else setting_path
        self.index_name = index_name

        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)

        with open(self.setting_path, "r") as f:
            setting = json.load(f)
        self.es.indices.create(index=index_name, body=setting)
        self.index_name = index_name
        print("Index creation complete")

        wiki_corpus = corpus_loader(os.path.join(self.data_path, self.context_path))

        print(f" {index_name} 에 데이터를 삽입합니다.")
        print(f" 총 데이터 개수 : {len(wiki_corpus)}")

        for i, text in enumerate(tqdm(wiki_corpus)):
            try:
                self.es.index(index=index_name, id=i, body=text)
            except:
                print(f"Unable to load document {i}.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        쿼리나 데이터셋에 알맞은 document를 retrieve하는 함수
        
        Args:
            query_or_dataset: 검색 대상이 되는 쿼리 혹은 데이터셋
            topk: return할 top-k개 document 개수 
        
        Return:
        If the input is a string, it Return the scores and indices of the top documents.
        If the input is a Dataset, it Return a DataFrame containing the retrieved documents for each query.
        """

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices, docs = self.get_relevant_doc(
                query_or_dataset, k=topk
            )
            print("[query exhaustive search using Elastic Search]\n", query_or_dataset, "\n")

            for i in range(min(topk, len(docs))):

                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(doc_indices[i])
                print(docs[i]["_source"]["document_text"])

            return (doc_scores, [doc_indices[i] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            total = []
            doc_scores, doc_indices, docs = self.get_relevant_doc_bulk(
                query_or_dataset["question"], k=topk
            )

            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval(Elasticsearch): ")
            ):

                retrieved_context = []
                for i in range(min(topk, len(docs[idx]))):
                    retrieved_context.append(docs[idx][i]["_source"]["document_text"])

                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(retrieved_context),
                }

                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]

                total.append(tmp)

        cqas = pd.DataFrame(total)
        return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        쿼리와 가장 관련있는 documment를 찾는 함수 
        
        Args:
            query: 문자열 타입 쿼리 
            k: return할 top-k개 document 개수
        
        Return:
            점수, 인덱스, top document를 담은 tuple
        """
        doc_score = []
        doc_index = []
        res = es_search(self.es, self.index_name, query, k)
        docs = res["hits"]["hits"]

        for hit in docs:
            doc_score.append(hit["_score"])
            doc_index.append(hit["_id"])
            print("Doc ID: %3r  Score: %5.2f" % (hit["_id"], hit["_score"]))

        return doc_score, doc_index, docs

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """
        쿼리의 리스트와 가장 관련있는 documment를 찾는 함수 
        
        Args:
            queries: 문자열 타입 쿼리의 리스트
            k: 각 쿼리마다 return할 top-k개 document 개수
        
        Return:
            각 쿼리에 대한 점수, 인덱스, top document를 담은 tuple
        """

        total_docs = []
        doc_scores = []
        doc_indices = []

        for query in queries:
            doc_score = []
            doc_index = []
            res = es_search(self.es, self.index_name, query, k)
            docs = res["hits"]["hits"]

            for hit in docs:
                doc_score.append(hit["_score"])
                doc_indices.append(hit["_id"])

            doc_scores.append(doc_score)
            doc_indices.append(doc_index)
            total_docs.append(docs)

        return doc_scores, doc_indices, total_docs