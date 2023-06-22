import argparse
import json
import logging
import os
import pickle
import time
from contextlib import contextmanager
from pprint import pformat
from typing import List, Optional, Tuple, Union

import numpy as np
import faiss
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from datasets.arrow_dataset import Dataset
from torch.optim import AdamW
from torch.utils.data import (
    DataLoader,
    SequentialSampler,
    TensorDataset,
)
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    get_linear_schedule_with_warmup,
    PretrainedConfig,
    TrainingArguments,
    set_seed,
)
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


def neat_logger(message: str) -> None:
    separator = '-' * 105
    log_message = pformat(message)
    logger.info(f"\n{separator}\n{log_message}\n{separator}")


class BertEncoder(BertPreTrainedModel):
    # TODO: add docstring
    def __init__(self, config: PretrainedConfig = None):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
        ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs[1]
        return pooled_output


class BiEncoderTrainer:
    def __init__(
        self,
        args: TrainingArguments = None,
        train_dataset: torch.utils.data.Dataset = None,
        eval_dataset: torch.utils.data.Dataset = None,
        tokenizer: PreTrainedTokenizerBase = None,
        p_encoder: BertEncoder = None,
        q_encoder: BertEncoder = None,
        pq_encoders_dir: str = None,
        neg_samples: int = 7,
    ) -> None:
        # TODO: add docstring
        self.args = args

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.tokenizer = tokenizer

        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.pq_encoders_dir = pq_encoders_dir

        self.neg_samples = neg_samples

        self.prepare_in_batch_negatives(neg_samples=neg_samples)

    def prepare_in_batch_negatives(
        self,
        train_dataset: torch.utils.data.Dataset = None,
        eval_dataset: torch.utils.data.Dataset = None,
        tokenizer: PreTrainedTokenizerBase = None,
        neg_samples: int = 7,
    ) -> None:
        # TODO: add docstring
        if train_dataset is None:
            train_dataset = self.train_dataset

        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        # 1. Make in-batch negatives
        corpus = np.array(
            list(set([passage for passage in train_dataset["context"]]))
        )
        passage_with_neg_samples = []

        for p in train_dataset["context"]:
            while True:
                neg_ids = np.random.randint(len(corpus), size=neg_samples)

                if not p in corpus[neg_ids]:
                    neg_passages = corpus[neg_ids]
                    passage_with_neg_samples.append(p)
                    passage_with_neg_samples.extend(neg_passages)
                    break

        # 2. Construct (Query, Passage) pairs
        q_seqs = tokenizer(train_dataset["question"],
                           padding="max_length",
                           truncation=True,
                           return_tensors="pt")
        p_seqs = tokenizer(passage_with_neg_samples,
                           padding="max_length",
                           truncation=True,
                           return_tensors="pt")
        neat_logger(f"q_seqs length: {len(q_seqs)}, "
                    f"p_seqs length: {len(p_seqs)}")

        max_len = p_seqs["input_ids"].size(-1)

        p_seqs["input_ids"] = (
            p_seqs["input_ids"].view(-1, neg_samples+1, max_len)
        )
        p_seqs["attention_mask"] = (
            p_seqs["attention_mask"].view(-1, neg_samples+1, max_len)
        )
        p_seqs["token_type_ids"] = (
            p_seqs["token_type_ids"].view(-1, neg_samples+1, max_len)
        )
        neat_logger(f"input_ids, attention_mask, and token_type_ids "
                    f"shapes of p_seqs: {p_seqs['input_ids'].shape}")

        train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.args.per_device_train_batch_size,
        )

        eval_p_seqs = tokenizer(
            eval_dataset["context"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        eval_q_seqs = tokenizer(
            eval_dataset["question"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        eval_dataset = TensorDataset(
            eval_p_seqs["input_ids"],
            eval_p_seqs["attention_mask"],
            eval_p_seqs["token_type_ids"],
            eval_q_seqs["input_ids"],
            eval_q_seqs["attention_mask"],
            eval_q_seqs["token_type_ids"],
        )
        self.eval_dataloader = DataLoader(
            eval_dataset,
            shuffle=False,
            batch_size=self.args.per_device_eval_batch_size,
        )

    def train(
        self,
        args: TrainingArguments = None,
    ) -> None:
        # TODO: add docstring
        if args == None:
            args = self.args

        batch_size = args.per_device_train_batch_size
        self.p_encoder.named_parameters()

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay
            },
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon,
        )
        t_total = (
            len(self.train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total,
        )

        # Start training
        global_steps = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(
            range(int(args.num_train_epochs)),
            desc="Training with evaluation..",
        )

        flag = True
        for idx in train_iterator:
            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    if flag:
                        neat_logger(f"batch length: {len(batch)}")

                    self.p_encoder.train()
                    self.q_encoder.train()

                    targets = torch.zeros(batch_size).long().to(args.device)

                    p_inputs = {
                        "input_ids": (
                            batch[0].view(
                                batch_size * (self.neg_samples + 1), -1
                            ).to(args.device)
                        ),
                        "attention_mask": (
                            batch[1].view(
                                batch_size * (self.neg_samples + 1), -1
                            ).to(args.device)
                        ),
                        "token_type_ids": (
                            batch[2].view(
                                batch_size * (self.neg_samples + 1), -1
                            ).to(args.device)
                        ),
                    }

                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device),
                    }

                    p_outputs = self.p_encoder(**p_inputs)
                    q_outputs = self.q_encoder(**q_inputs)

                    if flag:
                        neat_logger(f"p_outputs shape: {p_outputs.shape}, "
                                    f"q_outputs shape: {q_outputs.shape}")

                    # Calculate similarity score & loss
                    p_outputs = (
                        p_outputs.view(batch_size, self.neg_samples + 1, -1)
                    )
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = (
                        torch.bmm(
                            q_outputs, torch.transpose(p_outputs, 1, 2)
                        ).squeeze()
                    )
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    if flag:
                        neat_logger(f"sim scores shape: {sim_scores.shape}, "
                                    f"targets shape: {targets.shape}")

                    loss = F.nll_loss(sim_scores, targets)
                    if flag:
                        neat_logger(f"loss shape: {loss.shape}")
                        flag = False
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_steps += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

            eval_loss = self.evaluate()
            neat_logger(f"eval_loss: {eval_loss}")

            neat_logger(f"Saving p/q encoder weights at epoch {idx+1}..")
            self.save_model_weights()

    def evaluate(
        self,
        args: TrainingArguments = None,
    ) -> None:
        # TODO: add docstring
        if args == None:
            args = self.args

        eval_loss = 0.0
        nb_eval_steps = 0

        self.p_encoder.eval()
        self.q_encoder.eval()

        with torch.no_grad():
            with tqdm(self.eval_dataloader, unit=" batch") as tepoch:
                for batch in tepoch:
                    batch_size = args.per_device_eval_batch_size

                    targets = torch.zeros(batch_size).long().to(args.device)

                    p_inputs = {
                        "input_ids": (
                            batch[0].view(
                                batch_size * (self.neg_samples + 1), -1
                            ).to(args.device)
                        ),
                        "attention_mask": (
                            batch[1].view(
                                batch_size * (self.neg_samples + 1), -1
                            ).to(args.device)
                        ),
                        "token_type_ids": (
                            batch[2].view(
                                batch_size * (self.neg_samples + 1), -1
                            ).to(args.device)
                        ),
                    }
                    
                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device),
                    }

                    p_outputs = self.p_encoder(**p_inputs)
                    q_outputs = self.q_encoder(**q_inputs)

                    p_outputs = (
                        p_outputs.view(
                            batch_size, self.neg_samples+1, -1
                        )
                    )
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = (
                        torch.bmm(
                            q_outputs, torch.transpose(p_outputs, 1, 2)
                        ).squeeze()
                    )
                    sim_scores = F.log_softmax(sim_scores, dim=1)
                    loss = F.nll_loss(sim_scores, targets)

                    eval_loss += loss.mean().item()
                    nb_eval_steps += 1

            eval_loss /= nb_eval_steps

        return eval_loss

    def save_model_weights(self) -> None:
        # TODO: add docstring
        code_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(code_dir, self.pq_encoders_dir), exist_ok=True)

        p_encoder_path = os.path.join(self.pq_encoders_dir, "p_encoder.pth")
        q_encoder_path = os.path.join(self.pq_encoders_dir, "q_encoder.pth")

        torch.save(self.p_encoder.state_dict(), p_encoder_path)
        torch.save(self.q_encoder.state_dict(), q_encoder_path)


class RetrievalDenseWithFaiss:
    def __init__(
        self,
        p_embds_path: str = None,
        indexer_path: str = None,
        num_clusters: Optional[int] = 48,
    ) -> None:
        # TODO: add docstring
        assert indexer_path is not None, "Index 저장 경로를 지정하세요."

        self.p_embds_path = p_embds_path
        self.num_clusters = num_clusters

        if (
            not os.path.isfile(p_embds_path)
            or not os.path.isfile(indexer_path)
        ):
            self.build_faiss()
            self.save_index(indexer_path=indexer_path)

        self.indexer = self.load_index(indexer_path=indexer_path)

    def build_faiss(self) -> None:
        """
        Note:
            위에서 Faiss를 사용했던 기억을 떠올려보면,
            Indexer를 구성해서 .search() 메소드를 활용했습니다.
            여기서는 Indexer 구성을 해주도록 합시다.
        """
        with open(self.p_embds_path, "rb") as f:
            p_embds = pickle.load(f)
        emb_dim = p_embds.shape[-1]

        quantizer = faiss.IndexFlatL2(emb_dim)
        self.indexer = faiss.IndexIVFScalarQuantizer(
            quantizer,
            quantizer.d,
            self.num_clusters,
            faiss.METRIC_L2,
        )
        self.indexer.train(p_embds)
        self.indexer.add(p_embds)

    def load_index(
        self,
        indexer_path: str = None,
    ) -> faiss.Index:
        assert indexer_path is not None, "Index 저장 경로를 지정하세요."

        neat_logger("Loading Faiss indexer..")
        index = faiss.read_index(indexer_path)
        return index

    def save_index(
        self,
        indexer_path: str = None,
    ) -> None:
        assert indexer_path is not None, "Index 저장 경로를 지정하세요."

        neat_logger("Saving Faiss indexer..")
        faiss.write_index(self.indexer, indexer_path)

    def get_relevant_doc(
        self,
        q_emb: torch.Tensor = None,
        top_k: Optional[int] = 1,
    ) -> Tuple[List, List]:
        """
        Args:
            query (torch.Tensor):
                Dense Representation으로 표현된 query를 받습니다.
                문자열이 아님에 주의합니다.

            top_k (int, default=1):
                상위 몇 개의 유사한 passage를 뽑을 것인지 결정합니다.

        Note:
            받은 query를 이 객체에 저장된 indexer를 활용해서
            유사한 문서를 찾아봅시다.
        """
        q_emb = q_emb.astype(np.float32)
        D, I = self.indexer.search(q_emb, top_k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_docs_for_multiple_queries(
        self,
        q_embs: torch.Tensor = None,
        top_k: Optional[int] = 1,
    ) -> Tuple[List, List]:
        q_embs = np.array(
            [q_emb.astype(np.float32) for q_emb in q_embs]
        )
        D, I = self.indexer.search(q_embs, top_k)

        return D.tolist(), I.tolist()

    def retrieve(
        self,
        query_or_dataset: Union[str, Dataset],
        tokenizer: AutoTokenizer = None,
        q_encoder: BertEncoder = None,
        data_dir: Optional[str] = "../data",
        context_path: Optional[str] = "wikipedia_documents.json",
        top_k: Optional[int] = 1,
        device: Optional[str] = "cuda",
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        assert self.indexer is not None, "build_faiss 메서드를 먼저 실행하세요."

        wiki_path = os.path.join(data_dir, context_path)
        with open(wiki_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)
        contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )

        if isinstance(query_or_dataset, str):
            neat_logger(
                "[Exhaustive search to query using dense passage retrieval "
                f"(DPR)]\n{query_or_dataset}"
            )
            input_query = tokenizer(
                query_or_dataset,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                output_query = q_encoder(**input_query).to("cpu").numpy()

            doc_scores, doc_indices = self.get_relevant_doc(
                output_query,
                top_k=top_k,
            )
            for i in range(top_k):
                neat_logger(
                    f"Top-{i+1} passages with score {doc_scores[i]:4f}\n"
                    f"Doc index: {doc_indices[i]}\n"
                    f"{contexts[doc_indices[i]]}"
                )
            return (
                doc_scores,
                [contexts[doc_indices[i]] for i in range(top_k)],
            )

        elif isinstance(query_or_dataset, Dataset):
            input_queries = query_or_dataset["question"]
            input_queries = tokenizer(
                input_queries,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                q_encoder.eval()
                output_queries = q_encoder(**input_queries).to("cpu").numpy()

            with timer("query exhaustive search using Faiss"):
                doc_scores, doc_indices = (
                    self.get_relevant_docs_for_multiple_queries(
                        output_queries, top_k=top_k
                    )
                )

            total = []
            for idx, row in enumerate(
                tqdm(query_or_dataset, desc="Dense passage retrieval")
            ):
                tmp = {
                    "question": row["question"],
                    "id": row["id"],
                    "context": " ".join(
                        [contexts[pid] for pid in doc_indices[idx]]
                    ),
                }

                if "context" in row.keys() and "anwers" in row.keys():
                    tmp["original_context"] = row["context"]
                    tmp["answers"] = row["answers"]

                neat_logger(f"Given: {row}\n\n Inferred result: {tmp}")

                total.append(tmp)

            return pd.DataFrame(total)


def main():
    SEED = 42
    set_seed(SEED)

    # PyTorch 버전과 XPU 가용 여부를 확인합니다.
    neat_logger(f"PyTorch version: [{torch.__version__}].")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    neat_logger(f"device: [{device}].")

    # 하이퍼파라미터를 설정합니다.
    neat_logger("Setting hyperparameters..")
    num_train_epochs = 10
    batch_size = 4
    top_k = 30
    neg_samples = 7
    num_faiss_clusters = 96

    # code, data, models 등의 디렉토리를 설정합니다.
    neat_logger("Setting the directory paths for code, data, models, etc..")
    code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(code_dir, '..', 'data')
    dataset_dir = os.path.join(data_dir, 'train_dataset')
    pq_encoders_dir = os.path.join(code_dir, "models")
    os.makedirs(pq_encoders_dir, exist_ok=True)

    # p_encoder과 q_encoder의 각 경로를 설정합니다.
    neat_logger("Defining paths for bi-encoders..")
    p_encoder_path = os.path.join(pq_encoders_dir, "p_encoder.pth")
    q_encoder_path = os.path.join(pq_encoders_dir, "q_encoder.pth")

    # 검색에 사용할 wikipedia documents의 경로를 설정합니다.
    neat_logger("Defining wiki docs path..")
    context_path = "wikipedia_documents.json"
    wiki_path = os.path.join(data_dir, context_path)

    # 지문 임베딩(passage embeddings), Faiss 클러스터 인덱스 경로를 지정합니다.
    neat_logger("Defining passage embedding path..")
    p_embds_path = "passage_embeddings.bin"
    indexer_path = f"faiss_clusters_{num_faiss_clusters}.index"
    p_embds_path = os.path.join(data_dir, p_embds_path)
    indexer_path = os.path.join(data_dir, indexer_path)

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name",
                        type=str,
                        default=dataset_dir,
                        help="데이터셋이 담긴 디렉토리 경로입니다.")
    parser.add_argument("--model_name_or_path",
                        default="bert-base-multilingual-cased",
                        type=str,
                        help="Base encoder 명입니다.")
    parser.add_argument("--data_path",
                        default=data_dir,
                        type=str,
                        help="위키 문서가 담긴 디렉토리 경로입니다.")
    parser.add_argument("--context_path",
                        default="wikipedia_documents.json",
                        type=str,
                        help="위키 문서 json 파일명입니다.")
    args = parser.parse_args()

    neat_logger("Loading dataset..")
    total_dataset = load_from_disk(args.dataset_name)

    train_dataset = total_dataset["train"]
    eval_dataset = total_dataset["validation"]

    neat_logger(f"Original dataset size: {len(total_dataset)}")
    neat_logger(f"Original dataset: {total_dataset}")

    neat_logger(f"Training set: {train_dataset}")
    neat_logger(f"Dev set: {eval_dataset}")

    # 일부 학습 데이터셋의 passage ("context")를 로깅합니다.
    for context in train_dataset["context"][:16]:
        neat_logger(context)

    neat_logger("Defining passage/query encoders..")
    encoder_checkpoint = "klue/roberta-large"
    config = AutoConfig.from_pretrained(encoder_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(encoder_checkpoint)
    p_encoder = BertEncoder(config).to(device)
    q_encoder = BertEncoder(config).to(device)

    # 이미 학습된 p_encoder, q_encoder가 없으면 학습합니다.
    if not os.path.isfile(q_encoder_path):
        neat_logger("Defining trainer..")
        training_args = TrainingArguments(
            output_dir="outputs_dpr",
            evaluation_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=1e-5,
            weight_decay=0.01,
            num_train_epochs=num_train_epochs,
            logging_strategy="epoch",
            save_strategy="epoch",
            fp16=True,
        )

        neat_logger("Defining retriever..")
        biencoder = BiEncoderTrainer(
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            p_encoder=p_encoder,
            q_encoder=q_encoder,
            pq_encoders_dir=pq_encoders_dir,
            neg_samples=neg_samples,
        )

        neat_logger("Training retriever..")
        neat_logger(f"Initial evaluation loss: {biencoder.evaluate()}")
        biencoder.train()

        # p_encoder & q_encoder를 저장합니다.
        neat_logger("Saving model weights..")
        biencoder.save_model_weights()

    # 지문 임베딩(passage embeddings)을 저장한 bin 파일이 없으면 새로이 만듭니다.
    if (
        not os.path.isfile(indexer_path)
        or not os.path.isfile(p_embds_path)
    ):
        neat_logger("Building p_embds setup..")
        p_encoder.load_state_dict(torch.load(p_encoder_path))

        # Wikipedia documents 파일 불러오기
        neat_logger("Loading wikipedia documents..")
        with open(wiki_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)
        search_corpus = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )

        neat_logger(
            "Constructing wiki docs tokenizer, dataset, and dataloader.."
        )
        eval_p_seqs = tokenizer(
            search_corpus,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        eval_dataset = TensorDataset(
            eval_p_seqs["input_ids"],
            eval_p_seqs["attention_mask"],
            eval_p_seqs["token_type_ids"],
        )

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=batch_size,
        )

        p_embds = []
        with torch.no_grad():
            epoch_iterator = tqdm(
                eval_dataloader,
                desc="Building Passage Embeddings",
                position=0,
                leave=True,
            )
            p_encoder.eval()

            for batch in epoch_iterator:
                batch = tuple(b.cuda() for b in batch)

                p_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                outputs = p_encoder(**p_inputs).to("cpu").numpy()
                p_embds.extend(outputs)
        p_embds = np.array(p_embds)

        neat_logger("Saving passage embeddings..")
        with open(p_embds_path, "wb") as f:
            pickle.dump(p_embds, f)

        # Faiss index 파일을 만들고 저장합니다.
        neat_logger("Building Faiss retriever..")
        retriever = RetrievalDenseWithFaiss(indexer_path=indexer_path)
        retriever.build_faiss(p_embds=p_embds)

        neat_logger("Saving Faiss retriever..")
        retriever.save_index(indexer_path=indexer_path)
    else:
        retriever = RetrievalDenseWithFaiss(indexer_path=indexer_path)

    q_encoder.load_state_dict(torch.load(q_encoder_path))
    # with open(p_embds_path, "rb") as f:
    #     p_embeddings = pickle.load(f)

    df = retriever.retrieve(
        eval_dataset,
        tokenizer=tokenizer,
        q_encoder=q_encoder,
        top_k=top_k,
    )
    neat_logger(f"DataFrame shape: {df.shape}")

    # 예제
    neat_logger("Loading wikipedia documents..")
    with open(wiki_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)
    search_corpus = list(
        dict.fromkeys([v["text"] for v in wiki.values()])
    )

    neat_logger("Examining a sample case..")
    # query = "금강산의 겨울 이름은?"
    query = "창씨개명령의 시행일을 미루는 것을 수락한 인물은?"

    doc_tuple = retriever.retrieve(
        query,
        tokenizer=tokenizer,
        q_encoder=q_encoder,
        top_k=top_k,
    )
    print(doc_tuple)
    neat_logger(f"Doc scores\n{doc_tuple[0]}")
    neat_logger(f"Docs\n{doc_tuple[1]}")
    neat_logger("ended")


if __name__ == '__main__':
    main()
