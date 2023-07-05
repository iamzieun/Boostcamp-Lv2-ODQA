# Open-Domain Question Answering
 주어진 지문을 이해하고, 주어진 질의의 답변을 추론하는 태스크

## 일정 Schedule
프로젝트 전체 기간(2주): 6월 7일 (수) 10:00 ~ 6월 22일 (목) 19:00


## 대회 플랫폼 Platform
[AI Stages](https://stages.ai/)

## 팀 Team
**훈제연어들**
|문지혜|박경택|박지은|송인서|윤지환|
|:---:|:---:|:---:|:---:|:---:|
|<img src="https://avatars.githubusercontent.com/u/85336141?v=4" width="120" height="120">|<img src="https://avatars.githubusercontent.com/u/97149910?v=4" width="120" height="120">|<img src="https://avatars.githubusercontent.com/u/97666193?v=4" width="120" height="120">|<img src="https://avatars.githubusercontent.com/u/41552919?v=4" width="120" height="120">|<img src="https://avatars.githubusercontent.com/u/37128004?v=4" width="120" height="120">|
|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:munjh1121@gmail.com)](mailto:afterthougt@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:afterthougt@gmail.com)](mailto:afterthougt@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:imhappyhill@gmail.com)](mailto:imhappyhill@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:songinseo0910@gmail.com)](mailto:songinseo0910@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:yjh091500@naver.com)](mailto:yjh091500@naver.com)|
|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/jihye-moon)](https://github.com/jihye-moon)|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/afterthougt)](https://github.com/afterthougt)|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/iamzieun)](https://github.com/iamzieun)|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/fortunetiger)](https://github.com/fortunetiger)|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/ohilikeit)](https://github.com/ohilikeit)|


## 랩업리포트 Wrap-up Report
```code/assets/MRC_NLP_팀 리포트(12조).pdf```

## 저장소 구조 Repository Structure
```
level2_nlp_mrc-nlp-12/
│
├── code/
│   ├── assets/
│   │
│   ├── eda/
│   │   ├── eda.ipynb
│   │   └── post_eda.ipynb
│   │
│   ├── install/
│   │   ├── elastic_install.sh
│   │   └── install_requirements.sh
│   │
│   ├── retriever/                              # Retiever 실험 코드 모음
│   │   ├── elastic_setting.json                # retrieval_elastic.py를 위한 설정 파일
│   │   ├── retrieval_bm25.py                   # BM25 실험
│   │   ├── retrieval_elastic.py                # elastic search 적용 코드
│   │   ├── retrieval_faiss.py                  # FAISS 적용 코드
│   │   └── retrieval_tfidf.py                  # TFIDF 실험 코드
│   │
│   ├── trainer/
│   │   └── trainer_qa.py
│   │
│   ├── utils/
│   │   ├── evalutaion.py
│   │   └── utils_qa.py
│   │
│   ├── arguments.py
│   ├── inference.py                            # ODQA 모델 평가 또는 제출 파일 (predictions.json) 생성
│   ├── inference.sh                            # inference.py를 실행하기 위한 스크립트
│   ├── load_data.py                            # 데이터셋을 정의하고 DatasetDict를 반환하는 스크립트
│   ├── run.sh                                  # train.py를 실행하기 위한 스크립트
│   ├── run_mrc.py
│   ├── train.py								# MRC, Retrieval 모델 학습 및 평가
│   └── README.md								# ODQA 모델 평가 또는 제출 파일 (predictions.json) 생성
│
├── data/
│   ├── retrieved_context_dataset/              # retrieval에 사용되는 corpus
│   │   ├── train_3.csv
│   │   └── valid_3.csv
│   │
│   ├── test_dataset/                           # MRC 모델 학습에 사용되는 데이터
│   │   ├── validation/
│   │   └── dataset_dict.json
│   │
│   └── train_dataset/                          # MRC 모델 학습에 사용되는 데이터
│       ├── trian/
│       ├── validation/
│       └── dataset_dict.json
│
└── README.md
```

## 사용법 Usage
### train
```bash
$ ./code/run.sh
```

### inference
```bash
$ ./code/inference.sh
```


## 평가 방법 Evaluation Metric
1. Exact Match(EM)
    - 모델의 예측과 실제 답이 정확하게 일치하는 경우 1, 아니면 0
    - 띄어쓰기나 특수문자를 제외하여 비교
    - 여러 개의 실제 답 중 하나라도 일치하는 경우 정답
2. F1 Score
    - 리더보드에 반영되지 않는 참고용 점수
    - 예측한 답과 ground-truth 사이의 token overlap을 f1으로 계산

## 대회 결과
|리더보드|순위|EM|F1|
|:---:|:---:|:---:|:---:|
|Public|8|68.33|78.57|
|Private|10 (2🔻)|65.0|77.03|