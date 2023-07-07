import json
import argparse
import os
import glob
import random
from collections import Counter

def voting(args:argparse.Namespace) -> None:

    files = []
    predictions_path = os.path.join(args.predictions, '*.json')

    for path in glob.glob(predictions_path):
        with open(path) as f:
            files.append(json.load(f))

    m = (len(files)//2) + 1             # 사용되는 prediction 파일 수의 과반수
    random_choice = args.random         # True가 default이며 False일 시 가장 먼저 등장한 최빈값을 선택함.

    predict = {}
    for key in files[0]:
        count = Counter([f[key] for f in files])
        poll = count.most_common(1)
        predict[key] = poll[0][0]

        if poll[0][1]<m and random_choice:
            poll = random.choice([(k, count[k]) for k in count if count[k]==poll[0][1]])
            predict[key] = poll[0]

    with open(args.output, 'w', encoding="utf-8") as f:
        json.dump(predict, f, ensure_ascii=False, indent=4)

    print('Done.')


if __name__ =='__main__':

    parser = argparse.ArgumentParser(description='Hard Voting')

    parser.add_argument('--predictions', type=str ,    default='predictions')
    parser.add_argument('--random'     , type=bool,    default=True)
    parser.add_argument('--output'     , type=str ,    default='hard_voting_output.csv')

    args = parser.parse_args()
    print(f'predictions path\t: {args.predictions}')
    print(f'random choice\t\t: {args.random}')
    print(f'output path\t\t: {args.output}')

    voting(args)
