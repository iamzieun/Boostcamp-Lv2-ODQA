import os
import json

class FileLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def load_nbest_files(self):
        nbest_files = os.listdir(self.folder_path)
        return nbest_files

    def load_ids(self, nbest_files):
        with open(os.path.join(self.folder_path, nbest_files[0]), 'r') as file:
            data = json.load(file)
        ids = list(data.keys())
        return ids

class DataProcessor:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def nbest_files_concat(self, nbest_files):
        nbest_files_concat_list = []

        for file in nbest_files:
            with open(os.path.join(self.folder_path, file), 'r') as contents:
                data = json.load(contents)
                nbest_files_concat_list.append(data)

        return nbest_files_concat_list


    def mk_answer_candidates_dict(self, ids, nbest_files_concat_list):
        '''
            각 nbest 파일마다 20개의 answer 후보가 만들어져 있습니다. 
            서로 다른 nbest 파일에 같은 answer 후보가 만들어져 answer candidate의 중복이 발생할 수 있습니다. 
            따라서, voting 전 answer candidate의 중복을 제거한 딕셔너리를 만들기 위한 과정입니다.
        '''
        answer_candidates_dict = {}

        for i in ids:
            answer = {}
            answer_candidate_text_01 = []
            num_nbest_files = len(nbest_files_concat_list)

            for f in range(num_nbest_files):
                answer_candidate_text_02 = []
                num_answer_candidates =len(nbest_files_concat_list[f][i])

                for k in range(num_answer_candidates):
                    answer_candidate_text_02.append(nbest_files_concat_list[f][i][k]['text'])

                answer_candidate_text_01 += answer_candidate_text_02

            # 앞서 만들어진 answer_candidate_text_01 에는 중복이 있을 수 있습니다. 
            # 중복을 제거하고 각 answer_text를 key로, 빈 리스트를 value로 하는 딕셔너리를 생성해 줍니다.
            for answer_text in list(set(answer_candidate_text_01)):
                answer[answer_text] = []
            
            # 앞서 만들어진 딕셔너리를 각 id에 맞춰 다시 딕셔너리로 만들어 줍니다.
            answer_candidates_dict[i] = answer

        return answer_candidates_dict


    def fill_answers_probability(self, ids, nbest_files_concat_list, answer_candidates_dict):
        '''
            앞서 만들어진 answer_candidates_dict의 최하단 value값은 빈 리스트 입니다.
            for문을 돌며, 각 id에 대한 probability 값을 채워줍니다.
        '''
        for i in ids :
            num_nbest_files = len(nbest_files_concat_list)
            for f in range(num_nbest_files) :
                num_answer_candidates = len(nbest_files_concat_list[f][i])
                for k in range(num_answer_candidates) :
                    text = nbest_files_concat_list[f][i][k]['text']
                    if text in answer_candidates_dict[i].keys() :
                        answer_candidates_dict[i][text] += [nbest_files_concat_list[f][i][k]['probability']]
        return answer_candidates_dict


    def voting(self, ids, answer_candidates_dict):
        final_ans_dict = {}

        for i in ids:
            # 각 아이디에 해당하는 answer 후보들의 평균을 구해 voting을 위한 딕셔너리를 만들어 줍니다.
            voting_dict = {}
            text_list = []
            mean_list = []

            ans_dict = answer_candidates_dict[i]
            for text in ans_dict.keys():
                value_list = ans_dict[text]
                mean = sum(value_list) / len(value_list)
                text_list.append(text)
                mean_list.append(mean)

            voting_dict['text'] = text_list
            voting_dict['mean'] = mean_list

            # answer 후보들의 평균 중, 가장 큰 값을 가지는 answer text를 최종 답으로 선정합니다. 
            max_value_index = mean_list.index(max(mean_list))
            final_ans_dict[i] = text_list[max_value_index]

        return final_ans_dict


    def mk_prediction_json(self, final_ans):
        with open('soft_voting_predictions.json', 'w') as f:
            json.dump(final_ans, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    folder_path = "./nbest_files"

    file_loader = FileLoader(folder_path)
    nbest_files = file_loader.load_nbest_files()
    ids = file_loader.load_ids(nbest_files)

    data_processor = DataProcessor(folder_path)
    nbest_files_concat_list = data_processor.nbest_files_concat(nbest_files)
    answer_candidates_dict = data_processor.mk_answer_candidates_dict(ids, nbest_files_concat_list)
    answer_candidates_dict = data_processor.fill_answers_probability(ids, nbest_files_concat_list, answer_candidates_dict)
    final_ans = data_processor.voting(ids, answer_candidates_dict)
    data_processor.mk_prediction_json(final_ans)
