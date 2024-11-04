from tqdm import tqdm
import pandas as pd
import re
from aug_with_llm import augmentation
import torch
import gc
from train_and_inference import baselinemodel
import numpy as np
from cleanlab.filter import find_label_issues
from backtranslate import BackTranslator
import os

class flywheel:
    def __init__(self):
        route = "/data/ephemeral/home/code/data/processed_train.csv"
        if not os.path.exists(route):
            route = "/data/ephemeral/home/code/data/train.csv"

        origin_data = pd.read_csv(route)
        origin_data['abnormal'] = origin_data['text'].apply(self.is_abnormal)
        print(f"이상으로 판단된 train의 수 : {len(origin_data[origin_data.abnormal])}\n")

        self.contaminated_label_data = origin_data[~origin_data['abnormal']]
        self.contaminated_text_data = origin_data[origin_data['abnormal']]

        # 라벨이 클린한 데이터는 라벨을 살리기 위해 문장을 복구
        # 텍스트가 클린한 데이터는 복구한 데이터와 합쳐 일단 학습
        # 그 뒤에 추론하고 라벨이 이상한 데이터를 cleanlab으로 복구
        # 텍스트가 클린한 데이터를 backtranslation 및 LLM으로 증강 후 추론
        # 데이터를 전부 합쳐서 새로운 데이터 구축

    # 이상한 문장인지 판단
    def is_abnormal(self, text):
        try:
            if not isinstance(text, str) or len(text.strip()) == 0:
                return True
                
            # 공백 제거 (길이 대비 비율 도출 위함)
            text_no_space = text.replace(" ", "")
            total_len = len(text_no_space)
            
            # 문자 유형별 개수 및 비율
            hangul_ratio = len(re.findall(r'[가-힣]', text_no_space)) / total_len
            special_chars = re.findall(r'[^가-힣a-zA-Z0-9\s,.…"\'%↑↓→←]', text_no_space)
            special_ratio = len(special_chars) / total_len
            
            # 비정상 패턴
            abnormal_patterns = re.findall(r'[A-Za-z][0-9]|[0-9][A-Za-z]|[A-Za-z][^A-Za-z0-9\s]|[^A-Za-z0-9\s][A-Za-z]', text)
            pattern_ratio = len(abnormal_patterns) / total_len
            
            # Threshold
            conditions = [
                special_ratio > 0.15,                    # 특수문자 비율
                hangul_ratio < 0.4,                      # 한글 비율
                pattern_ratio > 0.1,                     # 비정상 패턴 비율
                bool(re.search(r'[A-Za-z]\d[A-Za-z]', text)),  # 알파벳-숫자-알파벳 패턴
                bool(re.search(r'[^가-힣a-zA-Z0-9\s]{2,}', text)),  # 연속된 특수문자
                len(set(special_chars)) > 2              # 다양한 종류의 특수문자
            ]
            
            return any(conditions)
            
        except Exception as e:
            print("Error:", e)
            return True
    
    def text_cleaning(self, df):
        model = augmentation()
        result = model.generate_clean_data(df)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return result
    
    def text_synonym(self, df):
        model = augmentation()
        result = model.generate_synonym_data(df)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return result
    
    def inference_and_cleanlab(self):
        text_cleaned_data = self.text_cleaning(self.contaminated_text_data)
        all_data = pd.concat([text_cleaned_data, self.contaminated_label_data]).reset_index(drop = True)
        origin_data = all_data.copy()
        labels = all_data['target'].copy()
        model = baselinemodel()
        model.train(all_data)
        model.inference(all_data)
        probs = np.vstack(model.result['logits'])
        label_issues = find_label_issues(labels, probs, 
                                 return_indices_ranked_by = 'self_confidence')
        print('발견한 라벨이슈 :', len(label_issues))
        print(origin_data.loc[label_issues, 'target'].head(5))
        print(all_data.loc[label_issues, 'target'].head(5))
        origin_data.loc[label_issues, 'target'] = all_data.loc[label_issues, 'target']
        self.processed_data = origin_data
        print('텍스트를 고친 데이터 + 라벨이 이상한 데이터 :',len(origin_data))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    def make_new_data(self):
        self.processed_data['abnormal'] = self.processed_data['text'].apply(self.is_abnormal)

        model = BackTranslator()
        back_translated = []
        text_clean_data = self.processed_data[~self.processed_data['abnormal']]
        print('텍스트가 정상인 데이터의 수 :', len(text_clean_data))
        print('텍스트가 정상인 데이터를 역번역으로 증강합니다.')
        for idx, item in tqdm(enumerate(text_clean_data.iterrows())):
            tmp = {'ID' : f'aug_{idx}', 
                   'text' : model.back_translate(item['text']),
                   'target' : item['target']}
            back_translated.append(tmp)
        back_translated = pd.DataFrame(back_translated)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = augmentation()
        print('테스트가 정상인 데이터를 LLM을 통해 증강합니다.')
        regenerated = model.generate_synonym_data(text_clean_data)
        regenerated['len'] = regenerated['text'].apply(lambda x: len(x))
        regenerated = regenerated[regenerated['len'] < 100]
        print('이상하게 증강된 데이터를 삭제합니다.')
        print('이상하게 증강된 데이터의 개수 :', len(regenerated['len']>=100))

        del model
        gc.collect()
        torch.cuda.empty_cache()

        new_data = pd.concat([regenerated, back_translated, self.processed_data])[['ID', 'text', 'target']].reset_index(drop = True)
        print('역번역 데이터 + 재구성 데이터 + label cleaning 데이터의 개수 :', len(new_data))
        new_data.to_csv('/data/ephemeral/home/datacentric/processed_train.csv', index = False)
        
        