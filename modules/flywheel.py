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
from aug_with_solar import augumentation_solar

class flywheel:
    # 만약 이전에 처리된 데이터가 있다면 그 데이터를 대상으로 불러오고, 이전에 처리된 데이터가 없다면 복구된 데이터를 불러옵니다.
    def __init__(self, route = "/data/ephemeral/home/datacentric/processed_train.csv"):
        self.route = route
        
        if not os.path.exists(self.route):
            self.route = "/data/ephemeral/home/code/data/train_denoised_none.csv"

        origin_data = pd.read_csv(self.route).dropna()
        origin_data = origin_data[origin_data['text'].apply(lambda x: len(x)) > 10]
        if self.route == "/data/ephemeral/home/datacentric/processed_train.csv":
            origin_data = origin_data[~origin_data['text'].str.contains('문장')]
            origin_data = self.balance_target_data(origin_data).reset_index(drop = True)
        origin_data['abnormal'] = origin_data['text'].apply(self.is_abnormal)
        print(f"이상으로 판단된 train의 수 : {len(origin_data[origin_data.abnormal])}\n")

        self.contaminated_label_data = origin_data[~origin_data['abnormal']]
        self.contaminated_text_data = origin_data[origin_data['abnormal']]

        # 라벨이 클린한 데이터는 라벨을 살리기 위해 문장을 복구
        # 텍스트가 클린한 데이터는 복구한 데이터와 합쳐 일단 학습
        # 그 뒤에 추론하고 라벨이 이상한 데이터를 cleanlab으로 복구
        # 텍스트가 클린한 데이터를 backtranslation 및 LLM으로 증강 후 추론
        # 데이터를 전부 합쳐서 새로운 데이터 구축

    def balance_target_data(self, df, target_col='target', id_col='ID', prefix='aug'):
        # 'aug'로 시작하는 행들만 필터링
        aug_rows = df[df[id_col].str.startswith(prefix)]
        
        # 각 클래스별 개수 확인
        target_counts = df[target_col].value_counts()
        min_count = target_counts.min()
        max_count = target_counts.max()
        print('가장 적은 target :', min_count)
        print('가장 많은 target :', max_count)

        # 삭제할 행 목록을 담을 리스트 초기화
        rows_to_drop = []

        for target, count in target_counts.items():
            # 최소 개수에 맞춰 각 target에 대한 초과 행 계산
            excess_count = count - min_count
            if excess_count > 0:
                # 'aug'로 시작하는 초과 행만큼 삭제할 행을 선택
                rows_to_remove = aug_rows[aug_rows[target_col] == target].head(excess_count)
                rows_to_drop.extend(rows_to_remove.index)

        # 초과 행 삭제
        df_balanced = df.drop(rows_to_drop)
        return df_balanced

    # 이상한 문장인지 판단
    def is_abnormal(self, text):
        abnormal_patterns = [
            r'[A-Za-z]{3,}[0-9]{3,}',       # 영어 3자 이상 + 숫자 3자 이상 (너무 짧은 패턴 제외)
            r'[0-9]{3,}[A-Za-z]{3,}',       # 숫자 3자 이상 + 영어 3자 이상
            r'[A-Za-z]{2,}[^A-Za-z0-9가-힣\s]{2,}',  # 영어 2자 이상 + 특수문자 2자 이상
            r'[^A-Za-z0-9가-힣\s]{2,}[A-Za-z]{2,}',  # 특수문자 2자 이상 + 영어 2자 이상
            r'[A-Za-z]{2,}-[0-9]+-[A-Za-z]{2,}',  # 영어 2자 이상 - 숫자 - 영어 2자 이상
            r'[^A-Za-z0-9가-힣\s]{3,}',    # 연속된 특수 문자 3자 이상
            r'^(?=[^{}\(\*\[\]]*[}\(\*\[][^{}\(\*\[\]]*$)',  # 특정 기호가 하나만 포함된 문자열 필터링
            r'[A-Za-z]{1,2}[^A-Za-z0-9가-힣\s]+[가-힣]',  # 영어 1~2자 + 특수문자 + 한글
            r'[가-힣]+[^A-Za-z0-9가-힣\s]+[A-Za-z]{1,2}',  # 한글 + 특수문자 + 영어 1~2자
            r'[가-힣]+[^A-Za-z0-9가-힣\s]+[가-힣]+[A-Za-z]',  # 한글 + 특수문자 + 한글 + 영어  
            r'[^A-Za-z0-9가-힣\s]+[0-9]+[^A-Za-z0-9가-힣\s]+', # 특수문자 + 숫자 + 특수문자
        ]
        return any(re.search(pattern, text) for pattern in abnormal_patterns)
    
    def text_cleaning(self, df, name = 'llama'):
        gc.collect()
        torch.cuda.empty_cache()
        if name == 'llama':
            model = augmentation()
        elif name == 'solar':
            model = augumentation_solar()
        result = model.generate_clean_data(df)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return result
    
    def text_synonym(self, df, name = 'llama'):
        gc.collect()
        torch.cuda.empty_cache()
        if name == 'llama':
            model = augmentation()
        elif name == 'solar':
            model = augumentation_solar()
        model = augmentation()
        result = model.generate_synonym_data(df)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def backtranslation(self, df):
        model = BackTranslator()
        back_translated = []

        for idx, row in tqdm(df.iterrows()):
            tmp = {
                'ID': f'aug_{idx}',  # 원래 인덱스를 그대로 사용
                'text': model.back_translate(row['text']),
                'target': row['target']
            }
            back_translated.append(tmp)
        back_translated = pd.DataFrame(back_translated)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        return back_translated
    
    def inference_and_cleanlab(self, name = 'llama'):
        if len(self.contaminated_text_data) > 0:
            text_cleaned_data = self.text_cleaning(self.contaminated_text_data, name)
            all_data = pd.concat([text_cleaned_data, self.contaminated_label_data]).reset_index(drop = True)
            all_data['len'] = all_data['text'].apply(lambda x: len(x))
            all_data = all_data[all_data['len'] < 70].reset_index(drop = True)
        else:
            all_data = self.contaminated_label_data
        
        origin_data = all_data.copy()
        labels = all_data['target'].copy()
        model = baselinemodel(model_name = 'klue/roberta-large')
        if self.route == '/data/ephemeral/home/code/data/train_denoised_none.csv':
            train_data = pd.read_csv('/data/ephemeral/home/level2-nlp-datacentric-nlp-10/noise_fix_label_right.csv').dropna()
            model.train(train_data)
        else:
            model.train(all_data)
        model.inference(all_data)
        probs = np.vstack(model.result['logits'])
        label_issues = find_label_issues(labels, probs, 
                                 return_indices_ranked_by = 'self_confidence')
        print('발견한 라벨이슈 :', len(label_issues))
        print(origin_data.loc[label_issues, 'target'].head(5))
        print(all_data.loc[label_issues, 'target'].head(5))
        origin_data.loc[label_issues, 'target'] = all_data.loc[label_issues, 'target']
        model.result['max_logits'] = model.result['logits'].apply(lambda x: max(x))
        low_logit_idxs = model.result[model.result['max_logits'] < 0.8].index
        del model
        gc.collect()
        torch.cuda.empty_cache()

        issue_data = origin_data.loc[label_issues]
        low_logit_data = origin_data.loc[low_logit_idxs]

        model = augmentation()
        print('issue_data들을 보강합니다.')
        correct_data = model.generate_correct_data(issue_data)
        print('logit이 낮은 데이터를 보강합니다.')
        correct_data2 = model.generate_correct_data(low_logit_data)
        # drop_labels = set(label_issues).union(set(low_logit_idxs))
        # origin_data = origin_data.drop(drop_labels)

        origin_data = pd.concat([origin_data, correct_data, correct_data2], ignore_index = True).reset_index(drop = True)

        
        filter_condition = origin_data['text'].str.contains('원래|문장|기사|키워드|문장', na=False)
        origin_data = origin_data[~filter_condition].reset_index(drop=True)
        self.processed_data = origin_data
        print('텍스트를 고친 데이터 + 라벨이 이상한 데이터 :',len(origin_data))
    
    def make_new_data(self, cnt, name = 'llama'):
        self.processed_data['abnormal'] = self.processed_data['text'].apply(self.is_abnormal)
        mm = self.processed_data['target'].value_counts().max() + cnt
        text_clean_data = self.processed_data[~self.processed_data['abnormal']]

        print(f'역번역과 LLM증강으로 각각 {cnt}개의 데이터를 만듭니다.')
        print('텍스트가 정상인 데이터의 수 :', len(text_clean_data))
        print('텍스트가 정상인 데이터를 역번역으로 증강합니다.')

        back_translated = pd.DataFrame()
        for i in range(7):
            tmp = self.processed_data[self.processed_data['target'] == i]
            back = self.backtranslation(tmp.sample(mm - tmp['target'].count()))
            back_translated = pd.concat([back_translated, back])
            torch.cuda.empty_cache()
            gc.collect()


        new_data = pd.concat([self.processed_data, back_translated])
        mm = new_data['target'].value_counts().max() + cnt

        print('테스트가 정상인 데이터를 LLM을 통해 증강합니다.')
        regenerated = pd.DataFrame()
        for i in range(7):
            tmp = self.processed_data[self.processed_data['target'] == i]
            regen = self.text_synonym(tmp.sample(mm - tmp['target'].count()))
            regenerated = pd.concat([regenerated, regen])
            torch.cuda.empty_cache()
            gc.collect()

        print('이상하게 증강된 데이터를 삭제합니다.')
        regenerated['len'] = regenerated['text'].apply(lambda x: len(x))
        abnormal_count = (regenerated['len'] >= 70).sum()  # 70 이상인 데이터 수 계산
        print('이상하게 증강된 데이터의 개수 :', abnormal_count)

        regenerated = regenerated[regenerated['len'] < 70]  # 70 이상인 데이터 삭제

        # 새로운 데이터 합치기
        new_data = pd.concat([new_data, regenerated])[['ID', 'text', 'target']].reset_index(drop=True)
        new_data = new_data[~new_data['text'].str.contains('문장')]
        new_data = new_data[~new_data['text'].str.contains('기사 제목')]
        new_data['text'] = new_data['text'].apply(lambda x: x.strip("'"))
        # 잘못 증강된 데이터들을 필터링
        print('역번역 데이터 + 재구성 데이터 + label cleaning 데이터의 개수 :', len(new_data))
        new_data.to_csv('/data/ephemeral/home/datacentric/processed_train.csv', index = False)
        
        
        