from tqdm import tqdm
import pandas as pd
import numpy as np
import transformers
import torch
import re

class augmentation:
    def __init__(self):
        model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        self.pipeline.model.eval()

    def generate(self, PROMPT, instruction, temperature):

        messages = [
            {"role": "system", "content": f"{PROMPT}"},
            {"role": "user", "content": f"{instruction}"}
            ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=50,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=0.9
        )
        output = outputs[0]['generated_text'][len(prompt):]
        output = re.sub(r'"([^"]*)"', r'\1', output)

        return output
    
    def generate_random_data(self, cnt):
        aug = []
        for i in tqdm(range(cnt)):
            temperature = np.random.uniform(0.6, 0.95)
            PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly.
            당신은 기자의 어시스턴트 입니다. 사용자의 질문에 대한 기사 제목을 만들어 주세요.'''
            instruction = f"다양한 주제로 뉴스 기사 제목을 한개만 만들어 줘."
            text = self.generate(PROMPT, instruction, temperature)[1:-1]
            tmp = {'ID' : f'aug{i}', 'text' : text, 'target' : -1}
            aug.append(tmp)

        aug_df = pd.DataFrame(aug)
        return aug_df
    
    def generate_clean_data(self, df):
        aug = []
        for idx, (index, row) in tqdm(enumerate(df.iterrows())):
            temperature = np.random.uniform(0.6, 0.95)

            PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly.
            당신은 오염된 텍스트를 정상적인 문장으로 복원하는 전문가입니다.'''
            instruction = f"아래에 주어진 문장을 읽고 정상적인 한국어 문장으로 고쳐줘:\n\n'{row['text']}'"
            
            text = self.generate(PROMPT, instruction, temperature)
            tmp = {'ID': f'aug{idx}', 'text': text, 'target': row['target']}
            aug.append(tmp)
        aug_df = pd.DataFrame(aug)

        return aug_df
    
    def generate_synonym_data(self, df):
        aug = []
        for idx, (index, row) in tqdm(enumerate(df.iterrows())):
            temperature = np.random.uniform(0.6, 0.95)
            PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly.
            당신은 문장을 재구성하는 전문가입니다. 주어진 문장과 의미가 동일한 새로운 문장을 생성해주세요.'''

            instruction = f"아래 문장과 동일한 의미를 갖는 새로운 문장을 하나만 만들어줘:\n\n'{row['text']}'"
            text = self.generate(PROMPT, instruction, temperature)
            tmp = {'ID': f'aug{idx}', 'text': text, 'target': row['target']}
            aug.append(tmp)
        aug_df = pd.DataFrame(aug)

        return aug_df
    # 0 - 생활문화 / 1 - 스포츠 / 2 - 국내정치 / 3 - 사회 / 4 - IT과학 / 5 - 경제 / 6 - 세계

    def generate_correct_data(self, df):
        # 토픽 모델링을 통해 얻은 라벨별 토픽들을 keyword로 사용합니다.
        a = pd.read_csv('/data/ephemeral/home/datacentric/아카이브/target_0_drop_df.csv')
        b = pd.read_csv('/data/ephemeral/home/datacentric/아카이브/target_1_drop_df.csv')
        c = pd.read_csv('/data/ephemeral/home/datacentric/아카이브/target_2_drop_df.csv')
        d = pd.read_csv('/data/ephemeral/home/datacentric/아카이브/target_3_drop_df.csv')
        e = pd.read_csv('/data/ephemeral/home/datacentric/아카이브/target_4_drop_df.csv')
        f = pd.read_csv('/data/ephemeral/home/datacentric/아카이브/target_5_drop_df.csv')
        g = pd.read_csv('/data/ephemeral/home/datacentric/아카이브/target_6_drop_df.csv')


        aug = []
        keywords = [a['word'].values,b['word'].values,c['word'].values,d['word'].values,e['word'].values,f['word'].values,g['word'].values]

        for index, row in tqdm(df.iterrows()):
            temperature = np.random.uniform(0.6, 0.95)
            PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly.
                    당신은 문장을 재구성하는 전문가입니다. 원래 문장과 몇 가지 키워드가 주어졌을 때, 이 키워드 중 몇 가지를 사용하여 새로운 기사 제목을 만들어주세요.'''

            instruction = f"주어진 키워드를 사용하여 40자 이내의 새로운 기사 제목을 한 개 만들어 주세요:\n\n원래 문장: '{row['text']}'\n키워드: {keywords[row['target']]}"
            text = self.generate(PROMPT, instruction, temperature)
            tmp = {'ID': f'aug{index}', 'text': text, 'target': row['target']}
            aug.append(tmp)
        aug_df = pd.DataFrame(aug)
        
        return aug_df
