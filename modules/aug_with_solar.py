import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re
import pandas as pd
import numpy as np

class augumentation_solar:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Upstage/SOLAR-10.7B-Instruct-v1.0")
        self.model = AutoModelForCausalLM.from_pretrained(
            "Upstage/SOLAR-10.7B-Instruct-v1.0",
            device_map="auto",
            torch_dtype=torch.float16,
        )
    def generate(self, PROMPT, instruction, temperature):

        messages = [
            {"role": "system", "content": f"{PROMPT}"},
            {"role": "user", "content": f"{instruction}"}
            ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 모델을 사용해 텍스트 생성
        outputs = self.model.generate(**inputs, use_cache=True, max_length=200, temperature=temperature)
        
        # 텍스트 디코딩
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = output_text[len(prompt):].strip()
        generated_text = re.sub(r'"([^"]*)"', r'\1', generated_text)

        return generated_text
    
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
