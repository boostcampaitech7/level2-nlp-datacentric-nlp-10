import random
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

class BackTranslator:
    def __init__(self):
        # M2M100 모델과 토크나이저 로드
        self.model_name = "facebook/m2m100_418M"
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name)

    def translate(self, text, src_lang, tgt_lang):
        # 소스 언어와 타겟 언어 설정
        self.tokenizer.src_lang = src_lang
        encoded_text = self.tokenizer(text, return_tensors="pt")
        
        # 번역 수행
        generated_tokens = self.model.generate(**encoded_text, forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang))
        return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    def back_translate(self, text):
        # 중간 언어 선택 (영어 또는 일본어 중 랜덤 선택)
        intermediate_language = random.choice(["en", "ja"])
        
        # 한국어에서 중간 언어로 번역
        translated_text = self.translate(text, src_lang="ko", tgt_lang=intermediate_language)
        
        # 중간 언어에서 한국어로 역번역
        back_translated_text = self.translate(translated_text, src_lang=intermediate_language, tgt_lang="ko")
        
        return back_translated_text
