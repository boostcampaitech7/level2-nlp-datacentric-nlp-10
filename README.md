# Data-Centric NLP (Topic Classification)
본 프로젝트는 뉴스의 헤드라인을 통해, 뉴스가 어떤 topic을 갖는지 분류하는 태스크입니다.

<br><br>

## 1. Overview

### 🚩 Data-Centric NLP (Topic Classification) Project
뉴스의 `헤드라인`으로 뉴스가 어떤 `topic`을 갖는지 `분류`하는 Project입니다.

<br>

> **🔥 Caution**
> - Data-Centric의 취지에 맞게, 베이스라인 모델은 수정 불가
> - Only 데이터의 수정으로만 성능 향상을 이끌어내야 함

<br>

|가능한 방법|불가능한 방법|
|---|---|
|- 공개된 생성 모델 (T5 등)을 통한 Synthetic Data 생성 <br> - 각종 Data Augmentation 기법 적용 <br> - Data sampling 기법 <br> - Negative sampling 등 <br> - Data Quality Control <br> - Data labeling error detection, Data Cleaning, Data Filtering 등 <br> - 다양한 Self-training 기법 (코드가 아닌 데이터 변경) <br> - 모델의 변경이 아닌 베이스라인 코드의 변경 <br> | - 유료 버전의 비공개 생성 모델을 활용하는 모든 방법 <br> (GPT-4, ChatGPT (GPT-3.5), api key 결제 일체 포함) <br> - 베이스라인 코드 내 모델의 변경을 야기하는 모든 방법들 <br> - 테스트셋 정보를 활용한 모든 방법 <br> - 외부 데이터셋 사용 <br> - Active learning, Curriculum learning 등의 데이터 수정이 <br> 아닌 모델을 업데이트 하는 방법 <br>|

<br> 

**모델 성능 평가 지표(Metrics)**
- Accuracy(정확도) : TP+TN / TP+FP+TN+FN
- F1 score : 2 * Precision * Recall / Precision + Recall

<br><br>

## 2. 프로젝트 구성

### ⏰ 개발 기간
- 2024년 10월 28일(월) 10:00 ~ 2024년 11월 7일(목) 19:00
- 부스트캠프 AI Tech NLP 트랙 11-12주차
  
  |Title|Period|Days|Description|
  |:---:|:---:|:---:|:---:|
  |필요 지식 학습|10.28 ~ 10.31|4 days|Data-centric NLP에 대해 이해|
  |데이터 분석 및 EDA|10.31 ~ 11.01|2 days|데이터 구조 및 프로젝트 제한 사항에 대한 이해|
  |데이터 노이즈 제거|11.01 ~ 11.03|3 days|데이터 노이즈 제거 방법 개선|
  |LLM을 활용한 데이터 증강 및 고도화|11.03 ~ 11.07|4 days|실험 및 성능 개선|

<br>

### ✨ 분석 환경
- Upstage AI Stages 제공 V100 GPU Server 활용
- OS : Linux
- Language : Python
- Libraries(mainly used) : Pytorch, Hugging Face, Wandb etc.

<br>

### 💾 Data
- 결측치와 중복 값이 없는 Train, Test 데이터

||Samples|Description|
|:---:|:---:|:---:|
|Train|2,800|ID, text, target으로 이루어진 학습 데이터|
|Test|30,000|ID, text로 이루어진 평가 데이터|


<br><br>


### 💡 수행 내용
#### 1. 데이터 분리 Task
  1. Text 내 영어, 숫자, 특수문자를 기준으로 Noise 데이터 1차 분류
  2. Rule Base로 Noise 데이터 2차 분류

<br>

  **Noise Patterns**
  
```python
      abnormal_patterns = [
        r'[A-Za-z]{3,}[0-9]{3,}',       # 영어 3자 이상 + 숫자 3자 이상 (너무 짧은 패턴 제외)
        r'[0-9]{3,}[A-Za-z]{3,}',       # 숫자 3자 이상 + 영어 3자 이상
        r'[A-Za-z]{2,}[^A-Za-z0-9가-힣\s]{2,}',  # 영어 2자 이상 + 특수문자 2자 이상
        r'[^A-Za-z0-9가-힣\s]{2,}[A-Za-z]{2,}',  # 특수문자 2자 이상 + 영어 2자 이상
        r'[A-Za-z]{2,}-[0-9]+-[A-Za-z]{2,}',  # 영어 2자 이상 - 숫자 - 영어 2자 이상
        r'[^A-Za-z0-9가-힣\s]{3,}',    # 연속된 특수 문자 3자 이상
        r'^(?=[^{}\(\*\[\]]*[}\(\*\[][^{}\(\*\[\]]*$)',  # 특정 기호가 하나만 포함된 문자열 필터링
        r'[A-Za-z]{1,2}[^A-Za-z0-9가-힣\s]+[가-힣]',  # 영어 1~2자 + 특수문자 + 한글
        r'[가-힣]+[^A-Za-z0-9가-힣\s]+[A-Za-z]{1,2}',  # 한글 + 특수문자 + 영어 1~2자
        r'[가-힣]+[^A-Za-z0-9가-힣\s]+[가-힣]+[A-Za-z]',  # 한글 + 특수문자 + 한글 + 영어  
        r'[^A-Za-z0-9가-힣\s]+[0-9]+[^A-Za-z0-9가-힣\s]+' # 특수문자 + 숫자 + 특수문자
      ]
```
  
<br>

#### 2. 데이터 노이즈 복원

<br>

#### 3. 데이터 증강

<br><br>

## 3. 프로젝트 결과
|Idx  |Public Accuracy|Public F1|Private Accuracy|Private F1|
|:---:|:---:|:---:|:---:|:---:|
|1|||||
|2|||||

- 


<br><br>


## 4. Team
<table>
    <tbody>
        <tr>
            <td align="center">
                <a href="https://github.com/Kimyongari">
                    <img src="https://github.com/Kimyongari.png" width="100px;" alt=""/><br />
                    <sub><b>Kimyongari</b></sub>
                </a><br />
                <sub>김용준</sub>
            </td>
            <td align="center">
                <a href="https://github.com/son0179">
                    <img src="https://github.com/son0179.png" width="100px;" alt=""/><br />
                    <sub><b>son0179</b></sub>
                </a><br />
                <sub>손익준</sub>
            </td>
            <td align="center">
                <a href="https://github.com/P-oong">
                    <img src="https://github.com/P-oong.png" width="100px;" alt=""/><br />
                    <sub><b>P-oong</b></sub>
                </a><br />
                <sub>이현풍</sub>
            </td>
            <td align="center">
                <a href="https://github.com/Aitoast">
                    <img src="https://github.com/Aitoast.png" width="100px;" alt=""/><br />
                    <sub><b>Aitoast</b></sub>
                </a><br />
                <sub>정석현</sub>
            </td>
            <td align="center">
                <a href="https://github.com/uzlnee">
                    <img src="https://github.com/uzlnee.png" width="100px;" alt=""/><br />
                    <sub><b>uzlnee</b></sub>
                </a><br />
                <sub>정유진</sub>
            </td>
            <td align="center">
                <a href="https://github.com/hayoung180">
                    <img src="https://github.com/hayoung180.png" width="100px;" alt=""/><br />
                    <sub><b>hayoung180</b></sub>
                </a><br />
                <sub>정하영</sub>
            </td>
        </tr>
    </tbody>
</table>

<br><br>

---

<br>

## Reference
1.

