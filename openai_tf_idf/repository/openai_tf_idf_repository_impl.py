import asyncio
import os

import pandas as pd

import faiss
import openai
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

from dotenv import load_dotenv

from openai_tf_idf.repository.mongodb_repository_impl import MongodbRepositoryImpl
from openai_tf_idf.repository.openai_tf_idf_repository import OpenAITfIdfRepository


load_dotenv()

openaiApiKey = os.getenv('OPENAI_API_KEY')
if not openaiApiKey:
    raise ValueError('API Key가 준비되어 있지 않습니다!')

openai.api_key = openaiApiKey

class OpenAITfIdfRepositoryImpl(OpenAITfIdfRepository):
    __instance = None

    headers = {
        'Authorization': f'Bearer {openaiApiKey}',
        'Content-Type': 'application/json'
    }

    OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


    SYSTEM_MESSAGE = r"""
    너는 문체 변환 업무를 담당해.
    
    이제부터 내가 "문체 변환 <INTENTION> <TYPE> <PHASE>"라는 문단을 줄거야. 
    <INTENTION>에는 헬스케어 질문 목적 데이터가, <TYPE>에는 아래에서 정의한 변환 조건의 값이, <PHASE>에는 너가 변환해야하는 문체의 단락이 들어가 있어.
    작성 시 <INTENTION> 목적에 맞게, <TYPE> 조건에 맞게 <PHASE>의 문체를 바꿔줘.
    최종 출력은 출력 조건에 맞게 출력해줘.
    
    변환 조건: 
    - 만약 TYPE이 "F"라면:
        1. 아픔에 공감하는 방식으로 시작해.
        2. 2인칭으로 지칭해.
        3. 친한 지인과 나누는 구어체에 가깝게 작성해.
        4. 문장 종결은 '하십시오체' 말고 '해요체'로 작성해.
        5. 단락 당 걱정의 말을 한 문장 이상 넣어줘.
        6. 마지막 문장에서는 회복과 극복의 긍정 멘트를 넣어줘.
    
    - 만약 TYPE이 "T"라면:
        1. 정보 전달이 명확하게 가도록 가독성 있게 작성해.
        2. 정보가 나열되어 있다면 <ul style="list-style-position:inside;">로 나열해.
        3. 단락이 나누어지는 문장이면 <p>태그로 나눠서 작성해.
        4. 마지막을 결론 한 문장으로 작성해.
    
    출력 조건:
    - 만약 문단이 나눠졌다면:
        1. <p>태그로 분리해서 출력해.
        2. 띄어쓰기 넣지마.
        3. 문장 사이에 \n 넣지마. """


    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__mongodbRepository = MongodbRepositoryImpl.getInstance()

        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance


    def getFaissIndex(self, intention):
        embeddedAnswer = self.__mongodbRepository.getEmbeddings(intention)

        # FAISS 인덱스 생성 및 임베딩 데이터 추가
        embeddingVectorDimension = len(embeddedAnswer[0])
        faissIndex = faiss.IndexFlatL2(embeddingVectorDimension)
        faissIndex.add(np.array(embeddedAnswer).astype('float32'))

        return faissIndex

    def getOriginalAnswer(self, intention):
        ORIGINAL_DATA_PATH = os.path.join(os.getcwd(), 'assets', f'{intention}_original_answers.csv')
        return pd.read_csv(ORIGINAL_DATA_PATH)

    def openAiBasedEmbedding(self, text):
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )

        return response['data'][0]['embedding']

    def similarityAnalysis(self, openAIEmbedding, faissIndex, top_k, originalAnswersLength):
        embeddingUserQuestion = np.array(openAIEmbedding).astype('float32').reshape(1, -1)

        distanceList, indexList = faissIndex.search(embeddingUserQuestion, top_k)

        # 인덱스 유효성 확인 및 답변 가져오기
        # if any([idx >= originalAnswersLength for idx in indexList]):
        #     raise ValueError(f"Invalid index found in indexList: {indexList}")

        return indexList[0], distanceList[0]

    def getPredictedIntention(self, text):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertForSequenceClassification.from_pretrained(os.path.join(os.getcwd(), 'assets', 'checkpoint-25630'))

        # 모델을 평가 모드로 전환
        model.eval()

        # 라벨 매핑 정보
        label_mapping = {
            0: "예방",
            1: "원인",
            2: "증상",
            3: "진단",
            4: "치료"
        }

        # 입력된 질문을 토큰화
        inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt')

        # 예측 수행
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]

        # 예측 결과 반환
        return label_mapping[prediction]

    def predict_intention(self, text):
        system_message = "당신은 사용자의 질문을 입력받고 사용자의 질문 의도가 무엇인지 파악하여 예방, 원인, 증상, 진단, 치료 중 하나의 단어로만 대답합니다."

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=100,
            temperature=0.2,
        )

        return response['choices'][0]['message']['content'].strip()

    def openAiBasedChangeTone(self, foundAnswerSeries, intention, type):
        predicted_disease_name = foundAnswerSeries['disease_name_kor']
        predicted_disease_info = " ".join(foundAnswerSeries[["answer_intro", "answer_body", "answer_conclusion"]].values)
        predicted_disease_answer = predicted_disease_name + "이 의심됩니다. " + predicted_disease_info

        messages = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": f"문체 변환 <{intention}> <{type}>  <{predicted_disease_answer}>"}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=700,
            temperature=0.7,
        )

        return response['choices'][0]['message']['content'].replace('\n', '')

    def getAnswerFeatures(self, foundAnswerSeries):
        features = []
        wannaGetFeatures = ['disease_category', 'disease_name_kor', 'department']

        for feat in wannaGetFeatures:
            if feat == 'department':
                features.append(foundAnswerSeries['department'].strip("[]'"))
            else:
                features.extend(foundAnswerSeries[[feat]])

        return features