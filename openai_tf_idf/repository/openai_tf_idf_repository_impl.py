import os
import json, pickle
import pandas as pd

import faiss
import numpy as np
import openai

from dotenv import load_dotenv

from openai_tf_idf.repository.openai_tf_idf_repository import OpenAITfIdfRepository


load_dotenv()

openaiApiKey = os.getenv('OPENAI_API_KEY')
if not openaiApiKey:
    raise ValueError('API Key가 준비되어 있지 않습니다!')

openai.api_key = openaiApiKey

class OpenAITfIdfRepositoryImpl(OpenAITfIdfRepository):
    __instance = None

    EMBEDDEING_PICKLE_PATH=os.path.join(os.getcwd(), 'assets', '예방_Embedded_answers.pickle')
    ORIGINAL_DATA_PATH=os.path.join(os.getcwd(), 'assets', '예방_original_answers.csv')

    headers = {
        'Authorization': f'Bearer {openaiApiKey}',
        'Content-Type': 'application/json'
    }

    OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)

        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance

    def getFaissIndex(self):
        with open(self.EMBEDDEING_PICKLE_PATH, "rb") as file:
            embeddedAnswer = pickle.load(file)

        # FAISS 인덱스 생성 및 임베딩 데이터 추가
        embeddingVectorDimension = len(embeddedAnswer[0])
        faissIndex = faiss.IndexFlatL2(embeddingVectorDimension)
        faissIndex.add(np.array(embeddedAnswer).astype('float32'))

        return faissIndex

    def getOriginalAnswer(self):
        return pd.read_csv(self.ORIGINAL_DATA_PATH)

    def openAiBasedEmbedding(self, text):
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )

        return response['data'][0]['embedding']

    def openAiBasedChangeTone(self, input_text, mbti_type):
        # 새로운 프롬프트에 따라 대화형 모델을 사용
        if mbti_type == "T":
            system_message = "사용자의 질문과 가장 유사도가 높은 답변을 입력으로 받지만, 혹시나 이 답변이 정답이 아닐수도 있으니 확신이 아닌 가능성을 제시해주는 답변을 출력할거야. \
                                  MBTI T 성격 유형을 가진 의사가 환자에게 대답하는 것처럼 문제 해결과 실질적인 조언에 중점을 두어 신뢰가 가는 말투로 변환해줘"
        elif mbti_type == "F":
            system_message = "사용자의 질문과 가장 유사도가 높은 답변을 입력으로 받지만, 혹시나 이 답변이 정답이 아닐수도 있으니 확신이 아닌 가능성을 제시해주는 답변을 출력할거야. \
                                  단순히 이 데이터를 보여주는 게 아니라, 아픈 상황에 공감하고, 위로하는 멘트를 덧붙여서 답변으로 내보내고 싶어.\
                                  입력값의 내용이 들어가되, 다정하고 공감하는 말투로 변환해서 답변을 만들어줘"
        else:
            raise ValueError("잘못된 MBTI 유형입니다. 'T' 또는 'F'를 선택하세요.")

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_text}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )

        return response['choices'][0]['message']['content'].strip()



    def similarityAnalysis(self, openAIEmbedding, faissIndex, top_k, originalAnswersLength):
        embeddingUserQuestion = np.array(openAIEmbedding).astype('float32').reshape(1, -1)

        distanceList, indexList = faissIndex.search(embeddingUserQuestion, top_k)

        # 인덱스 유효성 확인 및 답변 가져오기
        # if any([idx >= originalAnswersLength for idx in indexList]):
        #     raise ValueError(f"Invalid index found in indexList: {indexList}")

        return indexList[0], distanceList[0]