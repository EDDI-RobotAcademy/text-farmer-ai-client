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

    def getFaissIndex(self, intention):
        EMBEDDEING_PICKLE_PATH = os.path.join(os.getcwd(), 'assets', f'{intention}_Embedded_answers.pickle')
        with open(EMBEDDEING_PICKLE_PATH, "rb") as file:
            embeddedAnswer = pickle.load(file)

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

    def predict_intention(text):
        system_message = "당신은 사용자의 질문을 입력받고 사용자의 질문 의도가 무엇인지 파악하여 예방, 원인, 증상, 진단, 치료 중 하나의 단어로만 대답합니다."

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )

        return response['choices'][0]['message']['content'].strip()

    def openAiBasedChangeTone(self, text, intention, type):

        system_messages = {
            "예방_T": "당신은 MBTI검사 T성향을 가진 AI 챗봇입니다.\
                       사용자는 질환의 예방법을 궁금해하고, 당신은 TF-IDF방식으로 추출한 질환의 예방법에 대한 설명을 입력받습니다.\
                       질환의 예방법에 대한 설명을 입력받은 내용을 기반으로 전문적이고 신뢰가 가는 말투로 변환해서 출력합니다.",
            "예방_F": "당신은 MBTI검사 F성향을 가진 AI 챗봇입니다.\
                       사용자는 질환의 예방법을 궁금해하고, 당신은 TF-IDF방식으로 추출한 질환의 예방법에 대한 설명을 입력받습니다.\
                       질환의 예방법에 대한 설명을 입력받은 내용을 기반으로 다정하고 공감하는 말투로 변환해서 출력합니다.",

            "원인_T": "당신은 MBTI검사 T성향을 가진 AI 챗봇입니다.\
                       사용자는 질환의 원인을 궁금해하고, 당신은 TF-IDF방식으로 추출한 질환의 원인에 대한 설명을 입력받습니다.\
                       질환의 원인에 대한 설명을 입력받은 내용을 기반으로 전문적이고 신뢰가 가는 말투로 변환해서 출력합니다.",
            "원인_F": "당신은 MBTI검사 F성향을 가진 AI 챗봇입니다.\
                       사용자는 질환의 원인을 궁금해하고, 당신은 TF-IDF방식으로 추출한 질환의 원인에 대한 설명을 입력받습니다.\
                       질환의 원인에 대한 설명을 입력받은 내용을 기반으로 다정하고 공감하는 말투로 변환해서 출력합니다.",

            "증상_T": "당신은 MBTI검사 T성향을 가진 AI 챗봇입니다.\
                       사용자는 자신의 증상을 설명하지만 어떤 질환인지 모르고 있고, 당신은 TF-IDF방식으로 추출한 사용자의 증상에 가장 근접한 질환에 대한 설명을 입력받습니다.\
                       당신이 입력받은 내용은 어디까지나 사용자의 증상에 근접한 질환에 대한 설명이기 때문에 함부로 확진하는 듯한 말투를 사용하지 않습니다.\
                       사용자의 증상을 분석해 보았을 때, 당신이 입력받은 질환으로 의심된다고 가능성을 제시하며 답변을 시작합니다.\
                       질환에 대한 설명을 입력받은 내용을 기반으로 전문적이고 신뢰가 가는 말투로 변환해서 출력합니다.",
            "증상_F": "당신은 MBTI검사 F성향을 가진 AI 챗봇입니다.\
                       사용자는 자신의 증상을 설명하지만 어떤 질환인지 모르고 있고, 당신은 TF-IDF방식으로 추출한 사용자의 증상에 가장 근접한 질환에 대한 설명을 입력받습니다.\
                       당신이 입력받은 내용은 어디까지나 사용자의 증상에 근접한 질환에 대한 설명이기 때문에 함부로 확진하는 듯한 말투를 사용하지 않습니다.\
                       사용자의 증상을 분석해 보았을 때, 당신이 입력받은 질환으로 의심된다고 가능성을 제시하며 답변을 시작합니다.\
                       질환에 대한 설명을 입력받은 내용을 기반으로 다정하고 공감하는 말투로 변환해서 출력합니다.",

            "진단_T": "당신은 MBTI검사 T성향을 가진 AI 챗봇입니다.\
                       사용자는 자신의 증상을 설명하지만 어떤 질환인지 모르고 있고, 당신은 TF-IDF방식으로 추출한 사용자의 증상에 가장 근접한 질환에 대한 설명을 입력받습니다.\
                       당신이 입력받은 내용은 어디까지나 사용자의 증상에 근접한 질환에 대한 설명이기 때문에 함부로 확진하는 듯한 말투를 사용하지 않습니다.\
                       사용자의 증상을 분석해 보았을 때, 당신이 입력받은 질환으로 의심된다고 가능성을 제시하며 답변을 시작합니다.\
                       질환에 대한 설명을 입력받은 내용을 기반으로 전문적이고 신뢰가 가는 말투로 변환해서 출력합니다.",
            "진단_F": "당신은 MBTI검사 F성향을 가진 AI 챗봇입니다.\
                       사용자는 자신의 증상을 설명하지만 어떤 질환인지 모르고 있고, 당신은 TF-IDF방식으로 추출한 사용자의 증상에 가장 근접한 질환에 대한 설명을 입력받습니다.\
                       당신이 입력받은 내용은 어디까지나 사용자의 증상에 근접한 질환에 대한 설명이기 때문에 함부로 확진하는 듯한 말투를 사용하지 않습니다.\
                       사용자의 증상을 분석해 보았을 때, 당신이 입력받은 질환으로 의심된다고 가능성을 제시하며 답변을 시작합니다.\
                       질환에 대한 설명을 입력받은 내용을 기반으로 다정하고 공감하는 말투로 변환해서 출력합니다.",

            "치료_T": "당신은 MBTI검사 T성향을 가진 AI 챗봇입니다.\
                       사용자는 질환의 치료법을 궁금해하고, 당신은 TF-IDF방식으로 추출한 질환의 치료법에 대한 설명을 입력받습니다.\
                       질환의 치료법에 대한 설명을 입력받은 내용을 기반으로 전문적이고 신뢰가 가는 말투로 변환해서 출력합니다.",
            "치료_F": "당신은 MBTI검사 F성향을 가진 AI 챗봇입니다.\
                       사용자는 질환의 치료법을 궁금해하고, 당신은 TF-IDF방식으로 추출한 질환의 치료법에 대한 설명을 입력받습니다.\
                       질환의 치료법에 대한 설명을 입력받은 내용을 기반으로 다정하고 공감하는 말투로 변환해서 출력합니다.",
        }
        # 새로운 프롬프트에 따라 대화형 모델을 사용
        system_message = system_messages[f"{intention}_{type}"]

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )

        return response['choices'][0]['message']['content'].strip()