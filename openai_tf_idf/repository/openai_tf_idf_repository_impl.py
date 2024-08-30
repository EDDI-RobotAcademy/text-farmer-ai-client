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

        print(f"response: {response}")
        return response['data'][0]['embedding']


    def similarityAnalysis(self, openAIEmbedding, faissIndex, top_k, originalAnswersLength):
        embeddingUserQuestion = np.array(openAIEmbedding).astype('float32').reshape(1, -1)

        distanceList, indexList = faissIndex.search(embeddingUserQuestion, top_k)

        # 인덱스 유효성 확인 및 답변 가져오기
        # if any([idx >= originalAnswersLength for idx in indexList]):
        #     raise ValueError(f"Invalid index found in indexList: {indexList}")

        return indexList[0], distanceList[0]