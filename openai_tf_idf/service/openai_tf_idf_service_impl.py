import os
import pickle
import pandas as pd

import numpy as np

from openai_tf_idf.repository.openai_tf_idf_repository_impl import OpenAITfIdfRepositoryImpl
from openai_tf_idf.service.openai_tf_idf_service import OpenAITfIdfService


class OpenAITfIdfServiceImpl(OpenAITfIdfService):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__openAiTfIdfRepository = OpenAITfIdfRepositoryImpl.getInstance()

        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance

    def __init__(self):
        self.__openAiTfIdfRepository = OpenAITfIdfRepositoryImpl()
        self.embedding_pickle_path = os.path.join(
            os.getcwd(), "assets", "예방_Embedded_answers.pickle")
        self.original_data_path = os.path.join(
            os.getcwd(), "assets", "예방_original_answers.csv")

        # 임베딩된 데이터 로드
        try:
            with open(self.embedding_pickle_path, "rb") as file:
                self.embedded_answer = pickle.load(file)

            # FAISS 인덱스 생성 및 임베딩 데이터 추가
            embeddingVectorDimension = len(self.embedded_answer[0])
            self.faissIndex = self.__openAiTfIdfRepository.createL2FaissIndex(embeddingVectorDimension)
            self.faissIndex.add(np.array(self.embedded_answer).astype('float32'))

            # 원본 데이터를 CSV 파일에서 로드
            self.original_answers = pd.read_csv(self.original_data_path)

        except FileNotFoundError as e:
            print(f"File not found: {e}")
            raise e
        except Exception as e:
            print(f"An error occurred during initialization: {e}")
            raise e

    async def textSimilarityAnalysis(self, userQuestion, top_k=3):
        try:
            # 유사도 분석을 통해 인덱스 및 거리 리스트 가져오기
            indexList, distanceList = self.__openAiTfIdfRepository.similarityAnalysis(userQuestion, self.faissIndex, top_k)

            print(f"indexList: {indexList}, distanceList: {distanceList}")

            # 인덱스 유효성 확인 및 답변 가져오기
            if any(idx >= len(self.original_answers) for idx in indexList):
                raise ValueError(f"Invalid index found in indexList: {indexList}")

            best_answers = self.original_answers.iloc[indexList].to_dict(orient='records')

            return best_answers

        except Exception as e:
            print(f"An error occurred during text similarity analysis: {e}")
            raise e
