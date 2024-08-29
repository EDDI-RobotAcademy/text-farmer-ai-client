import os
import json
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

    SIMILARITY_TOP_RANK = 3

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

    # def getFaissIndex(self):
    #     try:
    #         with open(os.path.join(os.getcwd(), "assets", "예방_Embedded_answers.pickle"),
    #                   "rb") as file:
    #             EMBEDDED_ANSWER = pickle.load(file)
    #
    #         # FAISS 인덱스 생성 및 임베딩 데이터 추가
    #         embeddingVectorDimension = len(EMBEDDED_ANSWER[0])
    #         self.faissIndex = self.__openAiTfIdfRepository.createL2FaissIndex(embeddingVectorDimension)
    #         self.faissIndex.add(np.array(EMBEDDED_ANSWER).astype('float32'))
    #
    #         # 원본 데이터를 CSV 파일에서 로드
    #         self.original_answers = pd.read_csv(self.original_data_path)
    #
    #     except FileNotFoundError as e:
    #         print(f"File not found: {e}")
    #         raise e
    #     except Exception as e:
    #         print(f"An error occurred during initialization: {e}")
    #         raise e


    def openAiBasedEmbedding(self, text):
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )

        print(f"response: {response}")
        return response['data'][0]['embedding']

    def createL2FaissIndex(self, embeddingVectorDimension):
        return faiss.IndexFlatL2(embeddingVectorDimension)

    def similarityAnalysis(self, userQuestion, faissIndex, top_k=None):
        embeddingUserQuestion = np.array(
            self.openAiBasedEmbedding(userQuestion)).astype('float32').reshape(1, -1)
        # top_k를 지정하지 않으면 기본값으로 SIMILARITY_TOP_RANK 사용
        if top_k is None:
            top_k = self.SIMILARITY_TOP_RANK
        distanceList, indexList = faissIndex.search(embeddingUserQuestion, top_k)

        return indexList[0], distanceList[0]



