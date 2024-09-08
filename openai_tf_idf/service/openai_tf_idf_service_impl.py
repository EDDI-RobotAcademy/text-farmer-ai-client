import time

from openai_tf_idf.repository.openai_tf_idf_repository_impl import OpenAITfIdfRepositoryImpl
from openai_tf_idf.service.openai_tf_idf_service import OpenAITfIdfService


class OpenAITfIdfServiceImpl(OpenAITfIdfService):
    __instance = None
    TOP_K = 1

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__openAITfIdfRepository = OpenAITfIdfRepositoryImpl.getInstance()

        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance

    async def textSimilarityAnalysis(self, text, type):
        try:
            start_time = time.time()

            # intention = self.__openAITfIdfRepository.predict_intention(text)
            intention = self.__openAITfIdfRepository.getPredictedIntention(text)
            end_time = time.time()
            print(f"*** Time of Getting {intention} Intention : {end_time - start_time}")

            faissIndex = self.__openAITfIdfRepository.getFaissIndex(intention)

            originalAnswers = self.__openAITfIdfRepository.getOriginalAnswer(intention)
            openAIEmbedding = self.__openAITfIdfRepository.openAiBasedEmbedding(text)
            indexList, distanceList = self.__openAITfIdfRepository.similarityAnalysis(
                openAIEmbedding, faissIndex, self.TOP_K, len(originalAnswers))

            foundAnswerSeries = originalAnswers.iloc[indexList[0]]
            modifiedAnswer = self.__openAITfIdfRepository.openAiBasedChangeTone(foundAnswerSeries, intention, type)
            getAnswerFeatures = self.__openAITfIdfRepository.getAnswerFeatures(foundAnswerSeries)

            end_time = time.time()

            print(f"*** Total Time : {end_time - start_time}")
            return {"output": modifiedAnswer, "features": getAnswerFeatures}

            # output = r"""<p>약물 알레르기의 발생 원인은 다음과 같은 다양한 요인에 의해 결정됩니다:</p> <ul style="list-style-position:inside;"> <li>약물의 성분</li> <li>화학 구조</li> <li>투여 경로</li> <li>치료 기간</li> </ul> <p>이러한 요인들은 알레르기 반응을 유발할 수 있는 위험 요소로 작용합니다. 일반적으로 알레르기 환자들은 특정 약물과 연관된 알레르기 반응을 경험할 수 있으며, 이는 약물의 성분, 화학 구조, 투여 경로, 그리고 치료 기간에 따라 다양한 방식으로 나타날 수 있습니다.</p> <p>예를 들어, 항생제는 바이러스나 기생충 감염에 대한 알레르기 반응을 유발할 수 있으며, 면역 체계의 약화로 인해 알레르기 반응이 과도하게 작동하여 약물 알레르기가 발생할 가능성이 있습니다. 또한, 항히스타민제를 비염약이나 알레르기 비염약과 함께 복용할 경우 증상이 악화될 수 있으며, 피부 알레르기도 약물 알레르기의 원인 중 하나입니다.</p> <p>결론적으로, 약물 알레르기의 발생 원인은 다양한 요인과 개인의 민감성에 따라 달라질 수 있습니다.</p>"""
            # return {"output": output,
            #         "features": ["a", "b", "c"]}

        except Exception as e:
            print(f"[ERROR]: {e}")
            return {"output": "아프신 곳을 좀 더 정확하게 표현해주세요."}

