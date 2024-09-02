from openai_tf_idf.repository.openai_tf_idf_repository_impl import OpenAITfIdfRepositoryImpl
from openai_tf_idf.service.openai_tf_idf_service import OpenAITfIdfService


class OpenAITfIdfServiceImpl(OpenAITfIdfService):
    __instance = None
    TOP_K = 3

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

    async def textSimilarityAnalysis(self, userQuestion):
        faissIndex = self.__openAITfIdfRepository.getFaissIndex(userQuestion["intention"])

        originalAnswers = self.__openAITfIdfRepository.getOriginalAnswer(userQuestion["intention"])
        openAIEmbedding = self.__openAITfIdfRepository.openAiBasedEmbedding(userQuestion["text"])
        indexList, distanceList = self.__openAITfIdfRepository.similarityAnalysis(
            openAIEmbedding, faissIndex, self.TOP_K, len(originalAnswers))
        foundAnswer = originalAnswers.iloc[indexList].to_dict()
        modifiedAnswer = self.__openAITfIdfRepository.openAiBasedChangeTone(foundAnswer, userQuestion["intention"], userQuestion["type"])

        return modifiedAnswer

