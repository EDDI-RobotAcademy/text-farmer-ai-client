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

    async def textSimilarityAnalysis(self, text, intention, type):
        faissIndex = self.__openAITfIdfRepository.getFaissIndex(intention)

        originalAnswers = self.__openAITfIdfRepository.getOriginalAnswer(intention)
        openAIEmbedding = self.__openAITfIdfRepository.openAiBasedEmbedding(text)
        indexList, distanceList = self.__openAITfIdfRepository.similarityAnalysis(
            openAIEmbedding, faissIndex, self.TOP_K, len(originalAnswers))
        foundAnswer = originalAnswers.iloc[indexList[0]].to_dict()
        beforeModify = (
                foundAnswer.get("answer_intro") + " " +
                foundAnswer.get("answer_body") + " " +
                foundAnswer.get("answer_conclusion")
        )
        modifiedAnswer = self.__openAITfIdfRepository.openAiBasedChangeTone(beforeModify, intention, type)

        return modifiedAnswer

