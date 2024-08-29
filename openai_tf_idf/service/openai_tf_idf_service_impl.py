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

    def textSimilarityAnalysis(self, userQuestion):
        print(f"plzzzzzzzzzzzzzzzzzzzzzzz: {userQuestion}")



