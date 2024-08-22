from tf_idf_bow.repository.tf_idf_bow_repository_impl import TfIdfBowRepositoryImpl
from tf_idf_bow.service.tf_idf_bow_service import TfIdfBowService


class TfIdfBowServiceImpl(TfIdfBowService):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__tfIdfBowRepository = TfIdfBowRepositoryImpl.getInstance()

        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance

    def findSimilarAnswerInfo(self, userQuestion):
        return self.__tfIdfBowRepository.findSimilarText(userQuestion)