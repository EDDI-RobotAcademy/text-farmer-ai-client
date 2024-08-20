from tf_idf_bow.repository.tf_idf_bow_repository_impl import TfIdfBowRepositoryImpl
from tf_idf_bow.service.tf_idf_bow_service import TfIdfBowService


class TfIdfBowServiceImpl(TfIdfBowService):
    def __init__(self):
        self.__tfIdfBowRepository = TfIdfBowRepositoryImpl()

    def findSimilarAnswerInfo(self, userQuestion):
        print('****** findSimilarAnswerInfo **********')
        # similarAnswerList = self.__tfIdfBowRepository.findSimilarText(userQuestion)