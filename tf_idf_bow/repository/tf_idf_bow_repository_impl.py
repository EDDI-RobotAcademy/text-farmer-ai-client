import json
import os
import pickle
import time
import pandas

from sklearn.metrics.pairwise import cosine_similarity

from tf_idf_bow.repository.tf_idf_bow_repository import TfIdfBowRepository


class TfIdfBowRepositoryImpl(TfIdfBowRepository):
    __instance = None

    VECTORIZATION_FILE_PATH = os.path.join(
        os.getcwd(), "assets", "introAnswerVectorization.pickle"
    )
    RAW_ANSWERS_FILE_PATH = os.path.join(
        os.getcwd(), "assets", "answers_8cols.pickle"
    )
    TOP_RANK_LIMIT = 3
    SIMILARITY_THRESHOLD = 0.1

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)

        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance

    async def findSimilarText(self, userQuestion):
        stime = time.time()
        with open(self.VECTORIZATION_FILE_PATH, "rb") as pickleFile:
            countVectorizer = pickle.load(pickleFile)
            countMatrix = pickle.load(pickleFile)
            # answerList = pickle.load(pickleFile)

        with open(self.RAW_ANSWERS_FILE_PATH, "rb") as pickleFile:
            allAnswerData = pickle.load(pickleFile)

        userQuestionVector = countVectorizer.transform([userQuestion])
        cosineSimilarityList = cosine_similarity(userQuestionVector, countMatrix).flatten()
        similarIndexList = cosineSimilarityList.argsort()[-self.TOP_RANK_LIMIT:][::-1]
        similarTopAnswerDict = {f"id_{index}": allAnswerData.iloc[similar_index, :].to_dict()
                             for index, similar_index in enumerate(similarIndexList)
                             if cosineSimilarityList[similar_index] >= self.SIMILARITY_THRESHOLD}
        etime = time.time()

        # similarityValueList = [cosineSimilarityList[index]
        #                      for index in similarIndexList
        #                      if cosineSimilarityList[index] >= self.SIMILARITY_THRESHOLD]

        return similarTopAnswerDict # 딕셔너리로 반환
