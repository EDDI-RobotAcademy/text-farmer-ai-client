from abc import ABC, abstractmethod

class TfIdfBowService(ABC):
    @abstractmethod
    def findSimilarAnswerInfo(self, userQuestion):
        pass