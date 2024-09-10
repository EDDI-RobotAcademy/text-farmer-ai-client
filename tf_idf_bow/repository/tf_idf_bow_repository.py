from abc import ABC, abstractmethod

class TfIdfBowRepository(ABC):
    @abstractmethod
    def findSimilarText(self, userQuestion):
        pass
