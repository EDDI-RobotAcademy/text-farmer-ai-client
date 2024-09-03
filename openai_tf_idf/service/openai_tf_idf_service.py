from abc import ABC, abstractmethod


class OpenAITfIdfService(ABC):

    @abstractmethod
    def textSimilarityAnalysis(self, text, type):
        pass