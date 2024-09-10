from abc import ABC, abstractmethod


class OpenAITfIdfRepository(ABC):
    @abstractmethod
    def getFaissIndex(self, intention):
        pass

    @abstractmethod
    def getOriginalAnswer(self, intention):
        pass

    @abstractmethod
    def openAiBasedEmbedding(self, text):
        pass

    @abstractmethod
    def similarityAnalysis(self, openAIEmbedding, faissIndex, top_k, originalAnswersLength):
        pass

    @abstractmethod
    def openAiBasedChangeTone(self, text, intention, type):
        pass

    @abstractmethod
    def getPredictedIntention(self, text):
        pass

    @abstractmethod
    def predict_intention(self, text):
        pass

    @abstractmethod
    def getAnswerFeatures(self, foundAnswerSeries):
        pass