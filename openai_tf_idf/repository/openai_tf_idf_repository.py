from abc import ABC, abstractmethod


class OpenAITfIdfRepository(ABC):
    @abstractmethod
    def getFaissIndex(self):
        pass

    @abstractmethod
    def getOriginalAnswer(self):
        pass

    @abstractmethod
    def openAiBasedEmbedding(self, text):
        pass

    @abstractmethod
    def similarityAnalysis(self, openAIEmbedding, faissIndex, top_k, originalAnswersLength):
        pass

    @abstractmethod
    def openAiBasedChangeTone(self, input_text, mbti_type):
        pass