from abc import ABC, abstractmethod


class OpenAITfIdfRepository(ABC):
    @abstractmethod
    def openAiBasedEmbedding(self, text):
        pass

    @abstractmethod
    def createL2FaissIndex(self, embeddingVectorDimension):
        pass

    @abstractmethod
    def similarityAnalysis(self, userQuestion, faissIndex):
        pass
