from abc import ABC, abstractmethod

class MongodbRepository(ABC):
    def getEmbeddings(self, intention):
        pass