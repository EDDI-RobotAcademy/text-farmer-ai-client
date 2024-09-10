import os
import pickle
import urllib.parse

from dotenv import load_dotenv
from pymongo import MongoClient

from openai_tf_idf.repository.mongodb_repository import MongodbRepository

load_dotenv()

class MongodbRepositoryImpl(MongodbRepository):
    __instance = None
    DB_USER = os.getenv("MONGODB_USERNAME")
    DB_PASSWORD = os.getenv("MONGODB_PASSWORD")
    DB_DATABASE = os.getenv("MONGODB_DATABASE")
    DB_COLLECTION = "tfEmbeddingCollection"

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.client = MongoClient(
                f"mongodb://{cls.DB_USER}:{urllib.parse.quote(cls.DB_PASSWORD)}@localhost:27017")
            cls.__instance.db = cls.__instance.client[cls.DB_DATABASE]

        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance

    def loadIntentionPickle(self, intention):
        file_path = os.path.join(os.getcwd(), "assets", f"{intention}_Embedded_answers.pickle")

        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def getEmbeddings(self, intention):
        embeddingCollection = self.db[self.DB_COLLECTION]
        record = embeddingCollection.find_one({"intention": intention}, {"_id": 0, "embedding": 1})
        print(f"mongodbRepo -> record : {record}")

        if record:
            return record["embedding"]
        else:
            embeddingArray = self.loadIntentionPickle(intention)
            # try :
            #     embeddingCollection.insert_one({"intention": intention, "embedding": embeddingArray})
            # except Exception as e:
            #     print(f"MongoDB Insert Error : {e}")

            return embeddingArray