from elasticsearch import Elasticsearch, exceptions
import torch
from sentence_transformers import SentenceTransformer
import os

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://elasticsearch:9200")
INDEX_NAME = os.getenv("ELASTICSEARCH_INDEX", "pdfs")

es = Elasticsearch(hosts=[ELASTICSEARCH_HOST])



def get_embedding(model_name:str,text: str) -> list:
    #MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Change this based on your needs
    model = SentenceTransformer(model_name)
    return model.encode(text, convert_to_numpy=True).tolist()
















