from fastapi import FastAPI, Body, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel
import os
from elasticsearch import Elasticsearch, exceptions
import time
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
import numpy as np
import uuid
from pathlib import Path
from elasticsearch.helpers import bulk
import pickle
from eland.ml.pytorch import PyTorchModel
from eland.ml.pytorch.transformers import TransformerModel
from eland.common import es_version

import pandas as pd
from langchain.chains import RetrievalQA


from utils.agent import action

import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']



app = FastAPI()

# Connect to Elasticsearch
es = Elasticsearch("http://elasticsearch:9200")
es_cluster_version = es_version(es)

# if any(endpoint['inference_id'] == 'my-msmarco-minilm-model' for endpoint in es.inference.get().get('endpoints', [])):
#     print(f"Model exists")
# else:

#     tm = TransformerModel(  model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",task_type="text_similarity",es_version=es_cluster_version)

#     tmp_path = "models"
#     Path(tmp_path).mkdir(parents=True, exist_ok=True)
#     model_path, config, vocab_path = tm.save(tmp_path)

#     ptm = PyTorchModel(es, tm.elasticsearch_model_id())
#     ptm.import_model(model_path=model_path, config_path=None, vocab_path=vocab_path, config=config)

#     es.inference.put(
#         task_type="rerank",
#         inference_id="my-msmarco-minilm-model",
#         inference_config={
#             "service": "elasticsearch",
#             "service_settings": {
#                 "model_id":tm.elasticsearch_model_id() ,
#                 "num_allocations": 1,
#                 "num_threads": 1,
#             },
#         },
#     )


# Define the request model
class QueryRequest(BaseModel):
    query: str
    search_strategy: str  # User can choose between 'keyword', 'vector', 'hybrid'

# DATA_PATH = os.path.join("data", "dog_breeds.csv")
DATA_PATH = 'embedded_data.parquet'
try:
    df = pd.read_parquet(DATA_PATH)

    df.rename(columns={"Unnamed: 0": "breed"}, inplace=True)
    df[["min_expectancy", "max_expectancy"]] = df[["min_expectancy", "max_expectancy"]].fillna(0)
    df["lifespan"] = (df["min_expectancy"] + df["max_expectancy"]) / 2
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    df["popularity"] = df["popularity"].fillna(0).astype("int64")


    df = df.astype({
        "breed": "string",
        "description": "string",
        "temperament": "string",
        "popularity": "Int64",
        "min_height":  "Float64",
        "max_height":  "Float64",
        "min_weight":  "Float64",
        "max_weight":  "Float64",
        "min_expectancy": "Float64",
        "max_expectancy":  "Float64",
        "group": "string",
        "grooming_frequency_value": "Float64",
        "grooming_frequency_category": "string",
        "shedding_value":     "Float64",
        "shedding_category":  "string",
        "energy_level_value": "Float64",
        "energy_level_category": "string",
        "trainability_value": "Float64",
        "trainability_category": "string",
        "demeanor_value":     "Float64",
        "demeanor_category":  "string",
    })
    print(df.info())

except FileNotFoundError:
    df = pd.DataFrame()




# Create the Elasticsearch index with chunk and embedding support
def create_index_if_not_exists(index_name: str):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    # es.indices.create(
    #     index=index_name,
    #     body={
    #         "mappings": {
    #             "properties": {
    #                 "document_id": {"type": "keyword"},
    #                 "content": {"type": "text"},
    #                 "source": {"type": "text"},
    #                 "embedding": {
    #                     "type": "dense_vector",
    #                     "dims": 3072, 
    #                     "index": True,  # Ensure KNN search is possible on this field
    #                     "similarity": "cosine"
    #                 }
    #             }
    #         }
    #     }
    # )

    mapping = {
    "mappings": {
        "properties": {
            "breed":         {"type": "keyword"},
            "description":        {"type": "text"},
            "temperament":        {"type": "text"},
            "popularity":         {"type": "integer"},  # must be int in Python
            "min_height":         {"type": "float"},
            "max_height":         {"type": "float"},
            "min_weight":         {"type": "float"},
            "max_weight":         {"type": "float"},
            "min_expectancy":     {"type": "float"},
            "max_expectancy":     {"type": "float"},
            "group":              {"type": "keyword"},
            "grooming_frequency_value": {"type": "float"},
            "grooming_frequency_category": {"type": "keyword"},
            "shedding_value":     {"type": "float"},
            "shedding_category":  {"type": "keyword"},
            "energy_level_value": {"type": "float"},
            "energy_level_category": {"type": "keyword"},
            "trainability_value": {"type": "float"},
            "trainability_category": {"type": "keyword"},
            "demeanor_value":     {"type": "float"},
            "demeanor_category":  {"type": "keyword"},
            "embeddings":{
                            "type": "dense_vector",
                            "dims": 384, 
                            "index": True,  # Ensure KNN search is possible on this field
                            "similarity": "cosine"
                        }
            # "lifespan": {"type":"float"}
        }
    }
}

    es.indices.create(index=index_name, body=mapping)
    print(f"Index '{index_name}' created.")

def wait_for_elasticsearch():
    """Wait until Elasticsearch is available."""
    while True:
        try:
            if es.ping():
                print("Elasticsearch is available.")
                break
        except exceptions.ConnectionError:
            print("Waiting for Elasticsearch to be available...")
            time.sleep(5)

# def retrieve_documents(query, search_strategy):
#     """Helper function to retrieve documents based on the search strategy."""
    
#     # Ensure Elasticsearch is reachable
#     if not es.ping():
#         raise Exception("Elasticsearch is not reachable.")

#     # Select search strategy
#     search_strategy = search_strategy.lower()

#     if search_strategy == "keyword":
#         documents = keyword_search(query)
    
#     elif search_strategy == "vector":
#         query_embedding = get_embedding("text-embedding-3-large", query)
#         documents = vector_search(query_embedding)

#     elif search_strategy == "vector_rerank":
#         query_embedding = get_embedding("text-embedding-3-large", query)
#         documents = vector_search_with_reranking(query, query_embedding)

#     elif search_strategy == "hybrid":
#         query_embedding = get_embedding("text-embedding-3-large", query)
#         documents = hybrid_search(query, query_embedding)

#     elif search_strategy == "hybrid_rerank":
#         query_embedding = get_embedding("text-embedding-3-large", query)
#         documents = vector_search_with_reranking(query, query_embedding)
#         documents += hybrid_search(query, query_embedding)

#     elif search_strategy=="query_expansion":
#         augmented_queries = augment_multiple_query(query)
#         queries = [query] + augmented_queries
#         documents=[]
#         for q in queries:
#             if len(q) < 5:
#                 continue
#             query_embedding = get_embedding("text-embedding-3-large", q)

#             documents += vector_search_with_reranking(q,query_embedding,top_k=3)
#             documents += hybrid_search(q,query_embedding,top_k=6)

#         # return rerank(query,documents,model_name='cross-encoder/ms-marco-MiniLM-L-12-v2')

#     else:
#         raise ValueError("Invalid search strategy")

#     return documents

@app.on_event("startup")
def startup_event():
    """On startup, create the index and load documents into Elasticsearch."""
    wait_for_elasticsearch()

    index_name = "dog_breeds_index"  # Define the index name
    create_index_if_not_exists(index_name)

    


    # Index documents and chunks with embeddings
    # actions = []  # Initialize actions list for bulk indexing
    # for doc in documents:
    #     document_id = doc.metadata["document_id"]
    #     chunk = doc.page_content
        
    #     action = {
    #         "_index": index_name,
    #         "_id": f"{document_id}",
    #         "_source": {
    #             "document_id": document_id,
    #             "content": chunk,
    #             "source": doc.metadata["source"],
    #             "embedding": doc.metadata['embedding']  # Ensure embeddings are serialized
    #         }
    #     }
        
    #     actions.append(action)

    # Use the bulk API to index all documents at once
    df["lifespan"] = df["lifespan"].replace({0.0: None})

    for i, row in df.iterrows():
       
        
        doc = {
            "breed": row.get("breed", ""),
            "description": row.get("description", ""),
            "temperament": row.get("temperament", ""),
            # "popularity": row.get("popularity", 0),
            "min_height": row.get("min_height", 0),
            "max_height": row.get("max_height", 0),
            "min_weight": row.get("min_weight", 0),
            "max_weight": row.get("max_weight", 0),
            "min_expectancy": row.get("min_expectancy", 0),
            "max_expectancy": row.get("max_expectancy", 0),
            "group": row.get("group", ""),
            "grooming_frequency_value": row.get("grooming_frequency_value", 0),
            "grooming_frequency_category": row.get("grooming_frequency_category", ""),
            "shedding_value": row.get("shedding_value", 0),
            "shedding_category": row.get("shedding_category", ""),
            "energy_level_value": row.get("energy_level_value", 0),
            "energy_level_category": row.get("energy_level_category", ""),
            "trainability_value": row.get("trainability_value", 0),
            "trainability_category": row.get("trainability_category", ""),
            "demeanor_value": row.get("demeanor_value", 0),
            "demeanor_category": row.get("demeanor_category", ""),
            "embeddings": row.get("embeddings")
            # "lifespan": row.get("lifespan",-1)
            
        }
        try:
            es.index(index=index_name, document=doc)
        except:
            print(row)

    print(f"Ingested {len(df)} documents into {index_name}.")


@app.post("/ask_question/")
def ask_policy_question(request: QueryRequest):
    """Endpoint to handle the policy question."""
    query = request.query
    
    
    answer = action(query)
    return {
        "question": query,
        "answer": answer,
    }






    
    



    

    
    