import os
import pandas as pd
from utils.elasticsearch_utils import get_embedding

from langchain.agents import Tool, AgentType, initialize_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_elasticsearch import ElasticsearchRetriever
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


es_url = "http://elasticsearch:9200"

INDEX_NAME = "dog_breeds_index"

DATA_PATH = os.path.join("data", "dog_breeds.csv")
try:
    df = pd.read_csv(DATA_PATH)

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

except FileNotFoundError:
    df = pd.DataFrame()

# 4) Pandas DataFrame tool for analytics
llm_for_pandas = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")
pandas_agent = create_pandas_dataframe_agent(llm_for_pandas, df, verbose=False,allow_dangerous_code=True)


pandas_tool = Tool(
    name="DataFrameAnalysis",
    func=pandas_agent.run,
    description="Use this to answer numeric/analytical queries about the dog breeds (popularity, height, weight, etc.)."
)

def bm25_body_func(query: str) -> dict:
    """
    Build a request body for an Elasticsearch text-based (BM25) search
    across breed, description, and temperament fields.
    
    - 'breed' is a keyword field: only exact matches will be found.
    - 'description' and 'temperament' are text fields (tokenized).
    - BM25 is the default scoring mechanism in Elasticsearch.
    """
    return {
        "query": {
            "multi_match": {
                "query": query,
                # Boost 'breed' so exact name matches rank highly
                # and also give slight boost to 'description'
                "fields": [
                    "breed^5",        # if it exactly matches breed, give a big boost
                    "description^2",
                    "temperament",
                    "group",
                    "trainability_category",
                    "demeanor_category"
                ],
                "type": "best_fields"
            }
        }
    }

def hybrid_search_body_func(query: str) -> dict:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    vector = get_embedding(MODEL_NAME, query)

    return {
        "size": 10,
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "breed",
                                "description",
                                "temperament",
                                "group",
                                "grooming_frequency_category",
                                "shedding_category",
                                "energy_level_category",
                                "trainability_category",
                                "demeanor_category",
                            ],
                            "type": "best_fields"
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        },
        "knn": {
            "field": "embeddings",
            "query_vector": vector,
            "k": 5,
            "num_candidates": 7
        }
    }

es_retriever = ElasticsearchRetriever.from_es_params(
    index_name=INDEX_NAME,
    body_func=hybrid_search_body_func,
    content_field='description',
    url=es_url,
)



# Wrap the retriever in a RetrievalQA chain
from langchain.prompts import PromptTemplate

CUSTOM_TEMPLATE = """
You are an AI assistant for dog breed knowledge. Use the following pieces of context to answer the question.

{context}

Question: {question}
Answer:
"""

CUSTOM_PROMPT = PromptTemplate(
    template=CUSTOM_TEMPLATE,
    input_variables=["context", "question"],
)




retrieval_llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")
retrieval_qa_chain = RetrievalQA.from_chain_type(
    llm=retrieval_llm,
    retriever=es_retriever,
    chain_type="stuff",  # or "map_reduce", etc.
    chain_type_kwargs={"prompt": CUSTOM_PROMPT}
)

def es_tool_func(query: str) -> str:
    return retrieval_qa_chain.run(query)

elastic_tool = Tool(
    name="ElasticsearchKnowledgeBase",
    func=es_tool_func,
    description=(
        "Use this tool for free-form or textual queries about dog breeds, "
        "like temperament or descriptive info. It searches an Elasticsearch index."
    )
)

agent_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
tools = [pandas_tool, elastic_tool]
agent = initialize_agent(
    tools=tools,
    llm=agent_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def action(query):
    try:
        result = agent.run(query)
    except Exception as e:
        result = f"Error: {e}"
    
    return result
