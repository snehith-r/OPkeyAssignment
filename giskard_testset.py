from giskard import rag
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import uuid
import pickle
from tqdm import tqdm
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "chatgpt-4o-latest"



import pandas as pd

DATA_PATH = os.path.join("data", "dog_breeds.csv")

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


knowledge_base = rag.KnowledgeBase(df)


testset = rag.generate_testset(
    knowledge_base,
    num_questions=100,
    agent_description="A chatbot answering Questions requiring semantic comprehension and contextual understanding, Queries needing numerical analysis and precise data manipulation",
)

test_set_df = testset.to_pandas()

for index, row in enumerate(test_set_df.head(3).iterrows()):
    print(f"Question {index + 1}: {row[1]['question']}")
    print(f"Reference answer: {row[1]['reference_answer']}")
    print("Reference context:")
    print(row[1]['reference_context'])
    print("******************", end="\n\n") 

testset.save("test-set.jsonl")
