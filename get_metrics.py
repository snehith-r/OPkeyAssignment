import argparse
import pandas as pd
from giskard import rag
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import uuid
import pickle
from tqdm import tqdm
import openai
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv
import webbrowser
import json
import requests
from giskard.rag import QATestset

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "chatgpt-4o-latest"

def main(args):
    
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

    # Load the test set
    testset = QATestset.load(args.testset_file)

    def ask_policy_question(query, search_strategy="hybrid"):
        url = "http://localhost:8000/ask_question/"
        headers = {"Content-Type": "application/json"}
        payload = {
            "query": query,
            "search_strategy": search_strategy
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            answer = response.json().get('answer', '')
            print(f"Returned answer: {answer}")
            return answer
        else:
            return {"error": f"Request failed with status code {response.status_code}"}

    from giskard.rag import evaluate
    report = evaluate(ask_policy_question, testset=testset, knowledge_base=knowledge_base)

    # Save the report to an HTML file
    report.to_html(args.report_file)

    # Open the HTML file in the default web browser
    print(f"Report saved to {args.report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDFs and evaluate Q&A with Giskard.")
    parser.add_argument("--testset_file", type=str, required=True, help="Path to the test set JSONL file test-set-o1-chat.jsonl or test-set-o1.jsonl.")
    parser.add_argument("--report_file", type=str, required=True, help="File path to save the evaluation report.")

    args = parser.parse_args()
    main(args)
