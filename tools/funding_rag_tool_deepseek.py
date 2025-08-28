import os
import requests
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv
from chromadb import PersistentClient
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

load_dotenv()

# Initialize ChromaDB
chromadb = PersistentClient(path="./chroma_db_v2")
collection = chromadb.get_or_create_collection("funding_rag")

# Use DeepSeek Embedding API
def get_embedding(text: str, model="deepseek-embedding", dimensions=1024):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    url = f"{base_url}/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "input": text
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    return response.json()['data'][0]['embedding']

# Build Chroma vector store from CSVs
def build_vector_store(csv_paths: List[str]):
    if collection.count() > 0:
        print("Vector store already exists. Skipping.")
        return

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if 'Investment Areas' in df.columns and 'Funding Stage' in df.columns:
            df = df.dropna(subset=["Investment Areas", "Funding Stage"])

            for i, row in df.iterrows():
                doc_id = str(row.get("rowID", f"{csv_path}_{i}"))
                text = f"{row['Name']} invests in {row['Investment Areas']}, mainly in {row['Funding Stage']} stage."
                embedding = get_embedding(text)

                collection.add(
                    documents=[text],
                    ids=[doc_id],
                    embeddings=[embedding],
                    metadatas=[{
                        "name": row["Name"],
                        "link": row.get("Profile Link", ""),
                        "stage": row["Funding Stage"],
                        "areas": row["Investment Areas"],
                        "country": "UK" if "uk" in csv_path.lower() else "Germany"
                    }]
                )

    print(f"✅ Vector store build complete. Total entries: {collection.count()}")

# Tool input schema
class FundingQuery(BaseModel):
    industry: str = Field(description="Industry of the company")
    country: str = Field(description="Target country")
    stage: str = Field(description="Funding stage")
    description: str = Field(default="", description="Company description")

# Search funding function (main entry)
def search_funding(industry: str, country: str, stage: str, description: str = "") -> str:
    print("=" * 50)
    print("SearchFundingTool Invoked")
    print(f"industry: {industry}")
    print(f"country: {country}")
    print(f"stage: {stage}")
    print(f"description: {description}")
    print("=" * 50)

    if any(isinstance(p, str) and p.startswith("{") for p in [industry, country, stage, description]):
        return "❌ Error: Detected invalid parameter serialization. Please check agent inputs."

    try:
        query_text = f"{description} {industry} in {country} stage {stage}"

        if collection.count() == 0:
            print("⚠️ Vector store is empty. Returning mock results.")
            return generate_mock_results(industry, country, stage, description)

        query_embedding = get_embedding(query_text)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["metadatas", "distances", "documents"]
        )

        funds = []
        for meta, dist, doc in zip(results['metadatas'][0], results['distances'][0], results['documents'][0]):
            score = 1 - dist
            funds.append({
                "name": meta['name'],
                "link": meta['link'],
                "stage": meta['stage'],
                "areas": meta['areas'],
                "country": meta.get('country', 'Unknown'),
                "score": round(score, 3),
                "description": doc
            })

        top_funds = sorted(funds, key=lambda x: -x['score'])[:3]

        if not top_funds:
            return generate_mock_results(industry, country, stage, description)

        output = f"## Top Investor Matches for {industry} companies in {country} at {stage} stage\n\n"
        for i, fund in enumerate(top_funds, 1):
            output += f"### {i}. {fund['name']}\n"
            output += f"**Match Score**: {fund['score']:.1%}\n"
            output += f"**Funding Stage**: {fund['stage']}\n"
            output += f"**Investment Areas**: {fund['areas']}\n"
            output += f"**Country**: {fund['country']}\n"
            output += f"**Why Recommended**: {generate_recommendation_reason(fund, industry, stage, description)}\n"
            if fund['link']:
                output += f"**Website**: {fund['link']}\n"
            output += "\n"

        return output

    except Exception as e:
        return f"❌ Error during search: {str(e)}"

# Fallback if vector store is empty
def generate_mock_results(industry: str, country: str, stage: str, description: str) -> str:
    mock_funds = [{
        "name": f"{country} Tech Ventures",
        "stage": stage,
        "areas": f"{industry}, Technology",
        "country": country,
        "link": "https://example-vc.com"
    }]

    output = f"## Mock Recommendations for {industry} companies in {country} ({stage} stage)\n\n"
    output += "*Note: These are demo results. Please connect real investor data for live use.*\n\n"

    for i, fund in enumerate(mock_funds, 1):
        output += f"### {i}. {fund['name']}\n"
        output += f"**Funding Stage**: {fund['stage']}\n"
        output += f"**Investment Areas**: {fund['areas']}\n"
        output += f"**Country**: {fund['country']}\n"
        output += f"**Website**: {fund['link']}\n\n"

    return output

# Reason generation
def generate_recommendation_reason(fund: Dict, industry: str, stage: str, description: str) -> str:
    reasons = []
    if stage.lower() in fund['stage'].lower():
        reasons.append(f"Focuses on {stage} stage funding")
    if industry.lower() in fund['areas'].lower():
        reasons.append(f"Experienced in {industry} sector")
    if "oxford" in description.lower() or "cambridge" in description.lower():
        reasons.append("Likely supports elite university spinouts")
    if not reasons:
        reasons.append("Aligns with your business and funding goals")
    return "; ".join(reasons[:2])

# Register LangChain Tool
SearchFundingTool = StructuredTool.from_function(
    name="SearchFundingTool",
    description="Search for recommended funding institutions and investors",
    func=search_funding,
    args_schema=FundingQuery,
    return_direct=True
)
