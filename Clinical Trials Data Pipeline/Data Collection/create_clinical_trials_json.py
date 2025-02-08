import openai
import faiss
import numpy as np
import json
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

# Azure OpenAI Config
AZURE_OPENAI_ENDPOINT = "https://your-openai-endpoint.openai.azure.com/"
AZURE_OPENAI_API_KEY = "your-api-key"
AZURE_DEPLOYMENT_NAME_GPT4 = "your-gpt4-deployment"
AZURE_DEPLOYMENT_NAME_EMBEDDING = "your-embedding-deployment"

openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2023-05-15"
openai.api_key = AZURE_OPENAI_API_KEY

# FAISS Setup (Pre-built index)
FAISS_INDEX_PATH = "faiss_index.bin"
index = faiss.read_index(FAISS_INDEX_PATH)
documents = json.load(open("documents.json"))  # Load corresponding documents

def get_embedding(text):
    """Get embeddings from Azure OpenAI embedding model."""
    response = openai.Embedding.create(
        engine=AZURE_DEPLOYMENT_NAME_EMBEDDING,
        input=text
    )
    return np.array(response['data'][0]['embedding'])

def retrieve_documents(query, top_k=3):
    """Retrieve similar documents from FAISS."""
    query_embedding = get_embedding(query).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

def generate_response(query, retrieved_docs):
    """Use GPT-4 to generate a response based on retrieved documents."""
    context = "\n".join(retrieved_docs)
    prompt = f"Based on the following context, answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    response = openai.ChatCompletion.create(
        engine=AZURE_DEPLOYMENT_NAME_GPT4,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# Example Query
query = "What is the impact of AI in healthcare?"
retrieved_docs = retrieve_documents(query)
response = generate_response(query, retrieved_docs)

print("Response:", response)