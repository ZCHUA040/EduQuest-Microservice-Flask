import os
import requests
from openai import AzureOpenAI
from flask import Flask, request, jsonify
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import *

app = Flask(__name__)

# Load the .env file
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Initialize Azure Blob Service Client
blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))

# Initialize Azure Cognitive Search Clients
search_service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
search_service_key = os.getenv("AZURE_SEARCH_SERVICE_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

index_client = SearchIndexClient(
    endpoint=search_service_endpoint,
    credential=AzureKeyCredential(search_service_key)
)
search_client = SearchClient(
    endpoint=search_service_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(search_service_key)
)

# response = openai_client.chat.completions.create(
#     model=deployment_name, # model = "deployment_name".
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
#         {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
#         {"role": "user", "content": "Do other Azure AI services support this too?"}
#     ]
# )
#
# print(response.choices[0].message.content)


@app.route('/create_index', methods=['POST'])
def create_index():
    index_schema = SearchIndex(
        name=index_name,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="student_id", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single)),
        ],
        vectorizer=Vectorizer(
            type="TextEmbeddingVectorizer",
            embedding_model="openai-embedding-ada-002",
        ),
    )
    index_client.create_index(index_schema)

    return jsonify({"message": "Index created successfully."})

@app.route('/download_blob', methods=['POST'])
def download_blob():
    data = request.get_json()
    container_name = data['container_name']
    blob_name = data['blob_name']

    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    blob_data = blob_client.download_blob().readall()
    blob_content = blob_data.decode('utf-8')

    return jsonify({"content": blob_content})


@app.route('/get_embeddings', methods=['POST'])
def get_embeddings():
    data = request.get_json()
    text = data['text']

    response = openai_client.embeddings.create(model="text-embedding-ada-002", input=text)
    embedding = response.data[0].embedding

    return jsonify({"embedding": embedding})


@app.route('/upload_document', methods=['POST'])
def upload_document():
    data = request.get_json()
    documents = [
        {
            "id": data['id'],
            "content": data['content'],
            "content_vector": data['content_vector']
        }
    ]
    search_client.upload_documents(documents)

    return jsonify({"message": "Document uploaded successfully."})


@app.route('/query_vector_store', methods=['POST'])
def query_vector_store():
    data = request.get_json()
    query = data['query']
    top_k = data.get('top_k', 5)

    query_vector_response = requests.post(f"{FLASK_API_URL}/get_embeddings", json={"text": query})
    query_vector = query_vector_response.json()["embedding"]
    results = search_client.search(search_text="*", vector=query_vector, vector_fields=["content_vector"], top=top_k)

    return jsonify({"results": [result["content"] for result in results]})

if __name__ == '__main__':
    app.run(debug=True)
