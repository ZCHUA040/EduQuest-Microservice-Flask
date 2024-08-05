import os
from flask import Flask, request, jsonify
from azure_blob import AzureBlob
from dotenv import load_dotenv
load_dotenv()
from llm import LLM
from langchain_core.messages import HumanMessage, SystemMessage

app = Flask(__name__)

azure_blob = AzureBlob(
    connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
    container_name=os.getenv("AZURE_STORAGE_CONTAINER_NAME")
)
llm = LLM(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=float(os.getenv("AZURE_OPENAI_TEMPERATURE")),
)

# messages = [
#     SystemMessage(content="Translate the following from English into Italian"),
#     HumanMessage(content="hi!"),
# ]
#
# test_content = ("Databases are structured collections of data that allow for efficient storage, retrieval, "
#                 "and manipulation of information. They are integral to modern computing and are used in a wide "
#                 "range of applications, from simple personal record-keeping to complex enterprise-level systems. "
#                 "At their core, databases are designed to manage large volumes of data in a way that ensures "
#                 "consistency, accuracy, and security. They use a structured format, typically tables, to organize "
#                 "data into rows and columns, which makes it easy to perform queries and generate reports. Each "
#                 "table represents a specific entity, such as customers, products, or transactions, and each row "
#                 "in a table represents a single record. There are various types of databases, with relational "
#                 "databases (like MySQL, PostgreSQL, and Oracle) being the most common. Relational databases use "
#                 "Structured Query Language (SQL) for defining and manipulating data. They are known for their "
#                 "robust support for complex queries, transactions, and integrity constraints. In recent years, "
#                 "non-relational (NoSQL) databases, such as MongoDB, Cassandra, and Redis, have gained popularity. "
#                 "These databases are designed to handle unstructured data and provide flexibility in terms of "
#                 "data models. They are often used in big data and real-time web applications due to their "
#                 "scalability and performance. Overall, databases are foundational to the operation of software "
#                 "applications, enabling efficient data management, high performance, and scalability.")
#
# questions = llm.generate_questions_and_answers(
#     document_content=test_content,
#     num_questions=5,
#     difficulty="medium"
# )
#
# print(questions)

@app.route('/generate_questions_from_document', methods=['POST'])
def generate_questions_from_document():
    # Get the document content
    document_content = azure_blob.retrieve_document(
        document_id=f"documents/{request.json['document_id']}"
    )

    # Generate questions and answers
    questions = llm.generate_questions_and_answers(
        document_content=document_content,
        num_questions=request.json['num_questions'],
        difficulty=request.json['difficulty']
    )

    return jsonify(questions)


if __name__ == '__main__':

    app.run(debug=True)
