import os
import random
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from azure_blob import AzureBlob
from dotenv import load_dotenv

load_dotenv()
from llm import LLM

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

azure_blob = AzureBlob(
    connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
    container_name=os.getenv("AZURE_STORAGE_CONTAINER_NAME")
)
llm = LLM(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=float(os.getenv("AZURE_OPENAI_TEMPERATURE")),
)


@app.route('/generate_questions_from_document', methods=['POST'])
@cross_origin()
def generate_questions_from_document():
    try:
        # Get the document content
        document_content = azure_blob.retrieve_document(
            document_id=f"documents/{request.json['document_id']}"
        )
    except Exception as e:
        return jsonify({"error retrieving document": str(e)}), 404

    try:
        # Generate questions and answers
        questions = llm.generate_questions_and_answers(
            document_content=document_content,
            num_questions=request.json['num_questions'],
            difficulty=request.json['difficulty']
        )
    except Exception as e:
        return jsonify({"error generating questions": str(e)}), 500

    return jsonify(questions)


@app.route('/status', methods=['GET'])
@cross_origin()
def status():
    return jsonify({"status": "API is running"})

@app.route('/generate_feedback', methods=['POST'])
@cross_origin()
def generate_feedback():
    """
    Generate personalized feedback for a quest attempt
    """
    try:
        attempt_data = request.json
        print("[Attempt Data]", attempt_data, flush=True)
        feedback = llm.generate_personalised_feedback(attempt_data)
        print("[Generated Feedback]", feedback, flush=True)
        return jsonify(feedback)
    except Exception as e:
        return jsonify({"error generating feedback": str(e)}), 500


@app.route('/generate_bonus_game', methods=['POST'])
@cross_origin()
def generate_bonus_game():
    try:
        document_id = request.json['document_id']
    except Exception as e:
        print(f"[Bonus Game] Missing document_id: {str(e)}", flush=True)
        return jsonify({"error": f"Missing document_id: {str(e)}"}), 400

    try:
        document_content = azure_blob.retrieve_document(
            document_id=f"documents/{document_id}"
        )
    except Exception as e:
        print(f"[Bonus Game] Error retrieving document: {str(e)}", flush=True)
        return jsonify({"error retrieving document": str(e)}), 404

    try:
        game_type = random.choice(["matching", "ordering"])
        game = llm.generate_bonus_game(document_content=document_content, game_type=game_type)
    except Exception as e:
        print(f"[Bonus Game] Error generating bonus game: {str(e)}", flush=True)
        return jsonify({"error generating bonus game": str(e)}), 500

    return jsonify(game)



if __name__ == '__main__':
    port = int(os.getenv("PORT", "5000"))
    app.run(host='0.0.0.0', port=port, debug=True)
