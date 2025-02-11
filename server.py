from flask import Flask, request, jsonify
from rag_engine import chat_with_pdf  # Import the chat function
import json

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query_pdf():
    data = request.get_json()  # Get JSON request body
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    response = chat_with_pdf(query)  # Get response from RAG engine
    return jsonify(json.loads(response))  # Return response as JSON

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)