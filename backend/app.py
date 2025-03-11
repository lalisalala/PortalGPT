from flask import Flask, request, jsonify
from utils.llama_processing import generate_response

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    response_text = generate_response(user_query)
    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(debug=True)
