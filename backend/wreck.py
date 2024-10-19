from flask import Flask, request, jsonify
import random

from backend.breaking import break_code_randomly

app = Flask(__name__)

@app.route('/wreck', methods=['POST'])
def wreck_code():
    try:
        code_file = request.files['code']
        code_content = code_file.read().decode('utf-8')

        broken_code = break_code_randomly(code_content)
        
        return jsonify({"broken_code": broken_code}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)