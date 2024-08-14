from flask import Flask, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
from collections import defaultdict
from frozendict import frozendict
from logic.utils.agent import *


app = Flask(__name__)
CORS(app)

load_dotenv()  # Load environment variables from .env

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
API_KEY = os.getenv('API_KEY')

def failed_check_for_params(data, *params):
    for param in params:
        if param not in data:
            return f"Error: Missing '{param}' parameter", 400
    return None

@app.route('/', methods=['GET'])
def index():
    return 'Hello, World!'

cache = defaultdict(dict)

@app.route('/chat', methods=['POST'])
async def chat():
    """
    Handles incoming POST requests to the '/chat' endpoint.

    This function expects a JSON payload containing 'context', 'prompt', and 'agent' parameters.
    It validates the presence of these parameters and returns an error response if any are missing.

    If the parameters are valid, it creates an Agent instance from the provided 'agent' data,
    and uses it to generate a response to the given 'prompt' in the context of the provided 'context'.

    The function returns the response from the Agent as a JSON object.
    """
    data = request.get_json()
    if response := failed_check_for_params(data, 'context', 'prompt', 'agent'):
        return response
    agent = Agent.from_dict(data['agent'])
    context = data['context']
    prompt = data['prompt']
    if frozendict(context) in cache[agent.name] and prompt in cache[agent.name][frozendict(context)]: # check if prompt is already cached
        return cache[agent.name][frozendict(context)][prompt]
    response = await agent.async_chat(prompt, custom_context=[Interaction.from_dict(interaction) for interaction in context])
    cache[agent.name][frozendict(context)][prompt] = response
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
