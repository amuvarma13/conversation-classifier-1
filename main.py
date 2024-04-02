# Load model directly
from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)


# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello, World!'

@app.route('/classify', methods=['POST'])
def classify_audio():
    request_data = request.get_json()
    messages = request_data['messages']
    message_strings = [f"{message['role']}: {message['content']}" for message in messages]

    # Join all the message strings into a single string, separated by a newline
    all_messages = "\n".join(message_strings)

    print(all_messages)
    return all_messages




if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')