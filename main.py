# Load model directly
from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask_cors import CORS
import time
import torch
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)


tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

temperature = 1  # For creativity. 1.0 is the default. Lower for more deterministic.
max_length = 50  # Maximum number of tokens to generate.

emotion_list = ["happy", "sad", "neutral", "angry", "disgust", "excitement", "fear", "concern"]
encoded_list = [tokenizer(i)['input_ids'][1] for i in emotion_list]



utext = '''user: I have something happy to tell you
assistant: Oh great what's the amazing news
user: Someone just died'''

input_text = f'''You will classify the next emotion of the assistant from input conversation as one the following: [{", ".join(emotion_list)}]:

Here are few examples to illustrate only the format:

Input: I had a great day
Output: happy

Input: A bear is coming after me
Output: fear

Input: {utext}
Output: '''
input_ids = tokenizer.encode(input_text, return_tensors='pt')
token_to_add = torch.tensor([[199]])  # Shape [1, 1] to match the dimension of input_ids

# Append the token to the end of input_ids
input_ids = torch.cat((input_ids, token_to_add), dim=1)  # Concatenate along the second dimension



def normalize_probs_and_map(exp_probs):

    probs_list = exp_probs.tolist()
    flat_probs_list = [item for sublist in probs_list for item in sublist]

    # Calculate the sum of the probabilities
    sum_probs = sum(flat_probs_list)

    # Normalize each probability by dividing by the sum of all probabilities
    normalized_probs_list = [prob / sum_probs for prob in flat_probs_list]

    # Map each normalized probability to the corresponding emotion
    emotion_prob_map = {emotion: prob for emotion, prob in zip(emotion_list, normalized_probs_list)}

    return emotion_prob_map



model.to("cuda")



@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello, World!'


@app.route('/classify', methods=['POST'])
def classify_audio():
    request_data = request.get_json()
    messages = request_data['messages']
    message_strings = [
        f"{message['role']}: {message['content']}" for message in messages]

    # Join all the message strings into a single string, separated by a newline
    all_messages = "\n".join(message_strings)

    print(all_messages)
    return all_messages


if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
