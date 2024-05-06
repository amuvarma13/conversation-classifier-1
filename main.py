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

temperature = 1  
max_length = 50  

emotion_list = ["happy", "sad", "neutral", "angry", "disgust", "excitement", "fear", "concern"]
encoded_list = [tokenizer(i)['input_ids'][1] for i in emotion_list]






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
    utext = "\n".join(message_strings)


    input_text = f'''You will classify the emotion of the assistant from input conversation as one the following: [{", ".join(emotion_list)}]:

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


    num_generated_tokens = 1  # Parameter to control the number of tokens to generate
    temperature = 1  # Temperature for scaling logits

    new_input_ids = input_ids.to("cuda").clone()  # Start with the original input

    with torch.no_grad():
        for _ in range(num_generated_tokens):
            outputs_l = model(new_input_ids)
            logits = outputs_l.logits
            # Scale logits by temperature
            scaled_logits = logits / temperature
            selected_token_ids = encoded_list
            selected_logits = scaled_logits[:, :, selected_token_ids]
            probabilities = F.softmax(scaled_logits, dim=-1)
            next_token_logits = probabilities[:, -1, :]
            most_probable_token_id = torch.argmax(next_token_logits, dim=-1).item()
            print("most_probable_token_id", most_probable_token_id, tokenizer.decode(most_probable_token_id))
            selected_probabilities = probabilities[:, :, selected_token_ids]
            exp_probabilities = selected_probabilities[:, -1, :]
            next_token_logits = probabilities[:, -1, :]
            # Select the most probable next token ID
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            # Append the generated token ID to the input for the next iteration
            new_input_ids = torch.cat((new_input_ids, next_token_id), dim=1)

    # Decode the generated tokens to text, excluding the original input
    # Calculate the starting point to exclude the original input from decoding
    start_point = input_ids.shape[1]
    print(new_input_ids[0, start_point:])
    # decoded_tokens = tokenizer.decode(new_input_ids[0, start_point:], skip_special_tokens=True)

    emotion_prob_map = normalize_probs_and_map(exp_probabilities)

    return {
        # "decoded_tokens": decoded_tokens,
        "emotion_prob_map": emotion_prob_map
    
    }




if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
