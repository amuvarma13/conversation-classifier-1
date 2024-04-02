from llm import tokenizer
emotion_list = ["happy", "sad", "neutral", "angry", "disgust", "excitement", "fear", "concern"]
encoded_list = [tokenizer(i)['input_ids'][1] for i in emotion_list]