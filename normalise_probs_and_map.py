from lists import emotion_list

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