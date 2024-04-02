from lists import emotion_list
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