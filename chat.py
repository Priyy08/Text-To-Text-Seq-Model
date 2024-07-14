import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the tokenizer
with open('ML_Token.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the model
model = load_model('text_generation_model6.h5')

# Function to generate text
def generate_text(seed_text, next_words, model, tokenizer, max_sequence_len):
    original_seed_text = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        output_word = tokenizer.index_word.get(predicted_word_index, '')
        seed_text += " " + output_word
    generated_text = seed_text[len(original_seed_text):].strip()
    return generated_text

# Get the max_sequence_len
max_sequence_len = model.input_shape[1] + 1

# Function to generate a response
def generate_response(prompt, model, tokenizer):
    next_words = 50
    generated_text = generate_text(prompt, next_words, model, tokenizer, max_sequence_len)
    return [generated_text]

# Chat function
def chat_with_model():
    print("Welcome to the chat! Type 'exit' to end the conversation.")
    while True:
        prompt = input("You: ")
        if prompt.lower() == 'exit':
            print("Goodbye!")
            break
        responses = generate_response(prompt, model, tokenizer)
        for i, response in enumerate(responses):
            print(f"Model: {response}")

# Start the chat
chat_with_model()
