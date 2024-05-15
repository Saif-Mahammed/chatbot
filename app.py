import streamlit as st
import json
import torch
import random
from nltk_utils import bag_of_words, tokenize
from model import NeuralNet

# Define custom theme colors
custom_primary_color = "#3F72AF"
custom_secondary_color = "#DBE2EF"
custom_text_color = "#112D4E"
custom_user_color = "#F9F7F7"
custom_bot_color = "#F9F7F7"

# Apply custom theme
st.markdown(
    f"""
    <style>
    .reportview-container .main .block-container {{
        max-width: 800px;
        padding: 2em 3em;
    }}
    .sidebar .sidebar-content {{
        background-color: {custom_secondary_color};
        color: {custom_text_color};
    }}
    .stTextInput > div > div > div > input {{
        color: {custom_text_color};
    }}
    .stButton> button {{
        background-color: {custom_primary_color};
        color: {custom_secondary_color};
    }}
    .user-message {{
        background-color: {custom_user_color};
        color: {custom_text_color};
        border-radius: 15px;
        padding: 10px 15px;
        margin-bottom: 10px;
        max-width: 70%;
        align-self: flex-end;
    }}
    .bot-message {{
        background-color: {custom_bot_color};
        color: {custom_text_color};
        border-radius: 15px;
        padding: 10px 15px;
        margin-bottom: 10px;
        max-width: 70%;
        align-self: flex-start;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load intents file
with open('intents.json') as f:
    intents = json.load(f)

# Load the model and metadata
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Function to get response from the model
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).float()
    X = X.unsqueeze(0)  # Add a batch dimension
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Streamlit app
st.title("ChatGPT Chatbot")

# Chat history
chat_history = []

# User input
user_input = st.text_input("You:", "Say something...")

# Submit button
if st.button("Send"):
    if user_input.strip() != "":
        # Get response from the model
        response = get_response(user_input)
        # Add user input and response to chat history
        chat_history.append({"user": user_input, "chatgpt": response})
        # Clear user input
        user_input = ""

# Display chat history
for chat in chat_history:
    if chat['user'] != "":
        st.markdown(f'<div class="user-message">{chat["user"]}</div>', unsafe_allow_html=True)
    if chat['chatgpt'] != "":
        st.markdown(f'<div class="bot-message">{chat["chatgpt"]}</div>', unsafe_allow_html=True)
