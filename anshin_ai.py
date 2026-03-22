import streamlit as st
import torch
import json
import random
import nltk
from nltk.stem.porter import PorterStemmer
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

nltk.download('punkt')

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def detect_emotion(text):
    tokens = [stem(w) for w in tokenize(text)]
    emotions = {
        "happy": ["happy","good","great","fine","ok","cheerful"],
        "sad": ["sad","lonely","down","empty","depressed"],
        "angry": ["angry","hate","annoyed","frustrated"],
        "anxious": ["anxious","worried","scared","stress","nervous"]
    }
    scores = {}
    for emo, words in emotions.items():
        scores[emo] = sum(1 for t in tokens if t in words)
    if max(scores.values()) == 0:
        return "neutral"
    return max(scores, key=scores.get)

@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

with open('intents.json', 'r') as f:
    intents = json.load(f)

st.title("🌸 Anshin AI (安心AI)")
st.write("Your AI Mental Health Companion 💙")

if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = []

user_input = st.text_input("You:")

if user_input:
    emotion = detect_emotion(user_input)
    st.session_state.memory.append({"text": user_input, "emotion": emotion})
    st.session_state.messages.append(("You", f"{user_input} ({emotion})"))

    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    if st.session_state.chat_history_ids is not None:
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    st.session_state.chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    if emotion == "sad":
        response = "I'm really sorry you're feeling this way. " + response
    elif emotion == "anxious":
        response = "Take a deep breath. You're safe. " + response
    elif emotion == "angry":
        response = "It's okay to feel angry. " + response
    elif emotion == "happy":
        response = "That's wonderful to hear! " + response

    for intent in intents['intents']:
        for pattern in intent["patterns"]:
            if pattern.lower() in user_input.lower():
                response = random.choice(intent["responses"])
                break

    st.session_state.messages.append(("Anshin AI", response))

for sender, msg in st.session_state.messages:
    st.write(f"**{sender}:** {msg}")

if st.session_state.memory:
    st.subheader("🧠 Emotion Memory")
    emotions = [m["emotion"] for m in st.session_state.memory]
    st.write(Counter(emotions))
