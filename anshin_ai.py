import streamlit as st
import torch
import json
import random
import time
from nltk.stem.porter import PorterStemmer
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import pandas as pd

st.set_page_config(page_title="Anshin AI", page_icon="🌸")

st.markdown("""
<style>

/* GLOBAL TEXT FIX */
html, body, [class*="css"], p, span, div {
    color: #000000 !important;
    font-size: 18px !important;
    font-weight: 500;
}

/* BACKGROUND */
.stApp {
    background: linear-gradient(180deg, #fff0f5 0%, #ffe4e1 100%);
}

/* CHAT CONTAINER */
[data-testid="stChatMessage"] {
    font-size: 18px !important;
    color: #000000 !important;
}

/* USER BUBBLE */
[data-testid="stChatMessage"][data-testid*="user"] {
    background-color: #ffb6c1 !important;
    color: black !important;
    border-radius: 16px !important;
    padding: 14px !important;
    margin: 10px 0 !important;
    border: 2px solid #ff69b4;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}

/* BOT BUBBLE */
[data-testid="stChatMessage"][data-testid*="assistant"] {
    background-color: #ffffff !important;
    color: black !important;
    border-radius: 16px !important;
    padding: 14px !important;
    margin: 10px 0 !important;
    border: 2px solid #ffc0cb;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}

/* FORCE TEXT INSIDE BUBBLES */
[data-testid="stChatMessage"] * {
    color: #000000 !important;
    font-size: 18px !important;
}

/* INPUT BOX */
textarea, input {
    font-size: 18px !important;
    color: #000000 !important;
    background-color: #ffffff !important;
    border-radius: 10px !important;
}

/* HEADERS */
h1 {
    font-size: 36px !important;
    color: #000000 !important;
}
h2, h3 {
    font-size: 24px !important;
    color: #000000 !important;
}

/* SAKURA ANIMATION */
.sakura {
    position: fixed;
    top: -10px;
    font-size: 20px;
    animation: fall linear infinite;
    z-index: 9999;
}

@keyframes fall {
    to {
        transform: translateY(110vh) rotate(360deg);
    }
}

</style>

<div class="sakura" style="left:10%; animation-duration:10s;">🌸</div>
<div class="sakura" style="left:25%; animation-duration:12s;">🌸</div>
<div class="sakura" style="left:40%; animation-duration:9s;">🌸</div>
<div class="sakura" style="left:55%; animation-duration:11s;">🌸</div>
<div class="sakura" style="left:70%; animation-duration:13s;">🌸</div>
<div class="sakura" style="left:85%; animation-duration:10s;">🌸</div>

""", unsafe_allow_html=True)

stemmer = PorterStemmer()

def tokenize(sentence):
    return sentence.lower().split()

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
st.caption("安心して話していいよ 💗")

if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = []

for sender, msg in st.session_state.messages:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(msg)

user_input = st.chat_input("Talk to Anshin AI...")

if user_input:
    emotion = detect_emotion(user_input)
    st.session_state.memory.append({"text": user_input, "emotion": emotion})
    st.session_state.messages.append(("You", f"{user_input} ({emotion})"))

    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.markdown("💬 *Anshin is thinking...*")
        time.sleep(1)

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
            response = "I'm really sorry you're feeling this way. 💗 " + response
        elif emotion == "anxious":
            response = "Take a deep breath. You're safe. 🌿 " + response
        elif emotion == "angry":
            response = "It's okay to feel angry. 🌸 " + response
        elif emotion == "happy":
            response = "That's wonderful to hear! ✨ " + response

        for intent in intents['intents']:
            for pattern in intent["patterns"]:
                if pattern.lower() in user_input.lower():
                    response = random.choice(intent["responses"])
                    break

        thinking.markdown(response)

    st.session_state.messages.append(("Anshin AI", response))
    st.rerun()

if st.session_state.memory:
    st.divider()
    st.subheader("📊 Emotion Analytics")

    emotions = [m["emotion"] for m in st.session_state.memory]
    count = Counter(emotions)

    df = pd.DataFrame(list(count.items()), columns=["Emotion", "Count"])
    st.bar_chart(df.set_index("Emotion"))
