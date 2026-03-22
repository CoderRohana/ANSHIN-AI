import streamlit as st
import torch
import json
import random
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from collections import Counter
import pandas as pd

st.set_page_config(page_title="Anshin AI", page_icon="🌸")

st.markdown("""
<style>
html, body, [class*="css"], p, span, div {
    color: #000000 !important;
    font-size: 18px !important;
    font-weight: 500;
}

.stApp {
    background: linear-gradient(180deg, #fff0f5 0%, #ffe4e1 100%);
}

[data-testid="stChatMessage"] {
    font-size: 18px !important;
}

[data-testid="stChatMessage"][data-testid*="user"] {
    background-color: #ffb6c1 !important;
    border-radius: 16px !important;
    padding: 14px !important;
    border: 2px solid #ff69b4;
}

[data-testid="stChatMessage"][data-testid*="assistant"] {
    background-color: #ffffff !important;
    border-radius: 16px !important;
    padding: 14px !important;
    border: 2px solid #ffc0cb;
}

[data-testid="stChatMessage"] * {
    color: #000000 !important;
}

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
<div class="sakura" style="left:30%; animation-duration:12s;">🌸</div>
<div class="sakura" style="left:50%; animation-duration:9s;">🌸</div>
<div class="sakura" style="left:70%; animation-duration:11s;">🌸</div>
<div class="sakura" style="left:90%; animation-duration:13s;">🌸</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    chat_model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(chat_model_name)
    model = AutoModelForCausalLM.from_pretrained(chat_model_name)

    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1
    )

    return tokenizer, model, emotion_model

tokenizer, model, emotion_model = load_models()

with open('intents.json', 'r') as f:
    intents = json.load(f)

def detect_emotion(text):
    result = emotion_model(text)[0][0]
    return result['label'].lower()

def therapist_response(user_input, base_response, emotion):
    if emotion in ["sad", "depressed"]:
        return f"I’m really sorry you’re feeling this way. It sounds like you're going through something heavy. 💗\n\nCan you tell me what’s been on your mind?"

    elif emotion in ["anxious", "fear"]:
        return f"It sounds like you're feeling anxious. Let’s slow things down together. 🌿\n\nTry taking a deep breath. What’s worrying you the most right now?"

    elif emotion == "angry":
        return f"I hear that you're feeling frustrated. That’s completely valid. 🌸\n\nDo you want to talk about what triggered this feeling?"

    elif emotion == "happy":
        return f"That’s really nice to hear! ✨\n\nWhat made you feel this way today?"

    else:
        return base_response

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

        base_response = tokenizer.decode(
            st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )

        final_response = therapist_response(user_input, base_response, emotion)

        thinking.markdown(final_response)

    st.session_state.messages.append(("Anshin AI", final_response))
    st.rerun()

if st.session_state.memory:
    st.divider()
    st.subheader("📊 Emotion Analytics")

    emotions = [m["emotion"] for m in st.session_state.memory]
    count = Counter(emotions)

    df = pd.DataFrame(list(count.items()), columns=["Emotion", "Count"])
    st.bar_chart(df.set_index("Emotion"))
