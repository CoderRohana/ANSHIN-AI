import streamlit as st
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from collections import Counter
import pandas as pd

st.set_page_config(page_title="Anshin AI", page_icon="🌸", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@400;500&family=Zen+Kaku+Gothic+New:wght@400;500&display=swap');

html, body, p, span, div {
    color: #111111 !important;
    font-size: 17px !important;
}

.stApp {
    background: linear-gradient(180deg,#fff0f5 0%,#ffe4e1 100%);
}

.block-container {
    max-width: 720px !important;
    padding: 1rem 1rem 6rem !important;
}

h1 {
    text-align:center !important;
    color: #d97fa3 !important;
    font-family:'Noto Serif JP',serif !important;
}

.tagline {
    text-align:center;
    font-size:14px;
    color:#6b4c5c;
    margin-bottom:20px;
}

[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    padding: 1rem !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #ffc0cb !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #ffffff !important;
}

[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    width: 95%;
    max-width: 720px;
    background: #111111 !important;
    border-radius: 25px !important;
}

[data-testid="stChatInput"] textarea {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    return tokenizer, model, emotion_model

tokenizer, model, emotion_model = load_models()

def detect_emotion(text):
    return emotion_model(text)[0]['label'].lower()

def build_prompt(user_input, emotion, history):
    convo = ""
    for h in history[-5:]:
        role = "User" if h["role"] == "user" else "Therapist"
        convo += f"{role}: {h['content']}\n"

    prompt = f"""
You are a calm, empathetic therapist using CBT techniques.
Speak naturally like a human, not like a chatbot.
Do not repeat phrases.
Do not say generic things like "I don't understand".
Guide the user gently.

Conversation:
{convo}

User just said: "{user_input}"
Detected emotion: {emotion}

Respond with:
- empathy
- gentle reflection
- one guiding question
- optional CBT reframing

Therapist:
"""
    return prompt

def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=inputs.shape[1] + 120,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.85
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("Therapist:")[-1].strip()

for k,v in [("history",[]),("memory",[]),("show_graph",False)]:
    if k not in st.session_state:
        st.session_state[k]=v

st.title("🌸 Anshin AI · 安心 AI")
st.markdown('<div class="tagline">安心して話していいよ — You are safe here, take your time 💗</div>', unsafe_allow_html=True)

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("話してみてね… How are you feeling today? 💭")

if user_input:
    emotion = detect_emotion(user_input)

    st.session_state.memory.append({"text":user_input,"emotion":emotion})
    st.session_state.history.append({"role":"user","content":user_input})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🌸 *Anshin is thinking…*")

        prompt = build_prompt(user_input, emotion, st.session_state.history)
        response = generate_response(prompt)

        placeholder.markdown(response)

    st.session_state.history.append({"role":"assistant","content":response})

    if "show graph" in user_input.lower():
        st.session_state.show_graph = True

    st.rerun()

if st.session_state.show_graph and st.session_state.memory:
    st.divider()
    st.subheader("📊 Emotion Journal")
    emotions=[m["emotion"] for m in st.session_state.memory]
    df=pd.DataFrame(list(Counter(emotions).items()),columns=["Emotion","Count"])
    st.bar_chart(df.set_index("Emotion"))
