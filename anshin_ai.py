import streamlit as st
import torch
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from collections import Counter
import pandas as pd

st.set_page_config(page_title="Anshin AI", page_icon="🌸", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@400;500&family=Zen+Kaku+Gothic+New:wght@400;500&display=swap');

:root {
    --sakura-bg:#fff0f5;
    --sakura-accent:#f4afc2;
    --sakura-deep:#d97fa3;
    --text-main:#111111;
}

/* GLOBAL TEXT FIX */
html, body, p, span, div {
    color: var(--text-main) !important;
    font-size: 17px !important;
    line-height: 1.6 !important;
}

/* BACKGROUND */
.stApp {
    background: linear-gradient(180deg,#fff0f5 0%,#ffe4e1 100%);
}

/* CONTAINER FIX */
.block-container {
    max-width: 720px !important;
    padding: 1rem 1rem 7rem !important;
}

/* HEADER */
h1 {
    text-align:center !important;
    color: var(--sakura-deep) !important;
    font-family:'Noto Serif JP',serif !important;
}

[data-testid="stCaptionContainer"] p {
    text-align:center !important;
}

/* CHAT BUBBLES */
[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    padding: 1rem !important;
    margin: 0.5rem 0 !important;
}

/* USER */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #ffc0cb !important;
    border: 2px solid #ff69b4;
}

/* BOT */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #ffffff !important;
    border: 2px solid #f4afc2;
}

/* TEXT INSIDE CHAT */
[data-testid="stChatMessageContent"] p {
    color: #000000 !important;
    font-size: 17px !important;
}

/* INPUT BOX */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 45px;
    left: 50%;
    transform: translateX(-50%);
    width: 95%;
    max-width: 720px;
    background: white !important;
    border: 2px solid #f4afc2 !important;
    border-radius: 25px !important;
    padding: 8px 12px !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.15);
}

/* INPUT TEXT */
[data-testid="stChatInput"] textarea {
    font-size: 16px !important;
    color: black !important;
}

/* BUTTON */
[data-testid="stChatInput"] button {
    background: #f4afc2 !important;
    border-radius: 50% !important;
}

[data-testid="stChatInput"] button:hover {
    background: #d97fa3 !important;
}

/* FOOTER FIX */
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: rgba(255,255,255,0.8);
    backdrop-filter: blur(5px);
    text-align: center;
    font-size: 13px;
    padding: 8px;
    color: #444;
    border-top: 1px solid #f4afc2;
}
</style>

<div class="footer">🌸 Made with 💗 by Rohan Dutta</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    return tokenizer, model, emotion_model

tokenizer, model, emotion_model = load_models()

try:
    with open('intents.json','r',encoding='utf-8') as f:
        intents=json.load(f)
except:
    intents={"intents":[]}

def detect_emotion(text):
    return emotion_model(text)[0]['label'].lower()

def therapist_response(emotion, base):
    mapping={
        "sadness":"ごめんね、つらかったんだね。💗\n\nI'm really sorry you're feeling this way. Tell me what’s on your mind.",
        "fear":"大丈夫だよ、ゆっくりでいいよ。🌿\n\nYou seem anxious. What’s worrying you?",
        "anger":"その気持ちわかるよ。🌸\n\nI hear your frustration. What happened?",
        "joy":"それはいいね！✨\n\nThat’s wonderful. Tell me more!",
    }
    return mapping.get(emotion, base)

for k,v in [("chat_history_ids",None),("messages",[]),("memory",[])]:
    if k not in st.session_state:
        st.session_state[k]=v

st.title("🌸 Anshin AI · 安心 AI")
st.caption("安心して話していいよ — You can talk freely here 💗")

for sender,msg in st.session_state.messages:
    with st.chat_message("user" if sender=="You" else "assistant"):
        st.markdown(msg)

user_input=st.chat_input("話してみてね… How are you feeling today? 💭")

if user_input:
    emotion=detect_emotion(user_input)
    st.session_state.memory.append({"text":user_input,"emotion":emotion})
    st.session_state.messages.append(("You",f"{user_input} · *{emotion}*"))

    with st.chat_message("assistant"):
        placeholder=st.empty()
        placeholder.markdown("🌸 *Anshin is listening…*")

        new_ids=tokenizer.encode(user_input+tokenizer.eos_token,return_tensors='pt')
        bot_ids=torch.cat([st.session_state.chat_history_ids,new_ids],dim=-1) if st.session_state.chat_history_ids is not None else new_ids

        st.session_state.chat_history_ids=model.generate(
            bot_ids,
            max_length=300,
            pad_token_id=tokenizer.eos_token_id
        )

        base=tokenizer.decode(
            st.session_state.chat_history_ids[:,bot_ids.shape[-1]:][0],
            skip_special_tokens=True
        )

        final=therapist_response(emotion,base)

        for intent in intents.get('intents',[]):
            for pattern in intent.get("patterns",[]):
                if pattern.lower() in user_input.lower():
                    final=random.choice(intent["responses"])
                    break

        placeholder.markdown(final)
        st.session_state.messages.append(("Anshin AI",final))

    st.rerun()

if st.session_state.memory:
    st.divider()
    st.subheader("📊 Emotion Journal")
    emotions=[m["emotion"] for m in st.session_state.memory]
    df=pd.DataFrame(list(Counter(emotions).items()),columns=["Emotion","Count"])
    st.bar_chart(df.set_index("Emotion"))
