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
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@300;400;500&family=Zen+Kaku+Gothic+New:wght@300;400;500&display=swap');

:root {
    --sakura-blush:#f9d6df;
    --sakura-petal:#f4afc2;
    --sakura-deep:#d97fa3;
    --sakura-ink:#2a0f1a;
    --sakura-white:#fffafc;
    --sakura-stem:#6b4c5c;
}

html, body {
    margin:0;
    padding:0;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family:'Zen Kaku Gothic New',sans-serif !important;
}

.stApp {
    background:
        radial-gradient(ellipse at 15% 10%, rgba(249,214,223,0.6) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 90%, rgba(217,127,163,0.3) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(253,240,244,1) 0%, #fce8ef 100%);
    background-attachment: fixed;
}

.block-container {
    max-width:720px !important;
    padding:1rem 1rem 6rem !important;
}

/* MOBILE OPTIMIZATION */
@media (max-width: 768px) {
    .block-container {
        padding:1rem 0.7rem 6rem !important;
    }
    h1 {
        font-size:1.8rem !important;
    }
}

/* HEADER */
h1 {
    font-family:'Noto Serif JP',serif !important;
    font-size:2.2rem !important;
    color:var(--sakura-deep) !important;
    text-align:center !important;
}

[data-testid="stCaptionContainer"] p {
    text-align:center !important;
    color:var(--sakura-stem) !important;
}

/* CHAT */
[data-testid="stChatMessage"] {
    border-radius:18px !important;
    padding:1rem !important;
    margin:0.6rem 0 !important;
}

[data-testid="stChatMessageContent"] p {
    color:var(--sakura-ink) !important;
    font-size:1rem !important;
    line-height:1.6 !important;
}

/* USER */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background:rgba(244,175,194,0.6) !important;
    border:1px solid rgba(217,127,163,0.3) !important;
}

/* BOT */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background:rgba(255,250,252,0.95) !important;
    border:1px solid rgba(244,175,194,0.3) !important;
}

/* INPUT BOX */
[data-testid="stChatInput"] {
    position:fixed !important;
    bottom:10px;
    left:50%;
    transform:translateX(-50%);
    width:95%;
    max-width:720px;
    background:rgba(255,250,252,0.98) !important;
    border:2px solid #f4afc2 !important;
    border-radius:30px !important;
    padding:8px 12px !important;
    box-shadow:0 6px 20px rgba(0,0,0,0.1);
}

[data-testid="stChatInput"] textarea {
    font-size:16px !important;
    color:#2a0f1a !important;
    background:transparent !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color:#9c6b7d !important;
}

[data-testid="stChatInput"] button {
    background-color:#f4afc2 !important;
    border-radius:50% !important;
}

[data-testid="stChatInput"] button:hover {
    background-color:#d97fa3 !important;
}

/* FOOTER */
.footer {
    position:fixed;
    bottom:0;
    width:100%;
    text-align:center;
    font-size:12px;
    color:#6b4c5c;
    padding:5px;
}
</style>

<div class="footer">Made with 💗 by Rohan Dutta</div>
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
        "sadness":"ごめんね、つらかったんだね。💗\n\nI'm really sorry you're feeling this way. Tell me what’s been weighing on you.",
        "fear":"大丈夫だよ、ゆっくりでいいよ。🌿\n\nIt sounds like you're anxious. What’s worrying you right now?",
        "anger":"その気持ち、ちゃんとわかるよ。🌸\n\nI hear your frustration. What happened?",
        "joy":"それはいいね！✨\n\nThat’s wonderful. What made you feel this way?",
        "surprise":"そうなんだね 🌸\n\nHow did that make you feel?",
        "disgust":"それはつらいね。🌿\n\nDo you want to talk about it?"
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
    st.subheader("📊 Emotion Journal · 感情の記録")
    emotions=[m["emotion"] for m in st.session_state.memory]
    df=pd.DataFrame(list(Counter(emotions).items()),columns=["Emotion","Count"])
    st.bar_chart(df.set_index("Emotion"))
