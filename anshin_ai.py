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
    --bg:#fff0f5;
    --accent:#f4afc2;
    --deep:#d97fa3;
}

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
    color: var(--deep) !important;
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
    margin: 0.5rem 0 !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #ffc0cb !important;
    border: 2px solid #ff69b4;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #ffffff !important;
    border: 2px solid #f4afc2;
}

[data-testid="stChatMessageContent"] p {
    color: #000000 !important;
}

[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    width: 95%;
    max-width: 720px;
    background: #111111 !important;
    border: 2px solid #f4afc2 !important;
    border-radius: 25px !important;
    padding: 8px 12px !important;
}

[data-testid="stChatInput"] textarea {
    color: #ffffff !important;
    font-size: 16px !important;
    background: transparent !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #cccccc !important;
}

[data-testid="stChatInput"] button {
    background: #f4afc2 !important;
    border-radius: 50% !important;
}
</style>
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

def cbt_response(user_input, emotion):
    if emotion == "sadness":
        return f"""ごめんね、つらかったんだね。💗

You said: "{user_input}"

Let’s gently explore this together:

• What happened → It sounds like something didn’t go the way you hoped  
• How it made you feel → Maybe disappointed or heavy inside  
• Thought check → Does this situation truly define your ability or future?

Try this small step:
What is one thing you can still control right now?"""

    elif emotion == "fear":
        return f"""大丈夫だよ、ゆっくりでいいよ。🌿

You said: "{user_input}"

Let’s slow this down:

• What are you afraid might happen?  
• How likely is that outcome realistically?  
• If it *did* happen, how would you cope?

Small step:
Focus only on the next action, not the whole future."""

    elif emotion == "anger":
        return f"""その気持ちわかるよ。🌸

You said: "{user_input}"

Let’s understand this feeling:

• What triggered this anger?  
• What expectation was not met?  
• Is there another way to view the situation?

Try:
Pause for a moment — what would a calmer version of you say?"""

    elif emotion == "joy":
        return f"""それはいいね！✨

You said: "{user_input}"

Let’s anchor this feeling:

• What made this moment positive?  
• Can you recreate this feeling again?  

Hold onto this — it matters."""

    else:
        return f"""うん、聞いてるよ。💭

You said: "{user_input}"

Tell me more — what’s been on your mind lately?"""

for k,v in [("chat_history_ids",None),("messages",[]),("memory",[]),("show_graph",False)]:
    if k not in st.session_state:
        st.session_state[k]=v

st.title("🌸 Anshin AI · 安心 AI")
st.markdown('<div class="tagline">安心して話していいよ — You are safe here, take your time 💗</div>', unsafe_allow_html=True)

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

        final=cbt_response(user_input,emotion)

        for intent in intents.get('intents',[]):
            for pattern in intent.get("patterns",[]):
                if pattern.lower() in user_input.lower():
                    final=random.choice(intent["responses"])
                    break

        placeholder.markdown(final)
        st.session_state.messages.append(("Anshin AI",final))

    if "show graph" in user_input.lower():
        st.session_state.show_graph = True

    st.rerun()

if st.session_state.show_graph and st.session_state.memory:
    st.divider()
    st.subheader("📊 Emotion Journal")
    emotions=[m["emotion"] for m in st.session_state.memory]
    df=pd.DataFrame(list(Counter(emotions).items()),columns=["Emotion","Count"])
    st.bar_chart(df.set_index("Emotion"))
