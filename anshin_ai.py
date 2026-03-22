import streamlit as st
import json
import pandas as pd
from collections import Counter
from transformers import pipeline
from openai import OpenAI

st.set_page_config(page_title="Anshin AI", page_icon="🌸", layout="centered")

client = OpenAI(api_key=st.secrets["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"])

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@400;500&family=Zen+Kaku+Gothic+New:wght@400;500&display=swap');

html, body {
    font-family:'Zen Kaku Gothic New',sans-serif !important;
    color:#111 !important;
}

.stApp {
    background: linear-gradient(180deg,#fff0f5 0%,#ffe4e1 100%);
}

.block-container {
    max-width:720px !important;
    padding:1rem 1rem 6rem !important;
}

h1 {
    text-align:center !important;
    color:#d97fa3 !important;
    font-family:'Noto Serif JP',serif !important;
}

.tagline {
    text-align:center;
    font-size:14px;
    color:#6b4c5c;
    margin-bottom:20px;
}

[data-testid="stChatMessage"] {
    border-radius:16px !important;
    padding:1rem !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background:#ffc0cb !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background:#ffffff !important;
}

[data-testid="stChatInput"] {
    position:fixed !important;
    bottom:20px;
    left:50%;
    transform:translateX(-50%);
    width:95%;
    max-width:720px;
    background:#111 !important;
    border-radius:25px !important;
}

[data-testid="stChatInput"] textarea {
    color:white !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )

emotion_model = load_emotion_model()

def detect_emotion(text):
    return emotion_model(text)[0]['label'].lower()

def detect_crisis(text):
    crisis_keywords = [
        "suicide", "kill myself", "end my life", "die", "no reason to live",
        "hopeless", "want to disappear"
    ]
    text = text.lower()
    return any(word in text for word in crisis_keywords)

def generate_response(user_input, emotion, history):
    convo = ""
    for h in history[-6:]:
        role = "User" if h["role"] == "user" else "Therapist"
        convo += f"{role}: {h['content']}\n"

    system_prompt = f"""
You are Anshin AI, a deeply empathetic therapist.

Personality:
- warm, calm, emotionally intelligent
- human-like, never robotic
- supportive but not overly dramatic

Therapy approach (CBT):
1. Validate emotion
2. Reflect situation
3. Identify thinking pattern
4. Gently reframe
5. Suggest ONE small action
6. Ask ONE thoughtful question

Emotion detected: {emotion}

Rules:
- NEVER repeat phrases
- NEVER give generic responses
- ALWAYS be specific
- Keep responses natural and flowing
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": convo + f"\nUser: {user_input}"}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content

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
        placeholder.markdown("🌸 *Anshin is listening…*")

        if detect_crisis(user_input):
            response = """ごめんね、今とてもつらい状態かもしれないね。

I'm really glad you shared this with me. You don't have to go through this alone.

If you can, please consider reaching out to someone right now:
- A trusted friend or family member
- A mental health professional
- A local helpline in your country

If you're in immediate danger, please contact emergency services.

You matter. I'm here with you. 💗

Do you want to tell me what’s making things feel this overwhelming?"""
        else:
            response = generate_response(user_input, emotion, st.session_state.history)

        placeholder.markdown(response)

    st.session_state.history.append({"role":"assistant","content":response})

    if "show graph" in user_input.lower():
        st.session_state.show_graph = True

    st.rerun()

if st.session_state.show_graph and st.session_state.memory:
    st.divider()
    st.subheader("📊 Emotion Journal")

    emotions = [m["emotion"] for m in st.session_state.memory]
    df = pd.DataFrame(list(Counter(emotions).items()), columns=["Emotion","Count"])
    st.bar_chart(df.set_index("Emotion"))
