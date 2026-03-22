import streamlit as st
import torch
import json
import random
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from collections import Counter
import pandas as pd

st.set_page_config(page_title="Anshin AI", page_icon="🌸", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@300;400;500&family=Zen+Kaku+Gothic+New:wght@300;400;500&display=swap');

:root {
    --sakura-blush:  #f9d6df;
    --sakura-petal:  #f4afc2;
    --sakura-deep:   #d97fa3;
    --sakura-ink:    #3b1f2b;
    --sakura-mist:   #fdf0f4;
    --sakura-white:  #fffafc;
    --sakura-stem:   #7a5c6e;
}

html, body {
    font-family: 'Zen Kaku Gothic New', sans-serif !important;
}
p, span, label, h1, h2, h3, h4, li, .stMarkdown {
    font-family: 'Zen Kaku Gothic New', sans-serif !important;
    color: var(--sakura-ink) !important;
}

.stApp {
    background:
        radial-gradient(ellipse at 15% 10%, rgba(249,214,223,0.6) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 90%, rgba(217,127,163,0.3) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(253,240,244,1) 0%, #fce8ef 100%);
    background-attachment: fixed;
    min-height: 100vh;
}

.block-container {
    max-width: 780px !important;
    padding: 2rem 2rem 5rem !important;
}

h1 {
    font-family: 'Noto Serif JP', serif !important;
    font-size: 2.3rem !important;
    font-weight: 300 !important;
    color: var(--sakura-deep) !important;
    letter-spacing: 0.08em !important;
    text-align: center !important;
    text-shadow: 0 2px 20px rgba(217,127,163,0.2);
    margin-bottom: 0.2rem !important;
}

[data-testid="stCaptionContainer"] p {
    text-align: center !important;
    color: var(--sakura-stem) !important;
    font-size: 0.93rem !important;
    letter-spacing: 0.1em !important;
}

[data-testid="stChatMessage"] {
    border-radius: 20px !important;
    padding: 1rem 1.3rem !important;
    margin: 0.6rem 0 !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    animation: fadeInUp 0.35s ease both;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(244,175,194,0.5), rgba(249,214,223,0.6)) !important;
    border: 1px solid rgba(217,127,163,0.3) !important;
    box-shadow: 0 4px 20px rgba(217,127,163,0.12), inset 0 1px 0 rgba(255,255,255,0.5) !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: linear-gradient(135deg, rgba(255,250,252,0.92), rgba(253,240,244,0.85)) !important;
    border: 1px solid rgba(244,175,194,0.35) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,0.9) !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span {
    color: var(--sakura-ink) !important;
    font-size: 1rem !important;
    line-height: 1.78 !important;
    font-family: 'Zen Kaku Gothic New', sans-serif !important;
}

[data-testid="stChatInput"] {
    border-radius: 24px !important;
    border: 1.5px solid rgba(217,127,163,0.4) !important;
    background: rgba(255,250,252,0.95) !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 4px 20px rgba(217,127,163,0.12), inset 0 1px 0 rgba(255,255,255,0.8) !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
    overflow: hidden !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--sakura-deep) !important;
    box-shadow: 0 4px 28px rgba(217,127,163,0.28), inset 0 1px 0 rgba(255,255,255,0.9) !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    font-family: 'Zen Kaku Gothic New', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    caret-color: var(--sakura-deep) !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(122,92,110,0.5) !important;
    font-style: italic;
}

[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #f4afc2, #d97fa3) !important;
    border: none !important;
    border-radius: 50% !important;
    box-shadow: 0 2px 10px rgba(217,127,163,0.4) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"] button:hover {
    transform: scale(1.1) !important;
    box-shadow: 0 4px 16px rgba(217,127,163,0.55) !important;
}
[data-testid="stChatInput"] button svg {
    stroke: white !important;
    fill: white !important;
}

hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(217,127,163,0.4), transparent) !important;
    margin: 1.5rem 0 !important;
}

h2, h3 {
    font-family: 'Noto Serif JP', serif !important;
    font-weight: 400 !important;
    color: var(--sakura-deep) !important;
    letter-spacing: 0.06em !important;
}

[data-testid="stVegaLiteChart"] {
    border-radius: 16px !important;
    background: rgba(255,250,252,0.75) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(244,175,194,0.3) !important;
    padding: 1rem !important;
    box-shadow: 0 4px 20px rgba(217,127,163,0.1) !important;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(217,127,163,0.35); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: rgba(217,127,163,0.6); }

.petal-wrap {
    position: fixed;
    inset: 0;
    pointer-events: none;
    overflow: hidden;
    z-index: 9999;
}
.petal {
    position: absolute;
    top: -60px;
    opacity: 0;
    animation: petalFall linear infinite;
    will-change: transform, opacity;
}
@keyframes petalFall {
    0%   { transform: translateY(0) translateX(0) rotate(0deg); opacity: 0; }
    5%   { opacity: 0.82; }
    90%  { opacity: 0.65; }
    100% { transform: translateY(105vh) translateX(var(--drift)) rotate(var(--spin)); opacity: 0; }
}

.kamon {
    position: fixed;
    bottom: 1.5rem;
    right: 1.5rem;
    width: 72px;
    height: 72px;
    opacity: 0.07;
    pointer-events: none;
    z-index: 0;
}
</style>

<div class="petal-wrap" id="petals"></div>
<div class="kamon">
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="50" cy="50" r="46" fill="none" stroke="#d97fa3" stroke-width="2"/>
  <g transform="translate(50,50)">
    <ellipse rx="8" ry="20" fill="#d97fa3" transform="rotate(0)   translate(0,-22)"/>
    <ellipse rx="8" ry="20" fill="#d97fa3" transform="rotate(72)  translate(0,-22)"/>
    <ellipse rx="8" ry="20" fill="#d97fa3" transform="rotate(144) translate(0,-22)"/>
    <ellipse rx="8" ry="20" fill="#d97fa3" transform="rotate(216) translate(0,-22)"/>
    <ellipse rx="8" ry="20" fill="#d97fa3" transform="rotate(288) translate(0,-22)"/>
    <circle r="9" fill="#fce8ef"/>
  </g>
</svg>
</div>

<script>
(function() {
    var wrap = document.getElementById('petals');
    if (!wrap) return;
    var colors = ['#f9c9d8','#f4afc2','#fce4ec','#f8bbd0','#ffe0ea','#fadadd'];
    for (var i = 0; i < 20; i++) {
        var el = document.createElement('div');
        el.className = 'petal';
        var size  = 11 + Math.random() * 13;
        var left  = Math.random() * 100;
        var dur   = 10 + Math.random() * 12;
        var delay = Math.random() * 18;
        var drift = ((Math.random() - 0.5) * 160).toFixed(1);
        var spin  = (200 + Math.random() * 400).toFixed(1);
        var color = colors[Math.floor(Math.random() * colors.length)];
        var rot   = (Math.random() * 360).toFixed(1);
        el.style.cssText = 'left:'+left+'%;width:'+size+'px;height:'+size+'px;animation-duration:'+dur+'s;animation-delay:-'+delay+'s;--drift:'+drift+'px;--spin:'+spin+'deg;';
        el.innerHTML = '<svg viewBox="0 0 40 40" width="'+size+'" height="'+size+'" xmlns="http://www.w3.org/2000/svg"><g transform="rotate('+rot+' 20 20)"><ellipse cx="20" cy="20" rx="11" ry="19" fill="'+color+'" opacity="0.88" transform="rotate(0 20 20)"/><ellipse cx="20" cy="20" rx="11" ry="19" fill="'+color+'" opacity="0.60" transform="rotate(72 20 20)"/><ellipse cx="20" cy="20" rx="11" ry="19" fill="'+color+'" opacity="0.50" transform="rotate(144 20 20)"/><ellipse cx="20" cy="20" rx="11" ry="19" fill="'+color+'" opacity="0.50" transform="rotate(216 20 20)"/><ellipse cx="20" cy="20" rx="11" ry="19" fill="'+color+'" opacity="0.60" transform="rotate(288 20 20)"/><circle cx="20" cy="20" r="3.5" fill="rgba(255,230,240,0.95)"/></g></svg>';
        wrap.appendChild(el);
    }
})();
</script>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1
    )
    return tokenizer, model, emotion_model

tokenizer, model, emotion_model = load_models()

try:
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
except FileNotFoundError:
    intents = {"intents": []}

def detect_emotion(text):
    result = emotion_model(text)[0][0]
    return result['label'].lower()

def therapist_response(user_input, base_response, emotion):
    responses = {
        "sadness": (
            "ごめんね、つらかったんだね。💗\n\n"
            "I'm really sorry you're feeling this way. It sounds like you're carrying something heavy.\n\n"
            "Can you tell me more about what's been on your mind?"
        ),
        "fear": (
            "大丈夫だよ、ゆっくり深呼吸して。🌿\n\n"
            "It sounds like you're feeling anxious. Let's slow things down together.\n\n"
            "Try taking one deep breath. What's worrying you the most right now?"
        ),
        "anger": (
            "怒るのは当然だよ。🌸\n\n"
            "I hear that you're frustrated — that's completely valid.\n\n"
            "Do you want to talk about what triggered this feeling?"
        ),
        "joy": (
            "それは嬉しいね！✨\n\n"
            "That's really lovely to hear!\n\n"
            "What made you feel this way today?"
        ),
        "surprise": (
            "そうだったんだね！🌸\n\n"
            "Oh, that's unexpected! How are you feeling about it?"
        ),
        "disgust": (
            "それは嫌だったね. 🌿\n\n"
            "That sounds really uncomfortable. Would you like to share what happened?"
        ),
    }
    return responses.get(emotion, base_response)

for key, default in [
    ("chat_history_ids", None),
    ("messages", []),
    ("memory", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🌸 Anshin AI · 安心 AI")
st.caption("安心して話していいよ　— You can talk freely here 💗")

for sender, msg in st.session_state.messages:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(msg)

user_input = st.chat_input("今日はどんな気持ち？  ·  How are you feeling today…")

if user_input:
    emotion = detect_emotion(user_input)
    st.session_state.memory.append({"text": user_input, "emotion": emotion})

    with st.chat_message("user"):
        st.markdown(f"{user_input}  ·  *{emotion}*")
    st.session_state.messages.append(("You", f"{user_input}  ·  *{emotion}*"))

    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.markdown("🌸 *Anshin is listening…*")
        
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = (
            torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1)
            if st.session_state.chat_history_ids is not None
            else new_input_ids
        )

        st.session_state.chat_history_ids = model.generate(
            bot_input_ids, 
            max_length=1000, 
            pad_token_id=tokenizer.eos_token_id
        )

        decoded_response = tokenizer.decode(
            st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )

        final_response = therapist_response(user_input, decoded_response, emotion)

        matched_intent = False
        for intent in intents.get('intents', []):
            for pattern in intent.get("patterns", []):
                if pattern.lower() in user_input.lower():
                    final_response = random.choice(intent["responses"])
                    matched_intent = True
                    break
            if matched_intent: break

        thinking.markdown(final_response)
        st.session_state.messages.append(("Anshin AI", final_response))
    
    st.rerun()

if st.session_state.memory:
    st.divider()
    st.subheader("📊 Emotion Journal · 感情の記録")
    emotions = [m["emotion"] for m in st.session_state.memory]
    count = Counter(emotions)
    df = pd.DataFrame(list(count.items()), columns=["Emotion", "Count"])
    st.bar_chart(df.set_index("Emotion"))
