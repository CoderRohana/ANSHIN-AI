import streamlit as st
import torch
import json
import random
import time
from nltk.stem.porter import PorterStemmer
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import pandas as pd

st.set_page_config(page_title="Anshin AI", page_icon="🌸", layout="centered")

st.markdown("""
<style>

/* ─── GOOGLE FONTS ─── */
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@300;400;500&family=Zen+Kaku+Gothic+New:wght@300;400;500&display=swap');

/* ─── CSS VARIABLES ─── */
:root {
    --sakura-blush:    #f9d6df;
    --sakura-petal:    #f4afc2;
    --sakura-deep:     #d97fa3;
    --sakura-ink:      #3b1f2b;
    --sakura-mist:     #fdf0f4;
    --sakura-white:    #fffafc;
    --sakura-gold:     #c9a87c;
    --sakura-stem:     #7a5c6e;
    --blur-bg:         rgba(255,250,252,0.72);
}

/* ─── GLOBAL RESET ─── */
html, body, [class*="css"] {
    font-family: 'Zen Kaku Gothic New', sans-serif !important;
    color: var(--sakura-ink) !important;
}

/* ─── APP BACKGROUND ─── */
.stApp {
    background:
        radial-gradient(ellipse at 15% 10%, rgba(249,214,223,0.55) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 90%, rgba(217,127,163,0.3) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(253,240,244,1) 0%, #fce8ef 100%);
    background-attachment: fixed;
    min-height: 100vh;
}

/* ─── NOISE TEXTURE OVERLAY ─── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* ─── BLOCK CONTAINER ─── */
.block-container {
    max-width: 780px !important;
    padding: 2rem 2rem 5rem !important;
    position: relative;
    z-index: 1;
}

/* ─── TITLE ─── */
h1 {
    font-family: 'Noto Serif JP', serif !important;
    font-size: 2.4rem !important;
    font-weight: 300 !important;
    color: var(--sakura-deep) !important;
    letter-spacing: 0.08em !important;
    text-align: center !important;
    margin-bottom: 0.1rem !important;
    text-shadow: 0 2px 20px rgba(217,127,163,0.25);
}

/* ─── CAPTION ─── */
.stApp [data-testid="stCaptionContainer"] p,
.stApp .stMarkdown p:has(+ *) {
    text-align: center !important;
    color: var(--sakura-stem) !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.12em !important;
}

/* ─── ALL TEXT ─── */
p, span, div, label {
    color: var(--sakura-ink) !important;
    font-size: 1rem !important;
}

/* ─── CHAT MESSAGES WRAPPER ─── */
[data-testid="stChatMessage"] {
    border-radius: 20px !important;
    padding: 1rem 1.3rem !important;
    margin: 0.6rem 0 !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    animation: fadeInUp 0.35s ease both;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ─── USER BUBBLE ─── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(244,175,194,0.55), rgba(249,214,223,0.65)) !important;
    border: 1px solid rgba(217,127,163,0.35) !important;
    box-shadow: 0 4px 24px rgba(217,127,163,0.15), inset 0 1px 0 rgba(255,255,255,0.6) !important;
}

/* ─── BOT BUBBLE ─── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: linear-gradient(135deg, rgba(255,250,252,0.85), rgba(253,240,244,0.8)) !important;
    border: 1px solid rgba(244,175,194,0.4) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06), inset 0 1px 0 rgba(255,255,255,0.9) !important;
}

/* ─── AVATAR ICONS ─── */
[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, #f4afc2, #d97fa3) !important;
    border-radius: 50% !important;
    box-shadow: 0 2px 8px rgba(217,127,163,0.4) !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #fce8ef, #f4afc2) !important;
    border-radius: 50% !important;
    box-shadow: 0 2px 8px rgba(217,127,163,0.3) !important;
}

/* ─── CHAT TEXT ─── */
[data-testid="stChatMessage"] * {
    color: var(--sakura-ink) !important;
    font-size: 1rem !important;
    line-height: 1.7 !important;
}

/* ─── CHAT INPUT ─── */
[data-testid="stChatInput"] {
    border-radius: 24px !important;
    overflow: hidden !important;
    border: 1.5px solid rgba(217,127,163,0.4) !important;
    background: rgba(255,250,252,0.85) !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 4px 20px rgba(217,127,163,0.15), inset 0 1px 0 rgba(255,255,255,0.8) !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--sakura-deep) !important;
    box-shadow: 0 4px 28px rgba(217,127,163,0.3), inset 0 1px 0 rgba(255,255,255,0.9) !important;
}

textarea, [data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--sakura-ink) !important;
    font-family: 'Zen Kaku Gothic New', sans-serif !important;
    font-size: 1rem !important;
    border: none !important;
    outline: none !important;
    caret-color: var(--sakura-deep) !important;
}

/* ─── SEND BUTTON ─── */
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #f4afc2, #d97fa3) !important;
    border: none !important;
    border-radius: 50% !important;
    color: white !important;
    box-shadow: 0 2px 10px rgba(217,127,163,0.4) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

[data-testid="stChatInput"] button:hover {
    transform: scale(1.08) !important;
    box-shadow: 0 4px 16px rgba(217,127,163,0.55) !important;
}

/* ─── DIVIDER ─── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(217,127,163,0.4), transparent) !important;
    margin: 1.5rem 0 !important;
}

/* ─── SUBHEADER ─── */
h2, h3 {
    font-family: 'Noto Serif JP', serif !important;
    font-weight: 400 !important;
    font-size: 1.2rem !important;
    color: var(--sakura-deep) !important;
    letter-spacing: 0.06em !important;
}

/* ─── BAR CHART ─── */
[data-testid="stVegaLiteChart"] {
    border-radius: 16px !important;
    overflow: hidden !important;
    background: rgba(255,250,252,0.7) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(244,175,194,0.3) !important;
    padding: 1rem !important;
    box-shadow: 0 4px 20px rgba(217,127,163,0.1) !important;
}

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(217,127,163,0.35);
    border-radius: 99px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(217,127,163,0.6);
}

/* ─── SAKURA PETALS ─── */
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

.petal svg {
    display: block;
}

@keyframes petalFall {
    0%   { transform: translateY(0) translateX(0) rotate(0deg);   opacity: 0; }
    5%   { opacity: 0.85; }
    90%  { opacity: 0.7; }
    100% { transform: translateY(105vh) translateX(var(--drift)) rotate(var(--spin)); opacity: 0; }
}

/* ─── KAMON WATERMARK ─── */
.kamon {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    width: 80px;
    height: 80px;
    opacity: 0.07;
    pointer-events: none;
    z-index: 0;
}

</style>

<!-- SAKURA PETALS -->
<div class="petal-wrap" id="petals"></div>

<!-- KAMON WATERMARK -->
<div class="kamon">
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" fill="#d97fa3">
  <circle cx="50" cy="50" r="46" fill="none" stroke="#d97fa3" stroke-width="2"/>
  <g transform="translate(50,50)">
    <ellipse rx="8" ry="20" fill="#d97fa3" transform="rotate(0)  translate(0,-22)"/>
    <ellipse rx="8" ry="20" fill="#d97fa3" transform="rotate(72) translate(0,-22)"/>
    <ellipse rx="8" ry="20" fill="#d97fa3" transform="rotate(144) translate(0,-22)"/>
    <ellipse rx="8" ry="20" fill="#d97fa3" transform="rotate(216) translate(0,-22)"/>
    <ellipse rx="8" ry="20" fill="#d97fa3" transform="rotate(288) translate(0,-22)"/>
    <circle r="10" fill="#fce8ef"/>
  </g>
</svg>
</div>

<script>
(function() {
    const wrap = document.getElementById('petals');
    const colors = ['#f9c9d8','#f4afc2','#fce4ec','#f8bbd0','#ffe0ea'];
    const count = 18;

    function makePetal(i) {
        const el = document.createElement('div');
        el.className = 'petal';
        const size = 10 + Math.random() * 14;
        const left = Math.random() * 100;
        const dur  = 9 + Math.random() * 12;
        const delay = Math.random() * 16;
        const drift = (Math.random() - 0.5) * 140;
        const spin  = 200 + Math.random() * 360;
        const color = colors[Math.floor(Math.random() * colors.length)];

        el.style.cssText = `
            left: ${left}%;
            width: ${size}px;
            height: ${size}px;
            animation-duration: ${dur}s;
            animation-delay: -${delay}s;
            --drift: ${drift}px;
            --spin: ${spin}deg;
        `;

        el.innerHTML = `
        <svg viewBox="0 0 40 40" width="${size}" height="${size}" xmlns="http://www.w3.org/2000/svg">
          <ellipse cx="20" cy="20" rx="12" ry="18"
            fill="${color}"
            opacity="0.88"
            transform="rotate(${Math.random()*360} 20 20)"/>
          <ellipse cx="20" cy="20" rx="12" ry="18"
            fill="${color}"
            opacity="0.55"
            transform="rotate(${72 + Math.random()*30} 20 20)"/>
          <ellipse cx="20" cy="20" rx="12" ry="18"
            fill="${color}"
            opacity="0.45"
            transform="rotate(${144 + Math.random()*30} 20 20)"/>
          <ellipse cx="20" cy="20" rx="12" ry="18"
            fill="${color}"
            opacity="0.45"
            transform="rotate(${216 + Math.random()*30} 20 20)"/>
          <ellipse cx="20" cy="20" rx="12" ry="18"
            fill="${color}"
            opacity="0.55"
            transform="rotate(${288 + Math.random()*30} 20 20)"/>
          <circle cx="20" cy="20" r="3" fill="rgba(255,230,240,0.9)"/>
        </svg>`;
        wrap.appendChild(el);
    }

    for (let i = 0; i < count; i++) makePetal(i);
})();
</script>
""", unsafe_allow_html=True)

# ── NLP UTILITIES ──────────────────────────────────────────────────────────────
stemmer = PorterStemmer()

def tokenize(sentence):
    return sentence.lower().split()

def stem(word):
    return stemmer.stem(word.lower())

def detect_emotion(text):
    tokens = tokenize(text)
    text_lower = text.lower()

    if "not feeling good" in text_lower or "don't feel good" in text_lower:
        return "sad"

    emotions = {
        "happy":   ["happy","good","great","fine","ok","cheerful","joy","wonderful"],
        "sad":     ["sad","lonely","down","empty","depressed","bad","cry","hopeless"],
        "angry":   ["angry","hate","annoyed","frustrated","furious","mad"],
        "anxious": ["anxious","worried","scared","stress","nervous","panic","afraid"]
    }
    negations = ["not","no","never","n't"]
    scores = {emo: 0 for emo in emotions}

    for i, word in enumerate(tokens):
        for emo, keywords in emotions.items():
            if word in keywords:
                if i > 0 and tokens[i-1] in negations:
                    scores["sad" if emo == "happy" else emo] += (1 if emo == "happy" else -1)
                else:
                    scores[emo] += 1

    return "neutral" if max(scores.values()) == 0 else max(scores, key=scores.get)

# ── MODEL ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

with open('intents.json', 'r') as f:
    intents = json.load(f)

# ── SESSION STATE ──────────────────────────────────────────────────────────────
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = []

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🌸 Anshin AI · 安心 AI")
st.caption("安心して話していいよ　— You can talk freely here 💗")

# ── CHAT HISTORY ───────────────────────────────────────────────────────────────
for sender, msg in st.session_state.messages:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(msg)

# ── INPUT ──────────────────────────────────────────────────────────────────────
user_input = st.chat_input("今日はどんな気持ち？  ·  How are you feeling today…")

if user_input:
    emotion = detect_emotion(user_input)
    st.session_state.memory.append({"text": user_input, "emotion": emotion})
    st.session_state.messages.append(("You", f"{user_input} *({emotion})*"))

    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.markdown("🌸 *Anshin is listening…*")
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

        # Emotion prefix
        emotion_prefix = {
            "sad":     "ごめんね、つらかったんだね。💗  I'm really sorry you're feeling this way. ",
            "anxious": "大丈夫だよ、ゆっくり深呼吸して。🌿  Take a deep breath. You're safe here. ",
            "angry":   "怒るのは当然だよ。🌸  It's okay to feel angry. ",
            "happy":   "それは嬉しいね！✨  That's wonderful to hear! ",
        }
        response = emotion_prefix.get(emotion, "") + response

        # Intent override
        for intent in intents['intents']:
            for pattern in intent["patterns"]:
                if pattern.lower() in user_input.lower():
                    response = random.choice(intent["responses"])
                    break

        thinking.markdown(response)

    st.session_state.messages.append(("Anshin AI", response))
    st.rerun()

# ── ANALYTICS ──────────────────────────────────────────────────────────────────
if st.session_state.memory:
    st.divider()
    st.subheader("📊 Emotion Journal · 感情の記録")
    emotions = [m["emotion"] for m in st.session_state.memory]
    count = Counter(emotions)
    df = pd.DataFrame(list(count.items()), columns=["Emotion", "Count"])
    st.bar_chart(df.set_index("Emotion"))
