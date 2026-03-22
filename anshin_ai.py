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
    --sakura-mist:#fdf0f4;
    --sakura-white:#fffafc;
    --sakura-stem:#6b4c5c;
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
    max-width:780px !important;
    padding:2rem 2rem 5rem !important;
}

h1 {
    font-family:'Noto Serif JP',serif !important;
    font-size:2.3rem !important;
    font-weight:400 !important;
    color:var(--sakura-deep) !important;
    text-align:center !important;
}

[data-testid="stCaptionContainer"] p {
    text-align:center !important;
    color:var(--sakura-stem) !important;
}

[data-testid="stChatMessage"] {
    border-radius:20px !important;
    padding:1.2rem !important;
    margin:0.8rem 0 !important;
}

[data-testid="stChatMessageContent"] p {
    color:var(--sakura-ink) !important;
    font-size:1.05rem !important;
    line-height:1.7 !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background:rgba(244,175,194,0.5) !important;
    border:1px solid rgba(217,127,163,0.3) !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background:rgba(255,250,252,0.95) !important;
    border:1px solid rgba(244,175,194,0.3) !important;
}

[data-testid="stChatInput"] {
    border-radius:24px !important;
    background:rgba(255,250,252,0.98) !important;
    border:1px solid var(--sakura-petal) !important;
}

.petal-wrap {
    position:fixed;
    inset:0;
    pointer-events:none;
    overflow:hidden;
    z-index:9999;
}

.petal {
    position:absolute;
    top:-60px;
    animation:petalFall linear infinite;
}

@keyframes petalFall {
    0%{transform:translateY(0) translateX(0) rotate(0deg);opacity:0;}
    5%{opacity:0.8;}
    100%{transform:translateY(105vh) translateX(var(--drift)) rotate(var(--spin));opacity:0;}
}
</style>

<div class="petal-wrap" id="petals"></div>

<script>
(function(){
var wrap=document.getElementById('petals');
if(!wrap)return;
var colors=['#f9c9d8','#f4afc2','#fce4ec','#f8bbd0'];
for(var i=0;i<12;i++){
var el=document.createElement('div');
el.className='petal';
var size=10+Math.random()*10;
var left=Math.random()*100;
var dur=12+Math.random()*10;
var drift=((Math.random()-0.5)*140).toFixed(1);
var spin=(200+Math.random()*300).toFixed(1);
var color=colors[Math.floor(Math.random()*colors.length)];
el.style.cssText='left:'+left+'%;width:'+size+'px;height:'+size+'px;animation-duration:'+dur+'s;--drift:'+drift+'px;--spin:'+spin+'deg;';
el.innerHTML='<svg viewBox="0 0 40 40"><ellipse cx="20" cy="20" rx="11" ry="18" fill="'+color+'" opacity="0.7"/></svg>';
wrap.appendChild(el);
}
})();
</script>
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

user_input=st.chat_input("今日はどんな気持ち？ · How are you feeling today…")

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
