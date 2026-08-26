import streamlit as st
from transformers import pipeline

# Load a state-of-the-art emotion detection model from Hugging Face
# This model detects a heavy spectrum of emotions: joy, sadness, anger, fear, surprise, disgust, neutral
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

classifier = load_model()

st.set_page_config(page_title="Deep Sentiment & Emotion Analyzer", page_icon="🎭")
st.title("🎭 Deep Emotion & Sarcasm-Aware Sentiment Analyzer")
st.write("Type a heavy emotional text, a dramatic rant, or a sarcastic remark below to see the underlying emotional breakdown!")

# User input text box
user_input = st.text_area("Enter your text here:", "Oh fantastic, another bug in production on a Friday evening. I just love my life.")

if st.button("Analyze Emotions"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Decoding emotions and detecting sarcasm patterns..."):
            results = classifier(user_input)[0]
            
            # Sort results by score highest to lowest
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            top_emotion = results[0]['label'].upper()
            top_score = results[0]['score']

            st.subheader(f"Primary Detected Vibe: {top_emotion} ({top_score*100:.1f}%)")
            
            # Display full emotional spectrum progress bars
            st.write("### Full Emotional Breakdown:")
            for res in results:
                emotion_name = res['label'].capitalize()
                score = res['score']
                st.progress(float(score), text=f"{emotion_name}: {score*100:.1f}%")
                
            # Sarcasm / Tone heuristic helper note
            if top_emotion in ["DISGUST", "ANGER"] and any(word in user_input.lower() for word in ["just love", "great", "wonderful", "fantastic", "amazing"]):
                st.error("🚨 **Sarcasm Alert Detected:** The wording sounds positive, but the underlying emotional weight leans negative/cynical!")
