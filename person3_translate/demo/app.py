import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from streamlit_webrtc import webrtc_streamer

from person3_translate.translate import translate_glosses


st.title("Sign Language → Sentence Translator")

st.write(
    "Sign in front of the camera and convert recognized signs "
    "into a natural English sentence."
)

# -------------------------
# Live Webcam
# -------------------------

st.subheader("Live Camera")

webrtc_streamer(
    key="sign-language-camera",
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)

# -------------------------
# Temporary Gloss Input
# -------------------------

st.subheader("Recognized Glosses")

st.write(
    "Currently using manual gloss input. "
    "Person 2's recognition model will be connected here."
)

gloss_input = st.text_input(
    "Glosses",
    "STORE I GO"
)

# -------------------------
# Translation
# -------------------------

if st.button("Translate"):

    glosses = gloss_input.split()

    if not glosses:
        st.warning("No glosses detected.")

    else:
        sentence = translate_glosses(glosses)

        st.write("**Recognized Glosses:**", glosses)

        st.subheader("Translated Sentence")

        st.success(sentence)