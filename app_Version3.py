import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Summarizer setup
@st.cache_resource
def load_mistral():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_mistral()

def mistral_summarize(text):
    prompt = f"[INST] Summarize the following text:\n{text} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = summary.replace(prompt, "").strip()
    return summary

# Coqui TTS setup
@st.cache_resource
def load_coqui():
    from TTS.api import TTS
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=torch.cuda.is_available())
    return tts

tts = load_coqui()

def coqui_tts(text):
    output_path = "output.wav"
    tts.tts_to_file(text=text, file_path=output_path)
    return output_path

# Streamlit UI
st.title("Mistral + Coqui Summarize & Read Aloud")
st.write("Enter any text. The app summarizes it using Mistral-7B-Instruct and reads the summary aloud with Coqui TTS.")

input_text = st.text_area("Input Text", height=200)

if st.button("Summarize and Speak"):
    if input_text.strip():
        with st.spinner("Summarizing..."):
            summary = mistral_summarize(input_text)
        st.success("Summary ready!")
        st.text_area("Summary", summary, height=100)
        with st.spinner("Generating audio..."):
            audio_file = coqui_tts(summary)
        audio_bytes = open(audio_file, "rb").read()
        st.audio(audio_bytes, format="audio/wav")
    else:
        st.warning("Please enter some text to summarize.")