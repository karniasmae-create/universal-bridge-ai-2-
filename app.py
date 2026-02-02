import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import torch
import easyocr
import asyncio
import edge_tts
from PIL import Image
import numpy as np
import folium
from streamlit_folium import st_folium
import docx
import fitz  # PyMuPDF
import gc

# Configuration
st.set_page_config(page_title="Universal Bridge AI", layout="wide")

# --- 1. CONFIGURATION DES DONNÉES ---
LANG_CODES = {"Français": "fra_Latn", "Anglais": "eng_Latn", "Turc": "tur_Latn", "Espagnol": "spa_Latn", "Chinois": "zho_Hans", "Coréen": "kor_Hang"}
MAP_DATA = {"Français": [46.2, 2.2], "Anglais": [37.0, -95.0], "Turc": [38.9, 35.2], "Espagnol": [40.4, -3.7], "Chinois": [35.8, 104.1], "Coréen": [35.9, 127.7]}

# --- 2. FONCTIONS DE CHARGEMENT INTELLIGENT (ÉCONOMIE RAM) ---
def get_translator():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16)
    return tokenizer, model

def get_chatbot():
    chat_name = "facebook/blenderbot-400M-distill"
    tokenizer = AutoTokenizer.from_pretrained(chat_name)
    model = AutoModelForCausalLM.from_pretrained(chat_name, torch_dtype=torch.float16)
    return tokenizer, model

# --- 3. INTERFACE PRINCIPALE ---
st.title("🌐 Universal Bridge AI (Version Complète)")

tabs = st.tabs(["🔄 Traduction & Documents", "💬 Chatbot Assistant"])

with tabs[0]:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Entrée")
        uploaded_file = st.file_uploader("Importer (PDF, DOCX, Image)", type=["pdf", "docx", "png", "jpg"])
        
        extracted_text = ""
        if uploaded_file:
            with st.spinner("Extraction du texte..."):
                if uploaded_file.type == "application/pdf":
                    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    extracted_text = " ".join([page.get_text() for page in doc])
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    doc = docx.Document(uploaded_file)
                    extracted_text = " ".join([p.text for p in doc.paragraphs])
                else:
                    reader = easyocr.Reader(['fr', 'en'])
                    extracted_text = " ".join(reader.readtext(np.array(Image.open(uploaded_file)), detail=0))
                    del reader
                    gc.collect()

        input_text = st.text_area("Texte à traiter :", value=extracted_text, height=150)
        target_lang = st.selectbox("Traduire vers :", list(LANG_CODES.keys()))

        if st.button("🚀 TRADUIRE"):
            if input_text:
                with st.spinner("Chargement du traducteur..."):
                    tk, md = get_translator()
                    inputs = tk(input_text, return_tensors="pt")
                    out = md.generate(**inputs, forced_bos_token_id=tk.convert_tokens_to_ids(LANG_CODES[target_lang]))
                    st.session_state.last_res = tk.batch_decode(out, skip_special_tokens=True)[0]
                    del tk, md
                    gc.collect()

    with col2:
        st.subheader("📤 Résultat")
        if "last_res" in st.session_state:
            st.success(st.session_state.last_res)
            m = folium.Map(location=MAP_DATA[target_lang], zoom_start=3)
            folium.Marker(MAP_DATA[target_lang]).add_to(m)
            st_folium(m, height=250)

with tabs[1]:
    st.subheader("💬 Chatbot IA")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
        
    if p := st.chat_input("Posez une question..."):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.spinner("Le bot réfléchit..."):
            ctk, cmd = get_chatbot()
            inp = ctk([p], return_tensors="pt")
            res = cmd.generate(**inp, max_new_tokens=100)
            ans = ctk.decode(res[0], skip_special_tokens=True)
            st.session_state.messages.append({"role": "assistant", "content": ans})
            del ctk, cmd
            gc.collect()
            st.rerun()
