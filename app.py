import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import easyocr
import asyncio
import edge_tts
from PIL import Image
import numpy as np
import folium
from streamlit_folium import st_folium
import gc

# Configuration légère
st.set_page_config(page_title="Universal Bridge AI", layout="centered")

# --- DONNÉES ---
LANG_CODES = {"Français": "fra_Latn", "Anglais": "eng_Latn", "Turc": "tur_Latn", "Espagnol": "spa_Latn", "Chinois": "zho_Hans", "Coréen": "kor_Hang"}
MAP_DATA = {"Français": [46.2, 2.2], "Anglais": [37.0, -95.0], "Turc": [38.9, 35.2], "Espagnol": [40.4, -3.7], "Chinois": [35.8, 104.1], "Coréen": [35.9, 127.7]}

# --- CHARGEMENT À LA DEMANDE (LAZY LOADING) ---
def translate_text(text, target_lang_name):
    model_name = "facebook/nllb-200-distilled-600M"
    # Chargement local pour économiser la RAM globale
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16)
    
    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs, 
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANG_CODES[target_lang_name])
    )
    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    
    # Nettoyage immédiat
    del tokenizer
    del model
    gc.collect()
    return result

# --- INTERFACE ---
st.title("🌐 Universal Bridge AI")
st.write("Optimisé pour le déploiement Cloud")

input_text = st.text_area("Texte à traduire :", height=150)
target_lang = st.selectbox("Vers quelle langue ?", list(LANG_CODES.keys()))

if st.button("🚀 TRADUIRE"):
    if input_text.strip():
        with st.spinner("L'IA travaille (chargement du modèle)..."):
            try:
                res = translate_text(input_text, target_lang)
                st.success(f"**Traduction :** {res}")
                
                # Carte
                m = folium.Map(location=MAP_DATA[target_lang], zoom_start=3)
                folium.Marker(MAP_DATA[target_lang]).add_to(m)
                st_folium(m, height=250)
            except Exception as e:
                st.error(f"Erreur de mémoire : {e}. Essayez un texte plus court.")

# Option OCR en Sidebar
with st.sidebar:
    st.header("🖼️ Option Image (OCR)")
    img = st.file_uploader("Scanner une image", type=['png', 'jpg'])
    if img and st.button("Extraire le texte"):
        reader = easyocr.Reader(['fr', 'en'])
        image = Image.open(img)
        text_found = " ".join(reader.readtext(np.array(image), detail=0))
        st.info(f"Texte extrait : {text_found}")
        del reader
        gc.collect()
