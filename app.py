import os
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

# --- Page Config ---
st.set_page_config(page_title="CivicLens - Citizen Complaint Analyzer", layout="wide")

# --- Hardcoded Dataset ---
COMPLAINTS_DATA = [
    # Road Damage
    {"text": "Huge potholes on the MG road causing traffic jams.", "category": "road damage"},
    {"text": "The street lights are broken and the road is cracked.", "category": "road damage"},
    {"text": "Drainage covers are missing on the highway.", "category": "road damage"},
    {"text": "Road surface has completely eroded after the rain.", "category": "road damage"},
    # Water Supply
    {"text": "No water supply in our colony for the last three days.", "category": "water supply"},
    {"text": "Water coming from the taps is dirty and muddy.", "category": "water supply"},
    {"text": "Major pipeline leak in front of the community center.", "category": "water supply"},
    {"text": "Low water pressure making it impossible to use the shower.", "category": "water supply"},
    # Electricity
    {"text": "Frequent power cuts in our area every evening.", "category": "electricity"},
    {"text": "Transformer burst and we have no electricity since morning.", "category": "electricity"},
    {"text": "Sparking wires near the tree posing a fire hazard.", "category": "electricity"},
    {"text": "Electricity bill is incorrectly calculated this month.", "category": "electricity"},
    # Garbage Collection
    {"text": "Garbage hasn't been collected from our street for a week.", "category": "garbage collection"},
    {"text": "Huge pile of trash stinking near the public park.", "category": "garbage collection"},
    {"text": "Request for more dustbins in the market area.", "category": "garbage collection"},
    {"text": "Open dumping of waste causing health issues in the neighborhood.", "category": "garbage collection"},
    # Encroachment
    {"text": "Illegal construction on the public pavement.", "category": "encroachment"},
    {"text": "Shops are extending their counters into the main road.", "category": "encroachment"},
    {"text": "Unauthorized parking lot created on government land.", "category": "encroachment"},
    {"text": "A new wall is being built across a public path.", "category": "encroachment"},
    # Noise Pollution
    {"text": "Loud music from the club late at night is disturbing sleep.", "category": "noise pollution"},
    {"text": "Continuous honking near the hospital zone.", "category": "noise pollution"},
    {"text": "Construction noise even during the night hours.", "category": "noise pollution"},
    {"text": "Use of loudspeakers in the residential area after 10 PM.", "category": "noise pollution"},
]

CATEGORY_DEPT_MAP = {
    "road damage": "PWD",
    "water supply": "Jal Board",
    "electricity": "MPEB",
    "garbage collection": "Municipal Corporation",
    "encroachment": "Town Planning",
    "noise pollution": "Police"
}

CATEGORIES = list(CATEGORY_DEPT_MAP.keys())

# --- Models Caching ---
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- Helper Functions ---
def calculate_urgency(text):
    text = text.lower()
    score = 1
    urgency_keywords = {
        "fire": 10,
        "collapse": 10,
        "flood": 9,
        "flooding": 9,
        "accident": 8,
        "emergency": 8,
        "days": 4, # Adds to base score if mentioned
        "weeks": 3
    }
    
    found_scores = [score]
    for word, s in urgency_keywords.items():
        if word in text:
            found_scores.append(s)
            
    # Simple logic: take the maximum keyword score
    return min(max(found_scores), 10)

# --- App Logic ---
st.title("🏙️ CivicLens")
st.markdown("Analyzing citizen complaints with cutting-edge NLP.")

# --- Sidebar ---
st.sidebar.header("Data Insights")
df_data = pd.DataFrame(COMPLAINTS_DATA)
category_counts = df_data['category'].value_counts()
st.sidebar.subheader("Complaint Distribution")
st.sidebar.bar_chart(category_counts)

# --- Main Interaction ---
complaint_text = st.text_area("Enter the complaint description:", placeholder="e.g., There is a huge fire near the electricity transformer...")
analyze_btn = st.button("Analyze")

if analyze_btn and complaint_text:
    with st.spinner("Analyzing your complaint..."):
        # 1. Classification
        classifier = load_classifier()
        result = classifier(complaint_text, CATEGORIES)
        top_category = result['labels'][0]
        
        # 2. Urgency Score
        urgency_score = calculate_urgency(complaint_text)
        
        # 3. Semantic Similarity
        embedder = load_embedder()
        complaint_embedding = embedder.encode(complaint_text, convert_to_tensor=True)
        dataset_complaints = [c['text'] for c in COMPLAINTS_DATA]
        dataset_embeddings = embedder.encode(dataset_complaints, convert_to_tensor=True)
        
        cos_scores = util.cos_sim(complaint_embedding, dataset_embeddings)[0]
        top_results = torch.topk(cos_scores, k=3)
        
        # --- Display Results ---
        st.subheader("Analysis Results")
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Category", top_category.title())
        col2.metric("Department", CATEGORY_DEPT_MAP[top_category])
        col3.metric("Urgency Score", f"{urgency_score}/10")
        
        st.write("---")
        st.subheader("Top 3 Semantically Similar Complaints")
        
        similar_list = []
        for score, idx in zip(top_results[0], top_results[1]):
            similar_list.append({
                "Sample Complaint": COMPLAINTS_DATA[idx]['text'],
                "Category": COMPLAINTS_DATA[idx]['category'].title(),
                "Similarity Score": f"{float(score)*100:.2f}%"
            })
        
        st.table(pd.DataFrame(similar_list))

elif analyze_btn and not complaint_text:
    st.warning("Please enter some text to analyze.")
