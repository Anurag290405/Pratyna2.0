# CivicLens 🏙️

A Streamlit-based citizen complaint analyzer that uses NLP to streamline public grievance redressal.

## 🚀 How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## 🧠 Models Used

- **Classification**: `facebook/bart-large-mnli` (Zero-shot classification for categorizing complaints).
- **Embeddings**: `all-MiniLM-L6-v2` (Sentence transformers for semantic similarity search).

## ✨ Features

- **Automated Categorization**: Sorts complaints into Road Damage, Water Supply, Electricity, etc.
- **Department Mapping**: Automatically routes issues to the correct government body (PWD, Jal Board, etc.).
- **Urgency Scoring**: Calculates urgency (1-10) based on critical keywords.
- **Similarity Search**: Finds historical complaints that are semantically similar to the input.

## 🛣️ Next Steps

- **Fine-tuning**: Train the classification model on real-world PGPortal or local municipal data to improve accuracy.
- **Geospatial Integration**: Add map support to visualize complaint hotspots.
- **Chatbot Interface**: Implement an AI assistant to help citizens draft better complaints.