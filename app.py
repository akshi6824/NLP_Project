import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. Load Trained Model (Cached for Speed) ---

@st.cache_resource
def load_model():
    """Loads the fine-tuned DistilBERT model and tokenizer."""
    try:
        model_path = "./privacy_model"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        if model.config.id2label:
            label_columns = [model.config.id2label[i] for i in sorted(model.config.id2label.keys())]
        else:
            st.error("FATAL: Model config missing id2label. Rerun nlp.py to save the model correctly.")
            return None, None, None
            
        return tokenizer, model, label_columns
    except Exception as e:
        st.error(f"FATAL ERROR: Could not load model from './privacy_model'. Have you run nlp.py? Error: {e}")
        return None, None, None

tokenizer, model, label_columns = load_model()

# --- 2. Prediction Function (CORRECTED IMPLEMENTATION) ---

HIGH_RISK_THRESHOLD = 0.3  # Adjusted for better sensitivity
DEFAULT_THRESHOLD = 0.5
HIGH_CONCERN_LABELS = ['Third Party Sharing/Collection', 'International Data Transfer', 'Data Retention']

def is_negated(segment):
    """Checks if a sentence segment contains common negation phrases."""
    segment_lower = segment.lower()
    negation_patterns = [
        r'we do not', r'we will not', r'we never', r'is not shared',
        r'is not sold', r'is not disclosed', r'without your consent',
        r'will not be shared', r'we don\'t'
    ]
    for pattern in negation_patterns:
        if re.search(pattern, segment_lower):
            return True
    return False

def create_overlapping_segments(text_clean, segment_length=50, step_size=25):
    """Creates overlapping segments from a long text. Assumes text is already cleaned."""
    words = text_clean.split()
    if not words:
        return []

    segments = []
    for i in range(0, len(words), step_size):
        segment_words = words[i:i + segment_length]
        
        # This check is safer: only stop if we're not on the first chunk and it's too short.
        if i > 0 and len(segment_words) < 15:
            break
            
        segments.append(" ".join(segment_words))
    return segments


def predict_with_transformer(text, trained_model, trained_tokenizer, labels_list):
    """Predicts privacy risk categories with logic that handles both short and long text."""
    if trained_model is None or not text.strip():
        return [], 0, 0

    # <<< CRITICAL BUG FIX STARTS HERE >>>
    text_clean = text.replace('\n', ' ').strip()
    words = text_clean.split()
    segments = []

    # Heuristic: If the text is short, split by sentences. This is the fix for your test case.
    if len(words) < 30:
        potential_segments = re.split(r'[.?!]\s*', text_clean)
        for s in potential_segments:
            if len(s.strip()) > 4: # Filter out empty splits or junk
                segments.append(s.strip())
        # If there are no sentence breaks, treat the whole short text as one segment
        if not segments and text_clean:
             segments.append(text_clean)
    else:
        # For longer policies, use the robust overlapping segmenter.
        segments = create_overlapping_segments(text_clean)
    # <<< CRITICAL BUG FIX ENDS HERE >>>
    
    if not segments:
        return [], 0, 0

    MAX_SEGMENTS = 150
    if len(segments) > MAX_SEGMENTS:
        segments = segments[:MAX_SEGMENTS]

    inputs = trained_tokenizer(segments, return_tensors="pt", padding=True, truncation=True, max_length=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = trained_model(**inputs).logits
    
    probs = torch.nn.Sigmoid()(logits.cpu())
    if probs.ndim == 1:
        probs = probs.unsqueeze(0)
    
    all_detected_labels = set()
    high_concern_segment_count = 0
    
    label_indices = {label: labels_list.index(label) for label in labels_list}
    high_concern_indices = {label_indices.get(label) for label in HIGH_CONCERN_LABELS}
    critical_indices = {label_indices.get('Third Party Sharing/Collection'), label_indices.get('International Data Transfer')}

    for i, segment_text in enumerate(segments):
        if is_negated(segment_text):
            continue
        
        segment_probs = probs[i].numpy()
        is_high_concern_segment = False
        
        for label_idx, prob in enumerate(segment_probs):
            threshold = HIGH_RISK_THRESHOLD if label_idx in critical_indices else DEFAULT_THRESHOLD
            if prob >= threshold:
                detected_label = labels_list[label_idx]
                all_detected_labels.add(detected_label)
                if label_idx in high_concern_indices:
                    is_high_concern_segment = True
        
        if is_high_concern_segment:
            high_concern_segment_count += 1

    final_risk_score = (high_concern_segment_count / len(segments)) * 100 if segments else 0
    return list(all_detected_labels), final_risk_score, len(segments)

# --- 3. Streamlit UI Configuration ---

st.set_page_config(page_title="Privacy Policy Risk Analyzer", page_icon="🧠", layout="wide")
st.title("🧠 Privacy Policy Risk Analyzer (Transformer Model)")
st.write("Upload a Privacy Policy PDF or paste text to analyze its potential risks using a fine-tuned DistilBERT model.")

option = st.radio("Select Input Type:", ["📄 Upload PDF", "📝 Paste Text"], horizontal=True)
text = ""

if option == "📄 Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        try:
            text = ""
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
                for page in pdf:
                    text += page.get_text()
            st.success("✅ PDF extracted successfully!")
            st.text_area("Extracted Text (Full Content):", text, height=300)
        except Exception as e:
            st.error(f"Error reading PDF: {e}. The file may be corrupt or encrypted.")
else:
    text = st.text_area("Paste your privacy policy text here:", height=300)

if st.button("🔍 Analyze Policy"):
    if model is None:
        st.error("Cannot proceed. Model failed to load.")
    elif text.strip() == "":
        st.warning("Please upload or paste some text before analyzing.")
    else:
        with st.spinner("Analyzing policy... This may take a moment."):
            detected_labels, risk_score, segment_count = predict_with_transformer(text, model, tokenizer, label_columns)
        
        st.header("Risk Analysis Results")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Overall Risk Level")
            # Adjusted thresholds for a more standard distribution
            if risk_score < 20:
                st.success(f"## Low Risk ({risk_score:.1f}%)")
                st.write("Minimal detection of high-concern clauses.")
            elif 20 <= risk_score <= 40:
                st.warning(f"## Medium Risk ({risk_score:.1f}%)")
                st.write("A moderate number of segments mention high-concern activities.")
            else:
                st.error(f"## High Risk ({risk_score:.1f}%)")
                st.write("A significant portion of the policy discusses high-concern activities.")
            
            st.markdown("---")
            st.markdown(f"**Calculated High-Concern Score:** `{risk_score:.2f}%`")
            st.markdown(f"**Total Segments Analyzed:** `{segment_count}`")

        with col2:
            st.subheader("Detected Privacy Clauses")
            if detected_labels:
                critical_ui = [l for l in detected_labels if "Third Party" in l or "International" in l]
                other_labels_ui = [l for l in detected_labels if l not in critical_ui]
                if critical_ui:
                    st.markdown(f"**🔴 Critical Areas:** `{', '.join(critical_ui)}`")
                if other_labels_ui:
                    st.markdown(f"**🟡 Other Clauses:** `{', '.join(other_labels_ui)}`")
            else:
                st.info("✅ No specific privacy risk clauses were detected by the model.")

        st.markdown("---")
        # ... (rest of the UI code is unchanged and correct) ...
        st.subheader("📊 Clause Category Detection")
        categories_for_chart = ["Third Party Sharing/Collection", "International Data Transfer", "Data Retention", "User Choice/Control", "Data Security", "First Party Collection/Use"]
        chart_data = {cat: 1 if cat in detected_labels else 0 for cat in categories_for_chart}
        if any(chart_data.values()):
            chart_df = pd.DataFrame({'Detected': list(chart_data.values())}, index=list(chart_data.keys()))
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ['#FF4B4B' if c in HIGH_CONCERN_LABELS else '#FFC300' for c in chart_df.index]
            ax.barh(chart_df.index, chart_df['Detected'], color=colors, height=0.6)
            ax.set_title(f"Risk Detection by Category (Analyzed {segment_count} Segments)")
            ax.set_xlabel("Clause Detected")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Not Detected", "Detected"])
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No clauses from the primary chart categories were found.")

st.caption("Built with Streamlit and a fine-tuned DistilBERT model.")