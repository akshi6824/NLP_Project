# Privacy Policy Risk Detector: A Transformer-Based Multi-Label Classification System

This repository contains the implementation of a transformer-based system designed to automatically detect privacy risks in online privacy policies. The system employs the DistilBERT architecture and a custom multi-label classification framework to identify data practices such as information collection, third-party sharing, retention, security, and user control mechanisms.

The objective of this project is to support automated compliance analysis and privacy governance by providing structured predictions derived from unstructured policy text.

---

# Abstract

Privacy policies are often lengthy and difficult for users to interpret. Automated systems that classify privacy practices can improve transparency and compliance monitoring. This project presents a Privacy Policy Risk Detector trained on the OPP-115 dataset using a DistilBERT model fine-tuned for multi-label classification. The model integrates a weighted loss function to address label imbalance and uses safe-text augmentation to reduce false positives. A Streamlit application is included for interactive inference and demonstration.

---

# System Architecture

The system consists of the following components:

1. **Data Processing Module**  
   - Loads and preprocesses the OPP-115 dataset  
   - Tokenizes policy segments using Hugging Face tokenizers  
   - Supports integration of additional safe-text samples  

2. **Model Training Module**  
   - Fine-tunes DistilBERT using supervised multi-label classification  
   - Employs BCEWithLogitsLoss with class-specific weights  
   - Outputs evaluation metrics such as Micro-F1, precision, recall, and accuracy  

3. **Inference and Deployment Module**  
   - Includes a Streamlit-based interface for real-time predictions  
   - Provides multi-label confidence scores  
   - Supports batch or single-text inference  

4. **Model Storage and Checkpointing**  
   - Trained models and tokenizers are saved in the `privacy_model/` directory  
   - Intermediate checkpoints and logs are stored under `results/`

---

# Technologies and Frameworks

- Python 3.9+
- PyTorch
- Hugging Face Transformers
- Scikit-Learn
- Pandas, NumPy
- Streamlit (for inference UI)
- Hugging Face Datasets

---

# Directory Structure

```
NLP Project
│
├── nlp.py                         # Model training and evaluation script
├── app.py                         # Streamlit interface for inference
├── privacy_model/                 # Trained model files and tokenizer
├── OPP-115/                       # Annotated privacy policy dataset
├── results/                       # Checkpoints, logs, and metrics
└── README.md                      # Documentation
```

---

# Installation Instructions

## Clone the Repository
```bash
git clone https://github.com/akshi6824/NLP_Project.git
cd NLP_Project
```

## Install Required Packages
```bash
pip install -r requirements.txt
```

---

# Training Procedure

To reproduce the training process:

```bash
python nlp.py
```

This script performs the following operations:

- Loads the annotated dataset  
- Prepares tokenized inputs  
- Fine-tunes the DistilBERT model  
- Computes performance metrics  
- Saves the model and tokenizer  

The final model will be available in the `privacy_model/` directory.

---

# Running the Inference Interface

To launch the interactive UI:

```bash
streamlit run app.py
```

The interface supports:

- Real-time text classification  
- Multi-label output  
- Confidence scoring for each predicted label  

---

# Dataset Description

The project utilizes the **OPP-115 dataset**, which contains:

- 115 privacy policies from popular websites  
- Annotations across multiple privacy categories  
- Segment-level labels describing data practices  

This dataset is widely used in privacy policy analysis research.

---

# Model Specifications

| Component | Details |
|----------|---------|
| Base Architecture | DistilBERT (uncased) |
| Task Type | Multi-label classification |
| Loss Function | BCEWithLogitsLoss with custom class weights |
| Metrics | Micro-F1, precision, recall, accuracy |
| Augmentation | Safe-text injection for improved robustness |

---

# Experimental Results

The system achieves strong performance across multiple privacy risk labels, demonstrating the effectiveness of transformer-based architectures for policy analysis tasks. Detailed metrics are available in the `results/` directory.

---

# Future Research Directions

Potential extensions include:

- Integration of explainable AI (e.g., SHAP, attention visualization)
- Development of compliance scoring mechanisms (GDPR, CCPA)
- Deployment as a REST API for large-scale analysis
- Model compression and optimization for real-time applications
- Expansion to multilingual privacy policies

---

# Author

*1. Akshita Sharma*
*2. Shrutee Salpe*
*3. Parth Gupta*

Project Leads and Research Contributors
---

## License

This project is distributed under the MIT License (© 2025 Akshita Sharma).  
Users are permitted to use, modify, and distribute the software in accordance with the license terms.

