# YouTube Toxicity Detection - Dataset & Project Setup Guide

## Part 1: Dataset Download Strategy (Cheapest Approach)

### Overview
You have 8 public datasets to download. **Strategy:** Download locally, upload to GCP using `gsutil` once, then keep in cloud storage for training.

### Datasets List
From Spec A, here are the datasets:

| Phase | Dataset | Source | Size Est. | Priority |
|-------|---------|--------|-----------|----------|
| 0 | Toxic Comments Detection | Kaggle | ~100-200MB | 🔴 Core |
| 0 | Racist/Hate Speech YouTube Comments | Kaggle | ~50MB | 🔴 Core |
| 0 | Civil Comments | Kaggle | ~500MB | 🔴 Core |
| 0 | Hate Speech Reddit Dataset | Kaggle | ~100MB | 🔴 Core |
| 0 | YouTube-8M Comments | Kaggle | ~1-2GB | 🔴 Core |
| 1 | Multilingual Conversation Dataset | Kaggle | ~200MB | 🟠 Supplemental |
| 1 | GPT-4 AI Dataset (242k) | Kaggle | ~100MB | 🟠 Supplemental |
| 1 | Databricks ChatGPT Dataset | Kaggle | ~300MB | 🟠 Supplemental |

**Total estimated: ~2.5-3.5GB** (easily fits in development bucket)

---

## Part 2: Local-First Download Strategy

### Step 1: Create Local Dataset Directory
```bash
mkdir -p ~/datasets/youtube-toxicity
cd ~/datasets/youtube-toxicity
```

### Step 2: Download Datasets from Kaggle Locally

**Prerequisites:**
1. Install Kaggle CLI:
```bash
pip install kaggle
```

2. Get Kaggle API key:
   - Go to kaggle.com → Settings → API
   - Download `kaggle.json`
   - Place in `~/.kaggle/kaggle.json`
   - Run: `chmod 600 ~/.kaggle/kaggle.json`

### Step 3: Download Scripts

**Do this on your LOCAL machine first** (Windows PowerShell or WSL):

```bash
#!/bin/bash
# download_datasets.sh

cd ~/datasets/youtube-toxicity

# Core Datasets
kaggle datasets download -d miadul/toxic-comments-detection-dataset -p ./toxic-comments
kaggle datasets download -d thedevastator/das-racist-you-oughta-know-youtube-comments -p ./hate-speech-youtube
kaggle datasets download -d 'tunguz8/jigsaw-unintended-bias-in-toxicity-classification' -p ./civil-comments
kaggle datasets download -d mrmoralesf/hate-speech-dataset -p ./hate-speech-reddit
kaggle datasets download -d google/youtube-8m -p ./youtube-8m

# Supplemental Datasets
kaggle datasets download -d thedevastator/multilingual-conversation-dataset -p ./multilingual
kaggle datasets download -d thedevastator/gpt-4-ai-dataset-242k-entries -p ./gpt4
kaggle datasets download -d thedevastator/databricks-chatgpt-dataset -p ./databricks-chatgpt

echo "✅ All datasets downloaded to ~/datasets/youtube-toxicity"
```

### Step 4: Upload to GCP Bucket (One-Time)

Once all datasets are downloaded locally, upload them to your GCP bucket:

```bash
# Set your bucket name
BUCKET_NAME="youtube-toxicity-data-v1"

# Create dataset index in bucket
gsutil mb gs://${BUCKET_NAME}/raw/datasets/ 2>/dev/null || echo "Bucket exists"

# Upload all datasets
gsutil -m cp -r ~/datasets/youtube-toxicity/* gs://${BUCKET_NAME}/raw/datasets/

# Verify upload
gsutil ls -hr gs://${BUCKET_NAME}/raw/datasets/

echo "✅ All datasets uploaded to GCP!"
```

### Step 5: Create Local Data Format for Training

**Create a standardized CSV schema locally:**

```
content_id,text,label_toxicity,label_hate_racism,label_harassment,dataset_source,dataset_phase
youtube:comment123,text here,0,0,0,toxic-comments,0
reddit:comment456,text here,1,1,0,hate-speech-reddit,0
...
```

---

## Part 3: Project Structure (Adapted from CraigslistWebScraper)

### Recommended Local Directory Layout

```
YoutubeCommentToxicityDetector/
├── .github/
│   ├── workflows/
│   │   ├── deploy-ingest.yml          # Deploy YouTube ingestion function
│   │   ├── deploy-scorer.yml           # Deploy toxicity scoring function
│   │   ├── deploy-pipeline.yml         # Deploy Vertex AI training pipeline
│   │   └── tests.yml                   # Run tests on PR
│   └── (copy from CraigslistWebScraper)
├── cloud/
│   ├── functions/
│   │   ├── ingest/
│   │   │   ├── main.py                # YouTube comment ingestion
│   │   │   ├── requirements.txt
│   │   │   └── pytho
n_interpreter_config.txt
│   │   └── scorer/
│   │       ├── main.py                # Batch toxicity scoring
│   │       ├── requirements.txt
│   │       └── python_interpreter_config.txt
│   ├── pipelines/
│   │   ├── train.py                   # Vertex AI training pipeline
│   │   └── requirements.txt
│   └── Dockerfile                     # For Cloud Run
├── src/
│   ├── data/
│   │   ├── youtube_api.py             # YouTube Data API wrapper
│   │   ├── dataset_loader.py          # Load & parse datasets
│   │   ├── data_schema.py             # Unified data model
│   │   └── preprocessing.py           # Clean & filter text
│   ├── models/
│   │   ├── transformer_model.py       # PyTorch + HF model
│   │   ├── uncertainty.py             # MC Dropout + SHAP
│   │   └── training.py                # Training loop
│   ├── utils/
│   │   ├── logging.py
│   │   ├── gcs.py                     # GCS utilities
│   │   ├── bigquery.py                # BQ utilities
│   │   └── config.py
│   └── __init__.py
├── notebooks/
│   ├── 01_explore_datasets.ipynb      # LOCAL - data exploration
│   ├── 02_preprocess_datasets.ipynb   # LOCAL - train preprocessing
│   ├── 03_train_baseline_model.ipynb  # LOCAL - model training
│   ├── 04_evaluate_uncertainty.ipynb  # LOCAL - analyze uncertainty
│   └── 05_active_learning_ui.ipynb    # Streamlit for labeling
├── streamlit_app/
│   ├── app.py                         # Main Streamlit dashboard
│   ├── pages/
│   │   ├── overview.py
│   │   ├── explorer.py
│   │   ├── trends.py
│   │   ├── error_analysis.py
│   │   ├── uncertainty_view.py
│   │   ├── shap_viewer.py
│   │   └── labeling.py
│   ├── requirements.txt
│   └── .streamlit/
│       └── config.toml
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_api.py
├── config/
│   ├── datasets.yaml                  # Dataset definitions
│   ├── model_config.yaml              # Model hyperparameters
│   └── secrets_template.env           # Template (don'tcommit real one)
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATA_PIPELINE.md
│   ├── MODEL_TRAINING.md
│   └── DEPLOYMENT.md
├── .gitignore                         # ⚠️ Include: .env, *.pth, datasets/
├── .env.template                      # Template for secrets
├── README.md
├── pyproject.toml                     # Python 3.10+ with Poetry/pip
├── requirements.txt                   # All Python dependencies
└── setup.sh                           # Local setup script
```

---

## Part 4: Python Environment Setup (Local First)

### Step 1: Create Python 3.10+ Virtual Environment

```bash
# Check Python version (need 3.10+)
python --version

# Create venv
python -m venv venv

# Activate
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 2: Install Core Dependencies

**Create `requirements.txt`:**

```txt
# Core Data Science
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0

# PyTorch + HuggingFace
torch==2.0.1
transformers==4.30.2
datasets==2.13.0
accelerate==0.20.3

# Uncertainty & Interpretability
torch-uncertainty==0.1.0  # Or: pip install git+https://github.com/ENSTA-U2IS/torch-uncertainty.git
shap==0.42.2

# GCP & Cloud
google-cloud-storage==2.10.0
google-cloud-bigquery==3.12.0
google-cloud-aiplatform==1.26.1
google-cloud-logging==3.5.0

# APIs
google-api-python-client==2.95.1
google-auth-oauthlib==1.0.0
google-auth-httplib2==0.1.1
python-dotenv==1.0.0

# Dashboard
streamlit==1.26.0
plotly==5.16.1
altair==5.0.1

# NLP Utils
langdetect==1.0.9
nltk==3.8.1

# Development
pytest==7.4.0
black==23.7.0
flake8==6.1.0
```

Install:
```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data (for preprocessing)

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

---

## Part 5: Quick Reference - Milestones & Timeline

| Milestone | What | Tools | Timeline |
|-----------|------|-------|----------|
| **M1** | Ingest 10k YT comments | Python + YouTube API (local) | Week 1 |
| **M2** | Train baseline on 5 datasets | PyTorch + HF (local Jupyter) | Week 2-3 |
| **M3** | Streamlit dashboard (local) | Streamlit + Plotly | Week 3 |
| **M4** | Get 500-1k gold labels | Streamlit UI (local) | Week 4-5 |
| **M5** | Deploy Vertex AI pipeline | Cloud Functions + Scheduler | Week 5-6 |
| **Deploy** | Production on DigitalOcean | Docker + CI/CD | Week 6+ |

---

## Part 6: First Commands to Run

### 1. Clone/Fork CraigslistWebScraper as template:
```bash
git clone https://github.com/dehiska/CraigslistWebScraper.git youtube-toxicity
cd youtube-toxicity
# Rename & adapt structure
```

### 2. Get YouTube API Key (see YOUTUBE_API_SETUP.md - just completed ✅)

### 3. Run GCP Setup (when ready to deploy):
```bash
bash gcp_setup.sh
```

### 4. Download datasets locally (Week 1):
```bash
bash download_datasets.sh
```

### 5. Start with local exploration:
```bash
# Create first notebook
jupyter notebook notebooks/01_explore_datasets.ipynb
```

---

**Ready?** Let me know when you've got your YouTube API key, and I'll help you set up the **local project structure & data ingestion module**!

