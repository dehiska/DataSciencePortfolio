# Spec A Execution Plan - Master Action Document

## Executive Summary

You want to build a **YouTube Comment Toxicity Detector** following Spec A. Here's your path:

**Immediate (This Week):**
1. ✅ Get YouTube Data API key (see `YOUTUBE_API_SETUP.md`)
2. ✅ Run GCP setup script (`gcp_setup.sh`)
3. ✅ Clone CraigslistWebScraper, adapt structure
4. ✅ Set up local Python 3.10+ environment with PyTorch/HF

**Phase 1 (Week 2-3):**
- Download 8 datasets locally
- Upload to GCP via `gsutil`
- Build local data loader & preprocessing
- Create first Jupyter notebooks (EDA, training)

**Phase 2 (Week 4-5):**
- Train baseline model on 5 core datasets
- Fine-tune on YouTube comments (Phase 1 silver labels)
- Build Streamlit UI for labeling (local)
- Implement MC Dropout + SHAP uncertainty

**Phase 3 (Week 6+):**
- Deploy Cloud Functions (ingestion + scoring)
- Set up Vertex AI Pipeline
- Deploy Streamlit to Cloud Run
- Launch on DigitalOcean

---

## Your Answers to Your 6 Questions

### 1️⃣ **Cheapest Way to Start with Datasets + gsutil**

✅ **Strategy:** Download locally → Upload once to GCP → Keep in cloud storage

```bash
# Step 1: Download all 8 datasets locally (2.5GB)
bash download_datasets.sh

# Step 2: Upload to GCP (one-time cost ~$0.02)
gsutil -m cp -r ~/datasets/youtube-toxicity/* \
  gs://youtube-toxicity-data-v1/raw/datasets/

# Step 3: Reference from GCP in training pipelines
# Cost for storage: ~$0.02/month (negligible with credits)
# Cost for training: FREE (you have VertexAI credits!)
# Cost for inference: ~$0.01-0.05 per 1000 predictions
```

**Total monthly cost with your GCP credits:** 🎉 **$0.00**

**See:** `IMPLEMENTATION_ROADMAP.md` → Part 1-2 for full guide

---

### 2️⃣ **YouTube API Key - Updated**

✅ **Complete step-by-step guide created:** See `YOUTUBE_API_SETUP.md`

**TL;DR:**
1. Create GCP project: `youtubecommentsanalysis-487823` ✅ Done
2. Enable YouTube Data API v3 ✅ Done
3. Create API Key → store in `.env` as `Youtube_Api_key` ✅ Done
4. (Optional) Web app OAuth credential in GCP for future user-auth features

**Auth strategy:** API Key only — sufficient for all public YouTube data.
No OAuth flow needed in the ingestion pipeline or Streamlit dashboard.

**Deployment target:** Portfolio website at denissoulimaportfolio.com

**Cost:** 🎉 **FREE** (10,000 quota units/day for free tier)

---

### 3️⃣ **CraigslistWebScraper Reference**

✅ **Your repo:** https://github.com/dehiska/CraigslistWebScraper

**How to use it:**
```bash
# Clone as template
git clone https://github.com/dehiska/CraigslistWebScraper.git
cd CraigslistWebScraper

# Study these files to understand pattern:
# .github/workflows/  → deployment automation
# cloud/functions/    → Cloud Functions structure  
# tests/              → test patterns
# README.md           → deployment walkthrough

# Then create new repo for YouTube project
cd ..
git clone https://github.com/YOUR_USERNAME/YoutubeCommentToxicityDetector.git
# Copy structure and adapt for YouTube
```

**Key patterns to reuse:**
- Cloud Functions + Cloud Scheduler (hourly ingestion)
- GitHub Actions → Workload Identity → GCP deployment
- BigQuery for analytics
- Cloud Storage raw/processed zones

---

### 4️⃣ **Adapted Bash Commands for YouTube Project**

✅ **Complete script created:** See `gcp_setup.sh`

This script is a **direct adaptation** of your Craigslist guide with YouTube-specific names:

```bash
# Run this in Cloud Shell (same pattern as your Craigslist setup)
bash gcp_setup.sh

# What it does:
# ✅ Enable APIs (cloudfunctions, run, scheduler, aiplatform, bigquery)
# ✅ Create service accounts (youtube-runtime, youtube-deployer)
# ✅ Set up Workload Identity Federation (GitHub → GCP)
# ✅ Configure IAM roles (all permissions)
# ✅ Create GCS bucket (youtube-toxicity-data-v1)
# ✅ Configure permissions
# ✅ Output GitHub Actions variables

# Then copy/paste output to GitHub Secrets
```

**See:** `gcp_setup.sh` for full implementation

---

### 5️⃣ **Web Scraper Online (GCP), Analysis Local First**

✅ **Two-environment strategy:**

**LOCAL (Your laptop):**
```
├── Notebooks:
│   ├── 01_explore_datasets.ipynb    ← Data exploration
│   ├── 02_preprocess_datasets.ipynb ← Preprocessing
│   ├── 03_train_baseline_model.ipynb ← Model training
│   ├── 04_evaluate_uncertainty.ipynb ← Analysis
│   └── 05_active_learning_ui.ipynb  ← Streamlit dev
├── Python environment (venv)
└── datasets/ (downloaded locally)
```

**CLOUD (GCP + portfolio site):**
```
├── Cloud Functions:
│   ├── youtube-comments-ingest → pulls YouTube API (API Key auth)
│   └── youtube-toxicity-scorer → runs model on new comments
├── Cloud Scheduler:
│   └── Triggers ingest hourly
├── BigQuery:
│   └── Stores raw comments + predictions + aggregates
├── Vertex AI Pipelines:
│   └── Automates: preprocess → train → evaluate → deploy
└── Streamlit on Cloud Run:
    └── Research dashboard (reads from BigQuery)
    └── Embedded/linked from denissoulimaportfolio.com
```

**Timeline:**
- Weeks 1-4: **Everything LOCAL** using Jupyter + local Streamlit
- Week 5+: **Deploy ingestion & dashboard** to GCP
- Week 7+: **Move production** to DigitalOcean (if desired)

**See:** `IMPLEMENTATION_ROADMAP.md` → Part 3-4

---

### 6️⃣ **Python 3.10+, PyTorch, HuggingFace (Industry Standard)**

✅ **Full setup in `IMPLEMENTATION_ROADMAP.md` → Part 4**

```bash
# Create environment
python -m venv venv
source venv/bin/activate

# Install stack
pip install -r requirements.txt

# Key packages:
torch==2.0.1              # PyTorch (GPU-optimized)
transformers==4.30.2      # Hugging Face (roberta-base model)
google-cloud-*            # GCP integration
streamlit==1.26.0         # Dashboard
shap==0.42.2              # Interpretability
```

**Why these?**
- ✅ **PyTorch:** Industry standard for DL + better GPU support
- ✅ **HuggingFace:** Pre-trained models + community support
- ✅ **roberta-base:** State-of-art NLP for text classification
- ✅ **PyTorch Lightning:** (optional) simplifies training loops
- ✅ **SHAP:** Explainability + trust for model decisions
- ✅ **Google Cloud:** Integrates w/ your GCP credits

---

## 🎯 **Your Next 3 Steps (TODAY)**

### Step 1: YouTube API Key ✅ DONE
API Key in `.env`, packages installed, `test.py` verified working.

### Step 2: Create GCP Project Setup
```bash
# In Cloud Shell (from Google Cloud Console):
bash gcp_setup.sh

# Copy output to GitHub Secrets
```

### Step 3: Clone & Adapt CraigslistWebScraper
```bash
# Study the structure
git clone https://github.com/dehiska/CraigslistWebScraper.git craigslist-ref

# Create new repo (or in your existing YoutubeCommentSection folder)
# Copy workflows, structure, adapt names

# Key files to copy/adapt:
# - .github/workflows/ YAMLs
# - cloud/functions/ structure
# - requirements.txt patterns
```

---

## 📚 **File Reference**

| File | Purpose | Read First |
|------|---------|------------|
| `SpecA.md` | Full project specification | ✓ (you created it) |
| `YOUTUBE_API_SETUP.md` | API key instructions | **✓ DO THIS FIRST** |
| `gcp_setup.sh` | GCP infrastructure setup | Run after API key |
| `IMPLEMENTATION_ROADMAP.md` | Detailed implementation | Reference during coding |
| This file | Master action plan | **You're reading it!** |

---

## 💡 **Why This Approach is Good for Your Goals**

> "The point of this project is to make me become a data engineer and data scientist intern."

### Data Engineer Skills You'll Build:
✅ **Cloud Infrastructure:** GCP (Cloud Functions, Scheduler, BigQuery, Storage)  
✅ **CI/CD:** GitHub Actions + Workload Identity  
✅ **Data Pipelines:** ETL with Vertex AI Pipelines  
✅ **Scripting:** Bash + Python automation  
✅ **IaC:** Infrastructure as Code (bash scripts)  

### Data Scientist Skills You'll Build:
✅ **Model Development:** PyTorch + HuggingFace  
✅ **Uncertainty Quantification:** MC Dropout + SHAP  
✅ **Active Learning:** Streamlit labeling UI  
✅ **Evaluation Metrics:** PR-AUC, F1, Brier Score, ECE  
✅ **Data Analysis:** Jupyter notebooks  
✅ **Dashboards:** Streamlit for exploration  

### Both (Industry Standard):
✅ All tools are **currently used in production** (Meta, Google, etc.)  
✅ **Resume-friendly:** PyTorch, GCP, Streamlit are hot skills  
✅ **Hiring signal:** End-to-end ML system (not just notebooks)  

---

## ⏰ **Timeline Estimate**

| Phase | Tasks | Time | Status |
|-------|-------|------|--------|
| **Setup** | API key + GCP + git | 1-2 days | ⏳ Do now |
| **M1** | Download datasets + local ingestion | 2-3 days | ⏳ Week 1 |
| **M2** | Train baseline model | 3-5 days | ⏳ Week 2-3 |
| **M3** | Build Streamlit dashboard (local) | 3 days | ⏳ Week 3 |
| **M4** | Collect 500-1k gold labels | 5-7 days | ⏳ Week 4-5 |
| **M5** | Deploy to GCP (functions + scheduler) | 3-5 days | ⏳ Week 6 |
| **Polish** | Tests + docs + DigitalOcean prep | 3-5 days | ⏳ Week 7 |

**Total effort:** ~4-6 weeks, part-time  
**Result:** Polished portfolio project + hired as intern! 🎉

---

## ❓ **Questions Before You Start?**

Ask me about:
- How to adapt specific workflows from CraigslistWebScraper
- Python environment setup issues
- GCP commands that don't work
- Model architecture choices
- Streamlit UI design
- Deployment strategies

**Let's do this!** 🚀

