Please comment your recommendations on this spec A
✅ Spec A — Toxicity + Hate/Racism + Harassment (GCP + Vertex AI, YouTube-first, PyTorch)
Revised for Portfolio Integration — Reusing CraigslistWebScraper Architecture

1. Product Goal
Build an end-to-end pipeline that ingests public YouTube comments, scores them with a PyTorch transformer model + deep neural network for uncertainty quantification, and provides a Streamlit research dashboard to explore trends, error cases, model drift, and comment context (including parent/replies). Deploy on Google Cloud using the same architecture as CraigslistWebScraper.

2. Classification Targets
toxicity: General toxic/abusive language (insults, profane attacks).
hate_racism: Hate speech, racist content, slurs, dehumanization.
harassment_bin: Harassment/threats targeted at a person/group (one combined bin).
3. System Architecture (GCP — Reuse CraigslistWebScraper Bones)
Cloud Scheduler → triggers ingestion hourly.
Cloud Run (Ingestion Service) → calls YouTube Data API v3 with API Key, writes raw JSONL to Cloud Storage.
Cloud Storage (Raw Zone) → gs://<bucket>/raw/youtube/date=YYYY-MM-DD/*.jsonl
Cloud Run Job (Preprocess) → cleans text, enforces schema, filters English, writes parquet/JSONL.
BigQuery (Analytics) → stores cleaned comments + scores + aggregates + ingestion metadata.
Vertex AI Pipelines → orchestrates preprocess → train → evaluate → register → batch predict.
Vertex AI Custom Training (PyTorch/HF) → trains/fine-tunes model on labeled data.
Vertex AI Model Registry + Batch Prediction → produces daily scoring outputs to BQ.
UI: Streamlit deployed as a web app (portfolio page at denissoulimaportfolio.com) reading from BigQuery.
Logging/Monitoring: Cloud Logging + Error Reporting.
✅ Architecture identical to CraigslistWebScraper — no changes to core services.
✅ No user-facing OAuth required — API Key is sufficient for reading public YouTube data.

4. Data Ingestion (YouTube-first)
Auth: YouTube Data API v3 with API Key (stored in GCP Secret Manager for Cloud Run, .env for local dev).
No OAuth required — all target data is from public channels with public comments.
Primary endpoint: commentThreads.list with part=snippet,replies for top-level comments + replies.
Ingestion mode: Channel-based — maintain curated list of channel IDs; pull latest videos → pull comments.
Cache: store (video_id, page_token, last_seen_comment_id) to avoid re-fetching.
Quota controls: prefer commentThreads.list, use backoff and retries. Default quota: 10,000 units/day (free).
Language filter: Use langdetect or Google Cloud Language API → drop non-English → log to lang_detection_status, lang_confidence.
5. Data Model (Unified Schema — YouTube Only)
content_id = 'youtube:' + comment_id
platform = 'youtube'
source = channel_id (and channel_title if available)
thread_id = video_id
parent_id = parent comment id (null for top-level)
text_raw, text_clean
created_at, collected_at
lang = 'en' (enforced)
engagement: likeCount, replyCount
model_version, scores (JSON: per-label probability), decision (thresholded)
metadata JSON: video_title, video_publishedAt, parent_comment_text, replies_text_array
Ingestion Metadata (NEW):
ingestion_status (success/error)
quota_used, quota_remaining
api_call_duration_ms
lang_detection_status, lang_confidence
raw_file_size_bytes
6. Labeling Strategy (Fast path to a working model)
Phase 0 (Bootstrapping): train baseline on 5 public labeled datasets (see below).
Phase 1 (Silver labels): run baseline on YouTube comments → keep only high-confidence predictions.
Phase 2 (Gold set): label 500–1,000 comments via Streamlit UI (CSV in GCS).
Phase 3 (Iterate): active learning loop: sample high-uncertainty examples for labeling.
7. Model Training (Vertex AI + PyTorch + Deep NN for Uncertainty)
Framework: PyTorch + Hugging Face Transformers
Backbone: roberta-base (English-only)
Head: multi-label sigmoid classifier
Loss: BCEWithLogitsLoss with pos_weight for class imbalance
Eval: per-label PR-AUC + F1 + Brier Score + ECE
Artifacts: save tokenizer + model to GCS; register in Vertex AI Model Registry
✅ Add Deep Neural Network for Uncertainty Quantification (NEW)

Architecture:
Add MC Dropout (enable dropout at inference)
Use Monte Carlo sampling (T=10 forward passes)
Output: mean_score, variance_score (epistemic uncertainty)
Use SHAP values (via shap.Explainer) for feature attribution
Calculate aleatoric uncertainty via predictive variance (if using probabilistic output head)
Output Fields (in BigQuery):
uncertainty_epistemic (variance across MC samples)
uncertainty_aleatoric (model output variance)
shap_values (JSON: per-token SHAP scores for top 3 labels)
mc_dropout_samples (T=10 scores per label)
Why: Enables active learning, error analysis, and model trustworthiness in research dashboard.
8. Batch Scoring & Analytics
Daily batch prediction job on Vertex AI against new comments in GCS or BigQuery.
Write outputs to BigQuery: predictions table partitioned by date, clustered by source/video.
Compute aggregates: mean risk by day, top sources by spike, label co-occurrence.
Drift signals: score distribution shift per source (KL divergence).
9. Research Dashboard UI (Streamlit on Cloud Run — Portfolio Ready)
Overview: ingestion volume, % flagged, label distributions, top sources by risk.
Explorer: filter by channel/video/date, sort by highest risk, open full thread context (parent + replies).
Trends: time series of risk rates, spike detection, compare sources.
Error analysis: confusion matrix on gold set, threshold tuning controls.
Uncertainty View (NEW): show comments with highest epistemic/aleatoric uncertainty.
SHAP Viewer (NEW): visualize token-level importance for selected comment.
Labeling Tab: present random / uncertain samples for quick human labeling.
10. Milestones (CPU-friendly but Cloud-executed)
M1: Ingest 10,000 YouTube comments → store in GCS + BigQuery + ingestion metadata.
M2: Baseline model trained on 5 public datasets → run batch scoring on 10k comments.
M3: Streamlit research dashboard running on Cloud Run reading from BigQuery → include thread context + uncertainty view.
M4: 500–1,000 gold labels → retrain model on Vertex AI → improved calibration and fewer false positives.
M5: Vertex AI Pipeline to automate M1–M4 end-to-end.
11. Security / Compliance (practical)
Do not store personal identifiers; do not store author usernames unless hashed.
Keep raw text access controlled; publish only aggregates in dashboards if needed.
API Key stored in GCP Secret Manager for Cloud Run; .env locally (never committed to git).
No OAuth credentials needed for core pipeline — API Key only.
Restrict service account permissions (least privilege).
Log requests and quotas for audit and cost control.
📚 Datasets (Final Confirmed — 5 Active, 2 Discarded)
Kaggle slug → local folder in data/kaggle/

🔴 LABELED — toxicity ground truth, primary training signal
1. Social Media Toxic Comment Dataset (English only: 16,554/49,998 rows)
   Slug: miadul/toxic-comments-detection-dataset → toxic-comments-detection/
   Kaggle: https://www.kaggle.com/datasets/miadul/toxic-comments-detection-dataset
   → 6 labels: toxic, severe_toxic, obscene, insult, threat, hate_speech
   → Filter: language == 'English' only (dataset also has Bangla/Mixed — discard)
   Label mapping to our 3 targets:
     toxic/severe_toxic/obscene/insult → label_toxicity=1
     insult/threat                     → label_harassment=1
     hate_speech                       → label_hate_racism=1, label_toxicity=1

🟠 CLEAN / SUPPLEMENTAL — no toxicity labels, used as label=0 negative examples
2. Open Assistant (39,283 English rows)
   Slug: thedevastator/multilingual-conversation-dataset → open-assistant/
   Kaggle: https://www.kaggle.com/datasets/thedevastator/multilingual-conversation-dataset
   → Has pre-computed detoxify scores — use rows with toxicity < 0.1 as clean negatives

3. OpenHermes GPT-4 (242k entries)
   Slug: thedevastator/gpt-4-ai-dataset-242k-entries → openhermes-gpt4/
   Kaggle: https://www.kaggle.com/datasets/thedevastator/gpt-4-ai-dataset-242k-entries
   → Clean AI text → all label=0

4. Databricks Dolly (15k)
   Slug: thedevastator/databricks-chatgpt-dataset → databricks-dolly/
   Kaggle: https://www.kaggle.com/datasets/thedevastator/databricks-chatgpt-dataset
   → Clean prompt-response → all label=0

5. YouTube US Comments (691,400 rows — unlabeled)
   Slug: soylevbeytullah/youtube-comments-datasets → yt-comments-soylevbeytullah/
   Kaggle: https://www.kaggle.com/datasets/soylevbeytullah/youtube-comments-datasets
   → USE: UScomments.csv only (video_id, comment_text, likes, replies)
   → Join: UScomments + USvideos + US_category_id.json → adds video category per comment
   → Role: label=0 negatives + YouTube-domain enrichment for dashboard
   → Skip: all non-US country CSVs (DE/FR/JP/KR/etc.) — not English

❌ DISCARDED:
- Das Racist YouTube (thedevastator/das-racist-you-oughta-know-youtube-comments)
  → Only 9KB, no actual comment text in the text column
- Atifaliak YouTube Comments (atifaliak/youtube-comments-dataset)
  → Sentiment labels (positive/neutral/negative) ≠ toxicity — not suitable

✅ All active datasets are English-only or filtered to English.
✅ Download script: src/data/download_datasets.py

🧠 Model Enhancements (NEW)
MC Dropout: Enabled at inference → T=10 forward passes → calculate epistemic uncertainty.
SHAP Values: Compute per-token importance for top 3 labels → stored in JSON.
Aleatoric Uncertainty: Estimated via predictive variance (if using probabilistic head).
Output Fields in BigQuery:
uncertainty_epistemic
uncertainty_aleatoric
shap_values
mc_dropout_samples