#!/bin/bash
# ==========================================================
# YouTube Toxicity Detector - GCP + GitHub Actions Setup
# Adapted from CraigslistWebScraper pattern
# ==========================================================

# ==========================================================
# 🧩 EDITABLE VARIABLES - MODIFY THESE FIRST
# ==========================================================
PROJECT_ID="youtube-toxicity-detector"          # Change to your project ID
REGION="us-central1"                            # Change if needed (us-east1 also works)
BUCKET_NAME="youtube-toxicity-data-v1"         # Your data lake bucket name
GITHUB_REPO="https://github.com/dehiska/YoutubeCommentToxicityDetector"  # YOUR REPO

# Internal naming (keep unless you have strong reason to change)
WIF_POOL="github-pool"
WIF_PROVIDER="gh-actions"
RUNTIME_SA_ID="youtube-runtime"                 # Cloud Function runtime SA
DEPLOYER_SA_ID="youtube-deployer"              # GitHub Actions deployer SA
INGEST_FUNCTION_NAME="youtube-comments-ingest" # Ingestion function name
MODEL_FUNCTION_NAME="youtube-toxicity-scorer"  # Scoring function name

# Function deployment bodies (passed to GitHub Secrets)
INGEST_BODY='{}'
SCORER_BODY='{"batch_size": 32}'

# ==========================================================
echo "🚀 Starting YouTube Toxicity Detector GCP Setup"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Bucket: ${BUCKET_NAME}"
echo ""

# ==========================================================
# 1️⃣ SET GCP PROJECT & DERIVE VALUES
# ==========================================================
echo "⚙️ Setting project and deriving values..."
gcloud config set project "${PROJECT_ID}" >/dev/null

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
RUNTIME_SA="${RUNTIME_SA_ID}@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOYER_SA="${DEPLOYER_SA_ID}@${PROJECT_ID}.iam.gserviceaccount.com"
SCHED_SA="scheduler-invoker@${PROJECT_ID}.iam.gserviceaccount.com"
PRINCIPAL_SET="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL}/attribute.repository/${GITHUB_REPO}"

echo "✅ Project Number: ${PROJECT_NUMBER}"
echo ""

# ==========================================================
# 2️⃣ ENABLE REQUIRED APIs
# ==========================================================
echo "📡 Enabling GCP APIs..."
gcloud services enable \
  cloudfunctions.googleapis.com \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com \
  serviceusage.googleapis.com \
  cloudresourcemanager.googleapis.com \
  pubsub.googleapis.com \
  logging.googleapis.com \
  compute.googleapis.com \
  aiplatform.googleapis.com \
  bigquery.googleapis.com

echo "✅ APIs enabled"
echo ""

# ==========================================================
# 3️⃣ CREATE SERVICE ACCOUNTS
# ==========================================================
echo "👥 Creating service accounts..."
gcloud iam service-accounts create "${RUNTIME_SA_ID}" \
  --display-name="YouTube Ingest & Scoring Runtime" || echo "ℹ️ Runtime SA already exists"

gcloud iam service-accounts create "${DEPLOYER_SA_ID}" \
  --display-name="YouTube GitHub Deployer" || echo "ℹ️ Deployer SA already exists"

gcloud iam service-accounts create "scheduler-invoker" \
  --display-name="YouTube Scheduler Invoker" || echo "ℹ️ Scheduler SA already exists"

echo "✅ Service accounts created/verified"
echo ""

# ==========================================================
# 4️⃣ WORKLOAD IDENTITY FEDERATION (GitHub → GCP)
# ==========================================================
echo "🔗 Setting up Workload Identity Federation..."

gcloud iam workload-identity-pools create "${WIF_POOL}" \
  --location="global" \
  --display-name="GitHub Actions Pool" >/dev/null 2>&1 || echo "ℹ️ WIF Pool already exists"

gcloud iam workload-identity-pools providers create-oidc "${WIF_PROVIDER}" \
  --workload-identity-pool="${WIF_POOL}" \
  --location="global" \
  --display-name="GitHub OIDC Provider" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
  --attribute-condition="attribute.repository=='${GITHUB_REPO}' && attribute.ref=='refs/heads/main'" >/dev/null 2>&1 || echo "ℹ️ WIF Provider already exists"

echo "✅ Workload Identity Federation configured"
echo ""

# ==========================================================
# 5️⃣ GRANT GITHUB DEPLOYER ACCESS
# ==========================================================
echo "🔐 Binding GitHub Actions to Deployer SA..."
gcloud iam service-accounts add-iam-policy-binding "${DEPLOYER_SA}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="${PRINCIPAL_SET}" >/dev/null 2>&1 || echo "ℹ️ Binding already exists"

echo "✅ GitHub Actions can impersonate deployer SA"
echo ""

# ==========================================================
# 6️⃣ GRANT DEPLOYER PROJECT-LEVEL ROLES
# ==========================================================
echo "📋 Granting deployer project-level IAM roles..."
for ROLE in \
  roles/cloudfunctions.developer \
  roles/run.admin \
  roles/cloudscheduler.admin \
  roles/artifactregistry.writer \
  roles/serviceusage.serviceUsageAdmin \
  roles/aiplatform.admin \
  roles/bigquery.admin; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${DEPLOYER_SA}" \
    --role="${ROLE}" >/dev/null 2>&1 || echo "ℹ️ Role ${ROLE} already bound"
done

echo "✅ Deployer has project-level permissions"
echo ""

# ==========================================================
# 7️⃣ SA-TO-SA RELATIONSHIPS
# ==========================================================
echo "🔗 Configuring service account relationships..."

# Deployer can act as Runtime
gcloud iam service-accounts add-iam-policy-binding "${RUNTIME_SA}" \
  --member="serviceAccount:${DEPLOYER_SA}" \
  --role="roles/iam.serviceAccountUser" >/dev/null 2>&1 || echo "ℹ️ Binding already exists"

# Scheduler can mint OIDC tokens for Runtime
SCHEDULER_AGENT="service-${PROJECT_NUMBER}@gcp-sa-cloudscheduler.iam.gserviceaccount.com"
gcloud iam service-accounts add-iam-policy-binding "${RUNTIME_SA}" \
  --member="serviceAccount:${SCHEDULER_AGENT}" \
  --role="roles/iam.serviceAccountTokenCreator" >/dev/null 2>&1 || echo "ℹ️ Scheduler binding already exists"

# Deployer can act as default compute SA
gcloud iam service-accounts add-iam-policy-binding \
  "projects/-/serviceAccounts/${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --member="serviceAccount:${DEPLOYER_SA}" \
  --role="roles/iam.serviceAccountUser" >/dev/null 2>&1 || echo "ℹ️ Compute SA binding already exists"

echo "✅ Service account relationships configured"
echo ""

# ==========================================================
# 8️⃣ CREATE & CONFIGURE GCS BUCKET
# ==========================================================
echo "🪣 Creating GCS bucket..."

if ! gcloud storage buckets describe "gs://${BUCKET_NAME}" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --uniform-bucket-level-access
  echo "✅ Bucket created: gs://${BUCKET_NAME}"
else
  echo "ℹ️ Bucket already exists: gs://${BUCKET_NAME}"
fi
echo ""

# ==========================================================
# 9️⃣ GRANT RUNTIME SA ACCESS TO BUCKET
# ==========================================================
echo "🔐 Granting runtime SA bucket permissions..."

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/storage.objectAdmin" >/dev/null 2>&1 || echo "ℹ️ Permissions already set"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
  --member="serviceAccount:${DEPLOYER_SA}" \
  --role="roles/storage.objectViewer" >/dev/null 2>&1 || echo "ℹ️ Deployer viewer role already set"

echo "✅ Runtime SA has bucket access"
echo ""

# ==========================================================
# 🔟 VERTEX AI & BIGQUERY PERMISSIONS
# ==========================================================
echo "🤖 Granting Vertex AI & BigQuery permissions..."

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/aiplatform.user" >/dev/null 2>&1 || echo "ℹ️ Vertex AI role already bound"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/bigquery.dataEditor" >/dev/null 2>&1 || echo "ℹ️ BigQuery role already bound"

echo "✅ Vertex AI & BigQuery permissions granted"
echo ""

# ==========================================================
# 1️⃣1️⃣ VERIFY SETUP
# ==========================================================
echo "✅ Verifying setup..."
echo ""
echo "Service Accounts:"
gcloud iam service-accounts list --filter="email~${RUNTIME_SA_ID}|${DEPLOYER_SA_ID}" --format="table(email,displayName)"
echo ""

# ==========================================================
# 1️⃣2️⃣ GENERATE GITHUB ACTIONS VARIABLES
# ==========================================================
echo "=================================================="
echo "🔧 GITHUB ACTIONS VARIABLES"
echo "=================================================="
echo ""
echo "👉 Add these to: GitHub Repo → Settings → Secrets and variables → Actions → Variables"
echo ""

OUT="github_actions_variables_${PROJECT_ID}.txt"

{
  echo "WORKLOAD_IDENTITY_PROVIDER=projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL}/providers/${WIF_PROVIDER}"
  echo "DEPLOYER_SA=${DEPLOYER_SA}"
  echo "RUNTIME_SA=${RUNTIME_SA}"
  echo "PROJECT_ID=${PROJECT_ID}"
  echo "REGION=${REGION}"
  echo "BUCKET_NAME=${BUCKET_NAME}"
  echo "INGEST_FUNCTION_NAME=${INGEST_FUNCTION_NAME}"
  echo "MODEL_FUNCTION_NAME=${MODEL_FUNCTION_NAME}"
  echo "INGEST_BODY=${INGEST_BODY}"
  echo "SCORER_BODY=${SCORER_BODY}"
} | tee "${OUT}"

echo ""
echo "✅ Variables written to: ${OUT}"
echo "   Download and paste into GitHub Actions Secrets"
echo ""

# ==========================================================
# 1️⃣3️⃣ FINAL STATUS
# ==========================================================
echo "=================================================="
echo "✅ YouTube Toxicity Detector GCP Setup Complete!"
echo "=================================================="
echo ""
echo "📊 Summary:"
echo "   Project:  ${PROJECT_ID}"
echo "   Region:   ${REGION}"
echo "   Bucket:   gs://${BUCKET_NAME}"
echo "   Runtime SA: ${RUNTIME_SA}"
echo "   Deployer SA: ${DEPLOYER_SA}"
echo ""
echo "🚀 Next steps:"
echo "   1. Copy variables from ${OUT} to GitHub"
echo "   2. Clone/fork the project repo"
echo "   3. Create .github/workflows/ with deployment YAMLs"
echo "   4. Push to main and watch GitHub Actions deploy"
echo ""
