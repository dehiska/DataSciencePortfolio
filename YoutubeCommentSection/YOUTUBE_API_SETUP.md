# YouTube Data API v3 Setup Guide

## Step 1: Create a New GCP Project (or Use Existing)
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click **Select a Project** → **New Project**
3. Name it: `youtube-toxicity-detector`
4. Click **Create**
5. Wait for the project to be created (top notification bar)

## Step 2: Enable YouTube Data API v3
1. In Cloud Console, go to **APIs & Services** → **Library**
2. Search: `YouTube Data API v3`
3. Click on it and press **Enable**
4. Wait for confirmation

## Step 3: Create OAuth 2.0 Credential (Desktop App)
1. Go to **Credentials** (left sidebar)
2. Click **Create Credentials** → **OAuth client ID**
3. You'll see "Configure OAuth Consent Screen" - click **Configure Consent Screen**
   - Select **External** (for testing)
   - Click **Create**
   - Fill in:
     - **App name:** `YouTube Toxicity Detector`
     - **User support email:** your email
     - **Developer contact:** your email
     - Click **Save and Continue**
   - **Scopes:** Click **Add or Remove Scopes**
     - Search: `youtube.readonly`
     - Select it
     - Click **Update**
   - Click **Save and Continue** → **Back to Dashboard**

4. Now go back to **Credentials** → **Create Credentials** → **OAuth client ID**
5. Choose **Desktop application**
6. Name it: `youtube-scraper-desktop`
7. Click **Create**
8. **Download JSON** (click the download icon next to your credential)
   - Save as: `youtube_credentials.json` in your project root
   - ⚠️ **NEVER commit this to GitHub!**

## Step 4: Create API Key (For Public Data / Quota Monitoring)
1. Go to **Credentials** → **Create Credentials** → **API Key**
2. **Copy** the key immediately
3. Save it somewhere safe (or paste into `.env` file)

## Step 5: Configure API Quota & Restrictions
1. Go to **Credentials**
2. Click on your API Key (the one you just created)
3. Under **API restrictions**, select **YouTube Data API v3**
4. **Save**
5. Go to **APIs & Services** → **YouTube Data API v3** → **Quotas**
   - Your default quota: **10,000 units/day**
   - Each `commentThreads.list` call ≈ 2-3 units
   - Each video.list call ≈ 1 unit
   - Very budget-friendly!

## Step 6: Test Your Credentials
Run this Python snippet locally:

```python
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# OAuth 2.0 Setup (first time only)
SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']

def get_authenticated_service():
    flow = InstalledAppFlow.from_client_secrets_file(
        'youtube_credentials.json',
        SCOPES
    )
    # This will open a browser for you to authorize
    credentials = flow.run_local_server(port=8080)
    return build('youtube', 'v3', credentials=credentials)

# Test
youtube = get_authenticated_service()
request = youtube.channels().list(
    part='snippet',
    forUsername='YouTube'
)
response = request.execute()
print("✅ Connected! Channel name:", response['items'][0]['snippet']['title'])
```

## Step 7: Store Credentials Safely
Create a `.env` file (⚠️ add to `.gitignore`):
```
YOUTUBE_API_KEY=your_api_key_here
YOUTUBE_CLIENT_ID=your_client_id_here
YOUTUBE_CLIENT_SECRET=your_client_secret_here
```

Load in Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
```

## Step 8: For Production (Use Secret Manager in GCP)
Once deployed to Cloud Run, use **Secret Manager**:
```bash
echo -n "your_api_key" | gcloud secrets create youtube-api-key --data-file=-
gcloud secrets add-iam-policy-binding youtube-api-key \
  --member=serviceAccount:cf-runtime@project-id.iam.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor
```

---

## Quota Tips & Cost
- **FREE:** 10,000 quota units/day (plenty for development!)
- 1 video fetch ≈ 1 unit
- 1 comment thread fetch (up to 20 replies) ≈ 2-3 units
- **Estimate:** Can fetch ~100-150 videos + comments per day easily

You'll never exceed free tier with this setup. ✅
