# ContextPilot Cloud Deployment Guide

This guide walks you through deploying ContextPilot to Google Cloud (Cloud Run + Firebase).

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Firebase Project** (can be created during setup)
3. **API Keys**:
   - Google AI API Key (for embeddings)
   - Pinecone API Key (for vectors)
   - Firecrawl API Key (for crawling)

## Tools Required

Install the following CLI tools:

```bash
# Firebase CLI
npm install -g firebase-tools

# Google Cloud SDK
# Download from: https://cloud.google.com/sdk/docs/install
```

## Step 1: Create Firebase Project

```bash
# Login to Firebase
firebase login

# Create new project (or use existing)
firebase projects:create contextpilot-prod

# Select the project
firebase use contextpilot-prod
```

## Step 2: Enable Required APIs

```bash
# Enable Cloud Run API
gcloud services enable run.googleapis.com

# Enable Firestore
gcloud services enable firestore.googleapis.com

# Enable Secret Manager
gcloud services enable secretmanager.googleapis.com

# Enable Container Registry
gcloud services enable containerregistry.googleapis.com
```

## Step 3: Create Secrets

Store your API keys securely:

```bash
# Create secrets in Secret Manager
echo -n "your-google-api-key" | gcloud secrets create google-api-key --data-file=-
echo -n "your-pinecone-api-key" | gcloud secrets create pinecone-api-key --data-file=-
echo -n "your-firecrawl-api-key" | gcloud secrets create firecrawl-api-key --data-file=-
echo -n "your-contextpilot-api-key" | gcloud secrets create contextpilot-api-key --data-file=-

# Grant Cloud Run access to secrets
gcloud secrets add-iam-policy-binding google-api-key \
    --member="serviceAccount:YOUR_PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

## Step 4: Deploy Backend to Cloud Run

### Option A: Using gcloud (Manual)

```bash
# Build and deploy directly
gcloud run deploy contextpilot-api \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300s \
  --set-env-vars "FIREBASE_PROJECT_ID=$PROJECT_ID" \
  --set-secrets "GOOGLE_API_KEY=google-api-key:latest,PINECONE_API_KEY=pinecone-api-key:latest,FIRECRAWL_API_KEY=firecrawl-api-key:latest,CONTEXTPILOT_API_KEY=contextpilot-api-key:latest"
```

### Option B: Using Cloud Build (CI/CD)

```bash
# Submit build
gcloud builds submit --config=cloudbuild.yaml
```

Note the Cloud Run URL (e.g., `https://contextpilot-api-xxxxx-xx.a.run.app`)

## Step 5: Initialize Firestore

```bash
# Create Firestore database
gcloud firestore databases create --location=us-central

# Deploy security rules
firebase deploy --only firestore:rules

# Deploy indexes
firebase deploy --only firestore:indexes
```

## Step 6: Deploy Frontend

```bash
# Build the frontend
cd frontend
NEXT_PUBLIC_API_URL=https://YOUR_CLOUD_RUN_URL npm run build

# Deploy to Firebase Hosting
cd ..
firebase deploy --only hosting
```

Your dashboard will be available at: `https://YOUR_PROJECT.web.app`

## Step 7: Configure Cursor MCP

Update your Cursor MCP configuration to use the cloud API:

**~/.cursor/mcp.json**:
```json
{
  "mcpServers": {
    "contextpilot": {
      "command": "python",
      "args": ["/path/to/contextpilot/mcp_remote_client.py"],
      "env": {
        "CONTEXTPILOT_API_URL": "https://contextpilot-api-xxxxx-xx.a.run.app",
        "CONTEXTPILOT_API_KEY": "your-contextpilot-api-key"
      }
    }
  }
}
```

Restart Cursor for changes to take effect.

## Verification

Test the deployment:

```bash
# Health check
curl https://YOUR_CLOUD_RUN_URL/health

# Search (if you have indexed docs)
curl -X POST https://YOUR_CLOUD_RUN_URL/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Gemini API"}'
```

## Cost Estimation

Approximate monthly costs for light usage:

| Service | Estimated Cost |
|---------|----------------|
| Cloud Run | $0-5 (pay per use, free tier available) |
| Firestore | $0-1 (free tier: 50K reads, 20K writes/day) |
| Firebase Hosting | $0 (free tier: 10GB storage, 360MB/day transfer) |
| Pinecone | $0-20 (depends on plan) |
| Total | ~$0-25/month |

## Troubleshooting

### Cloud Run fails to start
- Check logs: `gcloud run logs read contextpilot-api`
- Verify secrets are accessible
- Check environment variables

### Firestore permission denied
- Verify Firestore rules are deployed
- Check service account permissions

### Frontend can't connect to API
- Verify CORS is enabled (it is by default)
- Check NEXT_PUBLIC_API_URL is correct
- Verify API key if authentication is enabled

## Maintenance

### Update Backend
```bash
gcloud run deploy contextpilot-api --source .
```

### Update Frontend
```bash
cd frontend && npm run build && cd .. && firebase deploy --only hosting
```

### View Logs
```bash
# Cloud Run logs
gcloud run logs read contextpilot-api

# Firebase Hosting logs
firebase functions:log
```

