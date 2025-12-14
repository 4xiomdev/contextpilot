#!/bin/bash
# ContextPilot Deployment Script
# Deploys backend to Cloud Run and frontend to Firebase Hosting

set -e

echo "🚀 ContextPilot Deployment Script"
echo "================================="

# Check prerequisites
command -v gcloud >/dev/null 2>&1 || { echo "❌ gcloud CLI not installed"; exit 1; }
command -v firebase >/dev/null 2>&1 || { echo "❌ Firebase CLI not installed"; exit 1; }

# Get project info
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo "❌ No GCP project selected. Run: gcloud config set project YOUR_PROJECT"
    exit 1
fi

echo "📋 Project: $PROJECT_ID"

# Confirm deployment
read -p "Deploy to $PROJECT_ID? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Deploy backend
echo ""
echo "🔧 Deploying backend to Cloud Run..."
gcloud run deploy contextpilot-api \
    --source . \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --timeout 300s \
    --set-env-vars "FIREBASE_PROJECT_ID=$PROJECT_ID"

# Get Cloud Run URL
CLOUD_RUN_URL=$(gcloud run services describe contextpilot-api --region us-central1 --format='value(status.url)')
echo "✅ Backend deployed: $CLOUD_RUN_URL"

# Build frontend
echo ""
echo "🏗️  Building frontend..."
cd frontend
NEXT_PUBLIC_API_URL=$CLOUD_RUN_URL npm run build
cd ..

# Deploy frontend
echo ""
echo "🌐 Deploying frontend to Firebase Hosting..."
firebase deploy --only hosting

# Get Firebase URL
FIREBASE_URL="https://$PROJECT_ID.web.app"
echo "✅ Frontend deployed: $FIREBASE_URL"

echo ""
echo "🎉 Deployment complete!"
echo "========================"
echo "Dashboard: $FIREBASE_URL"
echo "API: $CLOUD_RUN_URL"
echo ""
echo "📝 Update your Cursor MCP config (~/.cursor/mcp.json):"
echo "   CONTEXTPILOT_API_URL: $CLOUD_RUN_URL"

