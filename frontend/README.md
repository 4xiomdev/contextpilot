# ContextPilot Dashboard

React/Next.js dashboard for ContextPilot - the open-source context augmentation layer.

## Development

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to see the dashboard.

## Environment Variables

Create a `.env.local` file:

```bash
# Backend API URL (default: http://localhost:8000)
NEXT_PUBLIC_API_URL=http://localhost:8000

# For cloud deployment:
# NEXT_PUBLIC_API_URL=https://contextpilot-api-xxxxx.run.app

# API Key (optional, for authenticated endpoints)
NEXT_PUBLIC_API_KEY=your-api-key
```

## Build for Production

```bash
npm run build
```

This creates a static export in the `out/` directory.

## Deploy to Firebase Hosting

```bash
# Make sure you're in the frontend directory
npm run build

# From the project root
cd ..
firebase deploy --only hosting
```

## Features

- **Dashboard**: Overview of indexed sources, vectors, and crawl status
- **Crawl Manager**: Add URLs to crawl, view job history
- **Search**: Semantic search across indexed documentation
- **Normalized Docs**: View and create synthesized documents

## Tech Stack

- [Next.js 14](https://nextjs.org/) - React framework
- [Tailwind CSS](https://tailwindcss.com/) - Styling
- [shadcn/ui](https://ui.shadcn.com/) - UI components
- [React Query](https://tanstack.com/query) - Data fetching
