# Research Agent UI

A modern chat interface for the Research Agent with monitoring capabilities.

## Features

- 💬 **Chatting Tab**: Interactive chat interface to communicate with the research agent
- 📊 **Monitoring Tab**: Observability dashboard (coming soon)

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Make sure the backend API server is running (see `agent/api_server.py`)

## Environment Variables

Create a `.env` file in the `ui` directory:

```env
VITE_API_URL=http://localhost:8000
```

## Development

- Frontend runs on `http://localhost:5173` (Vite default)
- Backend API should run on `http://localhost:8000`

## Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory.
