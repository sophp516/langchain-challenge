# Research Agent UI

A modern chat interface for the Research Agent with monitoring capabilities.

> **Note**: For full project setup instructions, see the [main README](../README.md).

## Features

- 💬 **Chatting Tab**: Interactive chat interface to communicate with the research agent
  - Real-time streaming responses via LangGraph SDK
  - Markdown rendering for reports
  - Thread management
  - Agent configuration via settings modal
- 📊 **Monitoring Tab**: Observability dashboard that shows conversational history (threads)
  - View all conversation threads
  - Message history per thread
  - Thread management (create, delete)

## Quick Start

1. **Prerequisites**: Ensure LangGraph server (port 8000) and Backend API (port 3001) are running (see [main README](../README.md))

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Create `.env` file**:
   ```env
   VITE_API_URL=http://localhost:8000
   ```

4. **Start development server**:
   ```bash
   npm run dev
   ```

5. **Open browser**: Navigate to `http://localhost:5173`

## Development

- **Frontend**: `http://localhost:5173` (Vite default)
- **LangGraph API**: `http://localhost:8000` (agent API)
- **Backend API**: `http://localhost:3001` (MongoDB operations)

## Configuration

Click the settings icon (⚙️) in the UI to configure the agent:

- **Search API**: Choose between Tavily or Exa
- **Max Search Results**: Number of results per query (1-20)
- **Max Research Depth**: Iterative deepening rounds (1-5)
- **Max Subtopics**: Number of research subtopics (3-10)
- **Clarification Rounds**: Number of clarification questions (0-10)
- **Source Credibility**: Minimum credibility threshold (0.0-1.0)

## Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Tech Stack

- **React 18** with TypeScript
- **Vite** for build tooling
- **LangGraph SDK** for agent communication
- **React Markdown** for report rendering
- **Lucide React** for icons

## Project Structure

```
ui/
├── src/
│   ├── components/
│   │   ├── ChatTab.tsx          # Main chat interface
│   │   ├── MonitoringTab.tsx    # Thread monitoring
│   │   └── SettingsModal.tsx    # Agent configuration
│   ├── services/
│   │   └── db.ts                # Backend API client
│   └── App.tsx                  # Main app component
└── package.json
```
