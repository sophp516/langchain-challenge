# Deep Research Agent

A deep research agent built with LangGraph that accepts user queries and generates comprehensive reports grounded in web search results.

## Features

- **Multi-layer Research**: Parallel sub-researchers with dynamic research deepening
- **Persistent Storage**: MongoDB integration for report versioning
- **Source Quality Assessment**: Credibility scoring and domain-based filtering
- **Report Management**: Read, fetch, and update reports based on user feedback through tools
- **Configurable Agent**: Adjustable search depth, subtopics, and quality thresholds
- **Interactive UI**: Real-time chat interface with streaming responses and past conversation monitoring

## Architecture

- **Agent** (`agent/`): Python LangGraph agent with sub-researcher pattern
- **Backend** (`backend/`): Node.js API server for MongoDB operations
- **UI** (`ui/`): React + TypeScript frontend with real-time streaming

## Prerequisites

- Python
- Node.js
- MongoDB (I have attached the MongoDB uri to my submission email)
- API Keys:
  - `TAVILY_API_KEY` (or `EXA_API_KEY`)
  - OpenAI API key (for LLM)
  - Optional Gemini API key to activate evaluation features

## Setup

### 1. Agent Setup

```bash
cd agent

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and add:
# - TAVILY_API_KEY=your_key
# - EXA_API_KEY=your_key (optional)
# - OPENAI_API_KEY=your_key
# - MONGODB_URI=my_mongodb_connection_string
```

### 2. Backend Setup

```bash
cd backend

# Install dependencies
npm install

# Create .env file
echo "MONGODB_URI=your_mongodb_connection_string" > .env
echo "PORT=3001" >> .env

# Start backend server
npm start
```

### 3. UI Setup

```bash
cd ui

# Install dependencies
npm install

# Create .env file
echo "VITE_API_URL=http://localhost:8000" > .env

# Start development server
npm run dev
```

### 4. Start LangGraph Server

```bash
cd agent

# Start LangGraph server (exposes agent as API)
langgraph dev
```

The LangGraph server will run on `http://localhost:8000` by default.

## Running Locally

1. **Start MongoDB** (if using local instance):
   ```bash
   mongod
   ```

2. **Start Backend** (in `backend/` directory):
   ```bash
   npm start
   ```

3. **Start LangGraph Server** (in `agent/` directory):
   ```bash
   langgraph dev
   ```

4. **Start UI** (in `ui/` directory):
   ```bash
   npm run dev
   ```

5. **Open Browser**: Navigate to `http://localhost:5173`

## Usage

1. Enter a research query in the chat interface
2. The agent will:
   - Generate a research plan with subtopics
   - Execute parallel searches for each subtopic
   - Assess coverage and perform iterative deepening
   - Generate a comprehensive report with citations
3. View reports in the chat or retrieve them later using report IDs

## Configuration

Click the settings icon in the UI to configure:
- Search API (Tavily/Exa)
- Max search results per query
- Research depth (iterative deepening rounds)
- Number of subtopics
- Clarification rounds
- Source credibility thresholds

## Docker (Optional)

Each component has a Dockerfile. Use `docker-compose.yml` in the `agent/` directory for containerized deployment.

## Project Structure

```
langchain-challenge/
├── agent/              # LangGraph agent
│   ├── deep_researcher.py
│   ├── main.py         # CLI interface
│   ├── utils/
│   │   ├── nodes/      # Graph nodes
│   │   ├── subresearcher.py  # Sub-researcher graph
│   │   └── ...
│   └── requirements.txt
├── backend/           # Node.js API server
│   └── server.js
└── ui/                # React frontend
    └── src/
        └── components/
```

## API Endpoints

### LangGraph API (via `langgraph dev`)
- `POST /threads` - Create thread
- `POST /threads/{thread_id}/runs/stream` - Stream agent execution

### Backend API
- `GET /api/threads` - List all threads
- `GET /api/threads/:threadId/messages` - Get thread messages
- `POST /api/threads/:threadId/messages` - Save messages

## Troubleshooting

- **MongoDB Connection**: Ensure `MONGODB_URI` is set correctly
- **API Keys**: Verify all API keys are in `.env` files
- **Port Conflicts**: Default ports are 8000 (LangGraph), 3001 (Backend), 5173 (UI)
- **Rate Limits**: The agent includes rate limiting for search APIs

## License

MIT

