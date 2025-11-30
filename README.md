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

## Running the Project

There are two ways to run the project: **Development Mode** (local) or **Docker Compose** (containerized).

### Option 1: Development Mode

This mode runs each service locally. It provides the option to run the agent directly through the terminal or the development mode UI.

#### Setup

1. **Agent Setup**:
   ```bash
   cd agent
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env and add your API keys:
   # - TAVILY_API_KEY=your_key
   # - EXA_API_KEY=your_key (optional)
   # - OPENAI_API_KEY=your_key
   # - MONGODB_URI=your_mongodb_connection_string
   ```

2. **Backend Setup**:
   ```bash
   cd backend
   npm install
   echo "MONGODB_URI=your_mongodb_connection_string" > .env
   echo "PORT=3001" >> .env
   ```

3. **UI Setup**:
   ```bash
   cd ui
   npm install
   echo "VITE_API_URL=http://localhost:8000" > .env
   ```

#### Running

Start each service in separate terminals:

1. **Start Backend** (Terminal 1):
   ```bash
   cd backend
   npm start
   ```

2. **Start LangGraph Server** (Terminal 2):
   ```bash
   cd agent
   langgraph dev
   ```
   Runs on `http://localhost:8000` by default.

3. **Start UI** (Terminal 3):
   ```bash
   cd ui
   npm run dev
   ```
   Runs on `http://localhost:5173` by default.

4. **Open Browser**: Navigate to `http://localhost:5173`

### Option 2: Docker Compose

This mode runs all services in containers using Docker Compose.

#### Prerequisites
- Docker and Docker Compose installed
- All API keys and MongoDB URI configured

#### Setup

1. **Create `.env` file in `agent/` directory**:
   ```bash
   cd agent
   cp .env.example .env
   # Edit .env with your API keys and MongoDB URI
   ```

2. **Start all services**:
   ```bash
   cd agent
   docker-compose up
   ```

   Or run in detached mode:
   ```bash
   docker-compose up -d
   ```

#### Services

- **Backend**: `http://localhost:3001`
- **Agent (LangGraph)**: `http://localhost:2024`
- **UI**: `http://localhost:5173`

#### Docker Compose Commands

```bash
cd agent
# Start services
docker-compose up
```

## Usage

1. Enter a research query in the chat interface
2. The agent will:
   - Generate a research plan with subtopics
   - Execute parallel searches for each subtopic
   - Assess coverage and perform adaptive deepening research
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


## Troubleshooting

- **MongoDB Connection**: Ensure `MONGODB_URI` is set correctly
- **API Keys**: Verify all API keys are in `.env` files
- **Port Conflicts**: Default ports are 8000 (LangGraph), 3001 (Backend), 5173 (UI)
- **Rate Limits**: The agent includes rate limiting for search APIs
