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

### Agent Workflow

```
User Query
    ↓
Intent Classification (research vs. report retrieval/revision)
    ↓
[If Research]
    ↓
Topic Inquiry & Clarification (optional)
    ↓
Research Planning (generate subtopics + search queries)
    ↓
Parallel Sub-Researchers (each subtopic researched independently)
    ├─→ Execute Searches (with rate limiting)
    ├─→ Coverage Assessment
    ├─→ Adaptive Deepening (if needed)
    ├─→ Source Quality Filtering
    └─→ Summarize Findings
    ↓
Report Generation (combine all findings into comprehensive report)
    ↓
Save to MongoDB & Return to User

[If Report Retrieval/Management]
    ↓
Tool Execution (get_report, list_reports, revise_report)
    ↓
Return Formatted Results
```

## Prerequisites

- **Python+**
- **Node.js**
- **MongoDB** (local instance or Atlas connection string)
- **API Keys**:
  - `TAVILY_API_KEY` (or `EXA_API_KEY`) - Web search API
  - `OPENAI_API_KEY` - For LLM (GPT-4o for quality tasks, GPT-4o-mini for simple tasks)
  - `GEMINI_API_KEY` (optional) - For evaluation features

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
   # Edit .env and add your actual API keys and MongoDB URI
   ```

2. **Backend Setup**:
   ```bash
   cd backend
   npm install
   cp .env.example .env
   # Edit .env and add your MongoDB URI
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

### Basic Workflow

1. **Enter a research query** in the chat interface (e.g., "Research the impact of AI on software development")
2. **Agent processing** (automatic):
   - Generates a research plan with thematic subtopics
   - Executes parallel web searches for each subtopic
   - Assesses coverage and performs adaptive deepening if needed
   - Generates a comprehensive report with citations
3. **View the report** in the chat interface with clickable source links
4. **Manage reports**:
   - Retrieve past reports: "show me report abc123"
   - List all reports: "list all reports"
   - Revise reports: "revise report abc123 with additional information about X"

### Example Queries

- "Research the best MMORPG games coming out in 2026"
- "Compare investment philosophies of Warren Buffett vs Charlie Munger"
- "Analyze the impact of AI on K-12 education"
- "Research gut microbiota and its relationship to cancer"

## Configuration

Click the **Settings** icon (⚙️) in the UI to configure the agent:

### Search Settings
- **Search API**: Choose between Tavily or Exa
- **Max Search Results**: Number of results per query (1-20, default: 5)

### Research Depth
- **Max Research Depth**: Adaptive deepening rounds (1-10, default: 5)
- **Max Subtopics**: Number of research subtopics (3-10, default: 7)

### Interaction
- **Max Clarification Rounds**: Number of clarification questions (0-10, default: 0)

### Quality
- **Min Credibility Score**: Source quality threshold (0.0-1.0, default: 0.3)

### Feature Toggles
- **Enable User Feedback**: Allow feedback loops for report improvement
- **Enable Cross-Reference Verification**: Cross-check information across sources

Configuration is saved to browser localStorage and persists across sessions.


## Performance Optimization & Design Decisions

### Optimization Strategy

During development, my primary focus was to **reduce API calls and LLM invocations** while **maintaining or improving research quality**. Key optimizations include:

1. **Combined LLM Calls**: Merged topic evaluation and clarification question generation into a single structured output call, reducing costs by ~50% in clarification loops
2. **Pre-generated Research Plans**: Research queries are generated during outline creation, eliminating redundant LLM calls per section
3. **Parallel Execution**: All sub-researchers execute searches in parallel for maximum efficiency
4. **Shared Research Pool**: Cross-section learning prevents redundant research across subtopics
5. **Direct Tool Calls**: Report retrieval tools use direct function calls instead of LLM routing.

### Evaluation & Testing

During development, I used a Gemini-based LLM-as-a-judge evaluation node to iteratively improve research quality. I attempted to benchmark against the [Deep Research Bench](https://github.com/langchain-ai/deep-research-bench) repository following LangChain's evaluation methodology, but was limited by time and API token constraints.

**Future Improvements**: If time permits, I would like to:
- Implement comprehensive evaluation metrics (citation accuracy, coverage scores, source credibility)
- Benchmark against Deep Research Bench test cases


## Project Structure

```
langchain-challenge/
├── agent/                    # LangGraph agent (Python)
│   ├── deep_researcher.py    # Main agent graph definition
│   ├── main.py               # CLI interface for testing
│   ├── langgraph.json        # LangGraph server configuration
│   ├── requirements.txt      # Python dependencies
│   ├── docker-compose.yml    # Docker Compose configuration
│   ├── Dockerfile            # Agent container definition
│   └── utils/
│       ├── state.py          # Unified state definition
│       ├── configuration.py  # Agent configuration & API clients
│       ├── model.py          # LLM initialization
│       ├── db.py             # MongoDB operations
│       ├── subresearcher.py  # Sub-researcher graph (parallel research)
│       ├── tools.py          # Report management tools
│       ├── edges.py          # Graph routing logic
│       └── nodes/
│           ├── user_intent.py    # Intent classification & topic inquiry
│           ├── writing.py        # Report generation
│           ├── tools.py          # Tool execution
│           └── helpers.py        # Utility functions
├── backend/                  # Node.js API server
│   ├── server.js             # Express server for MongoDB operations
│   ├── package.json
│   └── Dockerfile
└── ui/                       # React + TypeScript frontend
    ├── src/
    │   ├── components/
    │   │   ├── ChatTab.tsx           # Main chat interface
    │   │   ├── MonitoringTab.tsx    # Thread monitoring
    │   │   └── SettingsModal.tsx    # Agent configuration
    │   ├── services/
    │   │   ├── db.ts                # Backend API client
    │   │   └── formatter.ts         # URL formatting utilities
    │   └── App.tsx                  # Main app component
    ├── package.json
    └── Dockerfile
```


## Key Design Decisions

### Architecture Choices

1. **Sub-Researcher Pattern**: Parallel sub-researchers handle different subtopics simultaneously, significantly reducing research time
2. **Adaptive Deepening**: Coverage assessment dynamically determines if additional research is needed, avoiding unnecessary API calls
3. **Intent Routing**: Separate flows for new research vs. report retrieval/management, improving user experience
4. **Source Credibility Scoring**: Domain-based quality filtering before expensive processing, improving efficiency

### Technology Stack

- **LangGraph**: State machine orchestration with interrupt/resume support
- **Tavily/Exa**: Web search APIs with rate limiting
- **MongoDB**: Persistent storage for reports and conversation threads
- **React + TypeScript**: Type-safe frontend with real-time streaming
- **Docker Compose**: Containerized deployment for production readiness

## Troubleshooting

### Common Issues

- **MongoDB Connection**: Ensure `MONGODB_URI` is set correctly in both `agent/.env` and `backend/.env`
- **API Keys**: Verify all required API keys are in `.env` files (check error messages for missing keys)
- **Port Conflicts**: Default ports are:
  - `8000` - LangGraph server (dev mode)
  - `2024` - LangGraph server (Docker)
  - `3001` - Backend API
  - `5173` - UI (Vite dev server)
- **Rate Limits**: The agent includes automatic rate limiting for search APIs. If you hit limits:
  - Reduce `max_search_results` in settings
  - Reduce `max_research_depth`
  - Reduce `max_subtopics`
- **LangGraph Server Not Starting**: Ensure you're in the `agent/` directory and have installed dependencies with `pip install -r requirements.txt`
