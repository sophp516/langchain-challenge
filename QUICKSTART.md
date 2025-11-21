# Quick Start Guide

## Running the Research Agent with UI

### 1. Backend Setup

Navigate to the `agent` directory:

```bash
cd agent
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set environment variables:

```bash
export OPENAI_API_KEY=your_openai_key
export TAVILY_API_KEY=your_tavily_key
```

Start the API server:

```bash
python api_server.py
```

The API will run on `http://localhost:8000`

### 2. Frontend Setup

In a new terminal, navigate to the `ui` directory:

```bash
cd ui
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The UI will run on `http://localhost:5173`

### 3. Using the Application

1. Open your browser to `http://localhost:5173`
2. You'll see two tabs in the vertical navbar:
   - **Chatting**: Interactive chat with the research agent
   - **Monitoring**: Observability dashboard (empty for now)
3. Type a research question in the chat input and press Enter
4. The agent will process your request and return a comprehensive research report

### Features

- **Vertical Navigation**: Switch between Chatting and Monitoring tabs
- **Real-time Chat**: Send messages and receive responses from the agent
- **Loading States**: Visual feedback while the agent processes requests
- **Responsive Design**: Works on desktop and mobile devices

### Troubleshooting

- **CORS Errors**: Make sure the backend is running on port 8000 and frontend on 5173
- **API Connection**: Check that `VITE_API_URL` in `ui/.env` matches your backend URL
- **Agent Errors**: Check that all environment variables are set correctly


