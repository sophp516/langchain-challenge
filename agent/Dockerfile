FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install langgraph-cli globally
RUN pip install langgraph-cli[inmem]

# Copy application code
COPY . .

# Expose LangGraph server port (default is 2024)
EXPOSE 2024

# Start LangGraph server
CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "2024"]

