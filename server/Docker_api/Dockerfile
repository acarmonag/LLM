FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make init script executable
RUN chmod +x init.sh

# Default environment variables
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=ollama
ENV OLLAMA_PORT=11434


EXPOSE 8002

CMD ["./init.sh"]