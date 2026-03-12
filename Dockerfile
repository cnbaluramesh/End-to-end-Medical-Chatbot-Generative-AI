FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf-cache
ENV TRANSFORMERS_CACHE=/app/hf-cache

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create cache directory and set permissions
RUN mkdir -p /app/hf-cache && \
    chmod -R 777 /app/hf-cache

# Create non-root user
RUN useradd -m appuser
USER appuser

# Expose port
EXPOSE 8080

CMD ["python", "app.py"]