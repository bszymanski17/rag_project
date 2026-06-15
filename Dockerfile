FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source code
COPY app.py .
COPY config/ config/
COPY prompts/ prompts/
COPY src/ src/
COPY schemas/ schemas/

RUN mkdir -p data/interim_images \
    databases \
    knowledge \
    .byaldi

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]