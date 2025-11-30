FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# These libraries prevent the "Exit Code 139" crash
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]