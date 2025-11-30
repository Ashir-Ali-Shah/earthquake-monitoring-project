FROM python:3.11-slim
# Set working directory
WORKDIR /app

# Install system dependencies required for building python packages
# (gcc for numpy/pandas, curl for healthchecks)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# This step will take a while due to TensorFlow/Torch
RUN pip install --no-cache-dir -r requirements.txt

# Download Spacy model during build so it's baked into the image
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code
COPY . .

# Create a volume mount point for model persistence
VOLUME /app/data

# Expose the port
EXPOSE 8000

ENV TF_USE_LEGACY_KERAS=1
# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]