FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure templates directory exists and is copied
# (The 'COPY . .' should handle this, but explicit check if needed)

# Expose port (HF Space uses 7860 by default for Docker spaces, but we can set it via ENV)
ENV PORT=7860
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
