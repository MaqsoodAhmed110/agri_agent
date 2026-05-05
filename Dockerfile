# Use Python 3.11-slim for a small, secure base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies needed for Postgres and common Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching
# This ensures that changing code doesn't trigger a full pip install
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# We exclude files using .dockerignore to keep the image secret-free
COPY . .

# Create a non-root user for security
RUN useradd -m agriuser && chown -R agriuser:agriuser /app
USER agriuser

# Expose the API port
EXPOSE 8000

# Healthcheck to monitor service status
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Start the application using uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
