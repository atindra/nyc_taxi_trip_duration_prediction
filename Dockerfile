FROM python:3.13-slim

WORKDIR /app

# Install dependencies first for Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and the serialized champion pipeline.
COPY src/ src/
COPY models/ models/

EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
