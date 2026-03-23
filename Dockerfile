FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Copy source
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
