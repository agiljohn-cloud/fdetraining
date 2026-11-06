FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY meal_concierge/ ./meal_concierge/
COPY evals/ ./evals/
COPY tests/ ./tests/
COPY pyproject.toml README.md ./

# Run static secret scan, unit tests, and golden evaluation suite at build time
RUN python3 -m unittest discover -s tests -v && \
    python3 -m evals.evaluate_agent

# Launch Google ADK API Server on Cloud Run port (parent dir of `meal_concierge`)
CMD ["adk", "api_server", "--host", "0.0.0.0", "--port", "8080", "."]
