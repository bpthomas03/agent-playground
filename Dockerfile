FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Ensure Poetry is on PATH (installer puts it in /root/.local/bin)
ENV PATH="/root/.local/bin:${PATH}"

# Workdir
WORKDIR /app

# Copy project metadata and source
COPY pyproject.toml README.md ./ 
COPY src ./src

# Install dependencies and project
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

CMD ["python", "-m", "src.main"]
