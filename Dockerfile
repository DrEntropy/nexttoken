FROM python:3.12-slim

WORKDIR /app

# Install Python deps — CPU-only torch to save ~2 GB
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    . gunicorn

# Copy app code
COPY demo.py .
COPY templates/ templates/

# Model weights download to this volume on first run
ENV HF_HOME=/data/huggingface
ENV NEXTTOKEN_DEVICE=cpu
EXPOSE 5005

CMD ["gunicorn", "--bind", "0.0.0.0:5005", "--workers", "1", \
     "--timeout", "120", "--preload", "demo:app"]
