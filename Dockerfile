FROM python:3.11-slim

EXPOSE 8000
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install only minimal runtime C libraries needed by ONNX Runtime and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements and ensure clean headless OpenCV without broken bindings
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true \
    && pip install --no-cache-dir opencv-python-headless \
    && rm -f /usr/local/lib/python3.11/site-packages/rapidocr_onnxruntime/models/ch_PP-OCRv4_*.onnx \
    && find /usr/local/lib/python3.11/site-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.11/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && rm -rf /root/.cache

# Pre-download NLTK tokenizers (preserve English and Danish, prune unused languages and zip archives)
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" \
    && rm -f /root/nltk_data/tokenizers/*.zip \
    && python -c "import shutil, pathlib; p = pathlib.Path('/root/nltk_data/tokenizers/punkt_tab'); [shutil.rmtree(d) for d in p.iterdir() if d.is_dir() and d.name not in ('english', 'danish')]" \
    && python -c "import pathlib; p = pathlib.Path('/root/nltk_data/tokenizers/punkt/PY3'); [f.unlink() for f in p.glob('*.pickle') if f.stem not in ('english', 'danish')]" 2>/dev/null || true

# Copy application source code
COPY . /app

# Pre-download PP-OCRv5 Mobile ONNX models and dictionary into /app/models for standalone execution
RUN python -c "from src.app.services.extraction_service import ensure_models_exist; ensure_models_exist('/app/models')" \
    && rm -rf /root/.cache /tmp/*

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


