FROM python:3.11-slim

EXPOSE 8000

# Install system dependencies required for OpenCV and PaddlePaddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK tokenizer resources
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Pre-download lightweight PP-OCRv5 Mobile ONNX models to accelerate container startup
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(text_detection_model_name='PP-OCRv5_mobile_det', text_recognition_model_name='en_PP-OCRv5_mobile_rec', engine='onnxruntime', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)"

COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

