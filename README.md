# PDF to Sentences Extraction Service

A high-performance, production-ready microservice for extracting, de-hyphenating, and segmenting natural text sentences from PDF documents.

This project mirrors features a dual-engine pipeline capable of handling both digital vector PDFs and degraded historical scanned documents without requiring GPU/CUDA hardware.

---

## Architecture Overview

```
                      ┌──────────────────────────────────────┐
                      │             Uploaded PDF             │
                      └──────────────────┬───────────────────┘
                                         │
                         Has embedded digital text layer?
                                         │
                        ┌────────────────┴────────────────┐
                       YES                               NO
                        │                                 │
             ┌─────────────────────┐           ┌─────────────────────┐
             │   Fast-Path Engine  │           │   Vision OCR Engine │
             │     (pypdfium2)     │           │ (PP-OCRv5 + ONNX)   │
             │   ~0.02s per page   │           │    ~14s per page    │
             └──────────┬──────────┘           └──────────┬──────────┘
                        │                                 │
                        │                       Baseline Line-Merging
                        │                                 │
                        └────────────────┬────────────────┘
                                         │
                           ┌───────────────────────────┐
                           │   NLTK Sentence Tokenizer │
                           └─────────────┬─────────────┘
                                         │
                             JSON Extracted Sentences
```

### 1. Dual-Engine Extraction Strategy
- **Fast-Path Engine (`pypdfium2`)**: Inspects the document for an embedded vector text layer. If present, text is normalized (Unicode NFKC) and extracted in **~0.02s per page** (a 400x speedup over blind vision OCR).
- **Vision OCR Engine (PP-OCRv5 Mobile + ONNX Runtime)**: When no digital text exists (e.g. scanned typewriter reports or photocopies), fallback to PP-OCRv5 running on **ONNX Runtime CPU**.
  - **AVX2 SIMD CPU Acceleration**: Native C++ multithreading bypasses Python/PaddlePaddle interpreter bottlenecks.
  - **Optimized Resolution Scaling**: `PADDLE_PDX_PDF_RENDER_SCALE=1.0` avoids unnecessary 4x pixel oversampling on large scans.
  - **Geometric Baseline Line Merging**: Automatically clusters detected word bounding boxes sharing the same horizontal baseline ($Y_{\text{center}}$ within line-height tolerance) and sorts them left-to-right ($X_1 \to X_2$) before sentence tokenization.
  - **Scan Noise Filtering**: DBNet box threshold set to `0.60` to discard paper grain, wrinkles, and ink bleed.

### 2. Microservice Layout
```
pdf-to-sentences/
├── backend.py            # Core extraction API, inspect endpoint, async job worker
├── frontend.py           # Single-page web dashboard & API proxy
├── frontend_service.py   # Client communication layer, latency tracking, self-test logic
├── main.py               # Unified application entry point (mounts backend + frontend)
├── dataset.py            # Built-in regression test datasets & expected sentences
├── test_main.py          # Pytest automated test harness
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container image definition with pre-cached ONNX models
├── compose.yaml          # Multi-container orchestration specification
├── template/
│   └── index.html        # Responsive frontend template
├── static/
│   ├── app.js            # Frontend JavaScript (inspection, extraction, live timer)
│   └── styles.css        # Modern typography and styling
└── examples/
    ├── studyboard.pdf    # Vector text sample (Fast-Path test)
    ├── 2303.15133.pdf    # Academic paper sample (Course spec test)
    └── CAB_Accident_Report,_United_Air_Lines_Flight_2.pdf  # 1941 scan sample (OCR test)
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or 3.11
- Recommended virtual environment: **`MLvenv`**

### 1. Setup Virtual Environment
```powershell
# Create a dedicated venv
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

---

## How to Launch the Application

### Option A: Unified Web Application (Recommended)
Launches both the extraction backend and the frontend user interface on port 8000:
```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
- **Web UI**: Open your browser at [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **OpenAPI Schema**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

### Option B: Independent Backend & Frontend Services
Simulates production deployment where the backend API and frontend proxy run in distinct processes:

1. **Start the Backend Service** (port 8000):
   ```powershell
   uvicorn backend:app --reload --port 8000
   ```

2. **Start the Frontend Service** (port 8080):
   ```powershell
   $env:SENTENCE_SERVICE_URL="http://localhost:8000/v1/extract-sentences"
   uvicorn frontend:app --reload --port 8080
   ```
   Open the UI at [http://localhost:8080](http://localhost:8080).

---

### Option C: Docker Container
Build and run the self-contained container:
```bash
docker compose up --build
```
The application will be accessible at [http://localhost:8000](http://localhost:8000).

---

## API Endpoints Reference

### 1. Synchronous Extraction
- **Endpoint**: `POST /v1/extract-sentences`
- **Payload**: `multipart/form-data` with field `pdf_file`
- **Response**:
  ```json
  {
    "sentences": [
      "Finn and Tyge were present.",
      "The meeting adjourned at 14:00."
    ],
    "method": "fast_path",
    "page_count": 1,
    "processing_time_ms": 42.5
  }
  ```

### 2. PDF Pre-Inspection & Duration Estimation
- **Endpoint**: `POST /v1/inspect-pdf`
- **Payload**: `multipart/form-data` with field `pdf_file`
- **Response**:
  ```json
  {
    "page_count": 4,
    "has_text_layer": false,
    "recommended_method": "paddleocr_onnx",
    "estimated_seconds": 60.0
  }
  ```

### 3. Asynchronous Job Processing (For Large Scans)
For multi-page scanned documents that take extended processing time on CPU:
- **Submit Job**: `POST /v1/jobs/submit` $\to$ returns `{"job_id": "...", "status": "pending", "total_pages": 4, "estimated_seconds": 60.0}`
- **Query Job Status**: `GET /v1/jobs/{job_id}` $\to$ returns current status (`processing`, `completed`), progress fraction (`0.0` to `1.0`), current page, and final sentences upon completion.

---

## Running Automated Tests

Run the full automated test harness with `pytest`:
```powershell
pytest test_main.py -v
```

### Tests Covered
1. **`test_extract_sentences_studyboard`**: Validates instantaneous Fast-Path extraction on digital PDF (`studyboard.pdf`).
2. **`test_extract_sentences_assignment_spec`**: Validates sentence segmentation and accuracy against assignment benchmark (`2303.15133.pdf`).
3. **UI Self-Test**: Click the **Run self-test** button in the web dashboard at `http://localhost:8000` to execute live validation directly from the browser.
