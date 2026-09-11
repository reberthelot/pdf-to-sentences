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
              ┌─────────────────────┐           ┌────────────────────────┐
              │   Fast-Path Engine  │           │    Vision OCR Engine   │
              │     (pypdfium2)     │           │  (PP-OCRv5 Mobile ONNX)│
              │   ~0.02s per page   │           │     ~12s per page      │
              └──────────┬──────────┘           └──────────┬─────────────┘
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
- **Fast-Path Engine (`pypdfium2`)**: Inspects the document for an embedded vector text layer. If present, text is normalized (Unicode NFKC) and extracted in **~0.02s per page** (a 500x speedup over blind vision OCR).
- **Vision OCR Engine (PP-OCRv5 Mobile + RapidOCR ONNX Runtime)**: When no digital text exists (e.g. scanned typewriter reports or photocopies), the system falls back to the **PP-OCRv5 Mobile** vision model running natively on **ONNX Runtime CPU**.
  - **AVX2 SIMD CPU Acceleration**: Native C++ multithreading runs inference in ~11-12s per dense historical page without requiring GPU/CUDA hardware.
  - **Geometric Baseline Line Merging**: Automatically clusters detected word bounding boxes sharing the same horizontal baseline ($Y_{\text{center}}$ within line-height tolerance) and sorts them left-to-right ($X_1 \to X_2$) before sentence tokenization.
  - **Scan Noise Filtering**: DBNet box threshold set to `0.60` to discard paper grain, wrinkles, and ink bleed.

---

## Models Used Behind the Scenes & Why ONNX Runtime?

### 1. Specific Model Architecture
- **Text Detection**: **`PP-OCRv5_mobile_det`**
  - Architecture: Real-time Differentiable Binarization (`DBNet`) paired with a lightweight MobileNet backbone.
  - Role: Detects text bounding boxes at arbitrary angles, filtering out noise with a calibrated `box_thresh=0.60` and `unclip_ratio=1.5`.
- **Text Recognition**: **`en_PP-OCRv5_mobile_rec`**
  - Architecture: Lightweight `SVTR-LCNet` sequence recognition network.
  - Dictionary: Complete 436-character CTC vocabulary (alphanumeric, punctuation, symbols, fractions, and diacritics) matching the 438 CTC output logits.
- **Line Reconstruction**: An analytical geometric baseline-merging algorithm groups word boxes into cohesive horizontal lines prior to `nltk.sent_tokenize()`.

### 2. Why ONNX Runtime for Lightweight & High-Performance Serving?
1. **Zero Training Framework Overhead**:
   - Standard deep learning training frameworks package autograd, graph builders, optimizers, distributed training stubs, and large GPU/CUDA runtime libraries, easily bloating container images to **> 2.5 GB**.
   - For inference serving, none of this training machinery is needed. ONNX Runtime provides a dedicated, lightweight forward-pass evaluation engine (~20 MB) wrapped by `rapidocr-onnxruntime`.
2. **Graph-Level Optimizations**:
   - **Operator Fusion**: Fuses multiple adjacent operations (such as Convolution + Batch Normalization + ReLU) into single optimized compute kernels, eliminating redundant memory round-trips.
   - **Constant Folding & Memory Reuse**: Static model weights are folded and intermediate tensor allocations are reused throughout the execution graph.
3. **Native SIMD CPU Acceleration (AVX2 / FMA)**:
   - Compiles down to native CPU vector instruction sets (AVX2, FMA, SSE).
   - Bypasses Python interpreter locks and framework abstraction layers, cutting inference time on dense historical scans from ~88s down to **~11-12s per page**.
   - Pairs with headless OpenCV (`opencv-python-headless`) to avoid heavy X11/OpenGL system libraries.


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

### Option C: Docker Container (Optimized < 450 MB)
Build and run the lightweight self-contained container:
```bash
docker compose up --build
```
> **Note on Image Size**: The Docker image is optimized down from 2.56 GB to **< 450 MB** (~85% size reduction) by completely removing `paddlepaddle` (~1.5 GB) in favor of native `rapidocr-onnxruntime`, using headless OpenCV, pre-baking ONNX weights, and pruning non-essential caches.


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
    "recommended_method": "rapidocr_onnx",
    "estimated_seconds": 48.0
  }
  ```

### 3. Asynchronous Job Processing (For Large Scans & Non-Blocking UI)
For multi-page scanned documents or background batching, avoid HTTP client timeouts (504 Gateway Timeout) by using background job execution:

#### Submit Job
- **Endpoint**: `POST /v1/jobs/submit` (or via frontend proxy `POST /api/jobs/submit`)
- **Payload**: `multipart/form-data` with field `pdf_file`
- **Response**:
  ```json
  {
    "job_id": "cf525c49-a099-4738-9708-d5dde3fedd25",
    "status": "pending",
    "total_pages": 13,
    "estimated_seconds": 0.05
  }
  ```

#### Query Job Status & Progress
- **Endpoint**: `GET /v1/jobs/{job_id}` (or via frontend proxy `GET /api/jobs/{job_id}`)
- **Response (Processing)**:
  ```json
  {
    "job_id": "cf525c49-a099-4738-9708-d5dde3fedd25",
    "filename": "document.pdf",
    "status": "processing",
    "progress": 0.40,
    "current_page": 2,
    "total_pages": 5,
    "message": "Processing page 2 of 5 with RapidOCR ONNX...",
    "created_at": 1789145728.16,
    "updated_at": 1789145740.50,
    "result": null,
    "error": null
  }
  ```
- **Response (Completed)**:
  ```json
  {
    "job_id": "cf525c49-a099-4738-9708-d5dde3fedd25",
    "filename": "document.pdf",
    "status": "completed",
    "progress": 1.0,
    "current_page": 5,
    "total_pages": 5,
    "message": "Extraction completed successfully via PP-OCRv5 Mobile RapidOCR ONNX.",
    "result": {
      "sentences": ["First extracted sentence.", "Second sentence."],
      "method": "rapidocr_onnx",
      "page_count": 5,
      "processing_time_ms": 46290.9
    },
    "created_at": 1789145728.16,
    "updated_at": 1789145774.45,
    "error": null
  }
  ```

---

## Interactive Web Dashboard Features

The web frontend (`http://localhost:8000`) provides an integrated dual-mode workspace:

1. **Synchronous Extraction**: Click **Extract sentences** for instant digital extraction or single-page scans with live stopwatch and preview.
2. **Asynchronous Jobs Table**: Click **Extract sentences (asynchrone)** to offload tasks to the background worker.
   - **Compact Single-Row Table**: Displays document name, short Job ID, frozen elapsed duration, page count (`current / total`), status pill, progress bar percentage, and OCR engine.
   - **Color-Coded Status**: Green for completed, amber/orange for pending/processing, and red for failed jobs.
   - **Expandable Drawer**: Right-aligned `View ▼` / `Hide ▲` toggle reveals the complete Job ID, status log, latency, and full extracted sentences with a one-click `.txt` download button.
   - **Stable Scrolling**: Scroll position is preserved inside text drawers during background polling, preventing abrupt page jumps.
3. **Automated Self-Test**: Validates baseline sentence segmentation against reference benchmark PDFs directly from the UI.
4. **Operational KPIs**: Live request counters, success/failure ratios, and average latency tracking.

---

## Running Automated Tests

Run the full automated test harness with `pytest`:
```powershell
pytest test_main.py -v
```

### Tests Covered
1. **`test_extract_sentences_studyboard`**: Validates instantaneous Fast-Path extraction on digital PDF (`studyboard.pdf`).
2. **`test_extract_sentences_assignment_spec`**: Validates sentence segmentation and accuracy against assignment benchmark (`2303.15133.pdf`).
3. **`test_async_job_lifecycle`**: Validates asynchronous submission (`/v1/jobs/submit`) and polling resolution (`/v1/jobs/{job_id}`).
4. **UI Self-Test**: Click the **Run self-test** button in the web dashboard at `http://localhost:8000` to execute live validation directly from the browser.

