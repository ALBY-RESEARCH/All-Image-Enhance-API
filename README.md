# 🔧 All-Image-Enhance-API

> A plug-and-play API system for state-of-the-art Image Enhancement and Restoration models, designed to run on both GPU and CPU with optimized latency (<10ms target).

## 🎯 Project Goal

Build a modular API interface for image enhancement tasks (e.g., image enhancement, super-resolution, denoising, deblurring), wrapping existing models like:

- BSRGAN
- Real-ESRGAN
- ESRGAN+
- SwinIR
- DnCNN

Each model should be accessible via a unified API endpoint.

---

## 📦 Folder Structure (Recommended)

```

ALL-Enhance-API/
├── models/
│   ├── real\_esrgan/
│   └── bsrgan/
├── enhancer/
│   └── base.py         # Base class for loading and using models
├── api/
│   └── main.py         # FastAPI/Flask entrypoint
├── utils/
│   └── benchmark.py    # Measure latency
├── requirements.txt
├── README.md
└── run.sh              # Script to run the API

````

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-org/ALL-Enhance-API.git
cd ALL-Enhance-API
````

### 2. Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Run API Server

```bash
cd api
uvicorn main:app --reload  # For FastAPI
# or
python main.py             # For Flask
```

---

## 🖼️ API Usage

### POST `/enhance`

**Request:**

```json
{
  "image": "<base64-encoded image>",
  "model": "real-esrgan",
  "device": "cuda"   // or "cpu"
}
```

**Response:**

```json
{
  "output_image": "<base64-encoded enhanced image>",
  "latency_ms": 8.4
}
```

---

## ⚙️ Current Supported Models

| Model       | Task             | Notes                    |
| ----------- | ---------------- | ------------------------ |
| Real-ESRGAN | Super-resolution | Stable, fast             |
| BSRGAN      | Blind SR         | Handles real-world noise |
| SwinIR      | SR + Denoising   | Large, needs tuning      |

---

## 📊 Benchmarking

Use `utils/benchmark.py` to evaluate latency on different devices and image sizes.

```bash
python utils/benchmark.py --model real-esrgan --device cuda --size 512
```

---

## 📌 To-Do

* [ ] Add ONNX/TensorRT optimization support
* [ ] Add batch processing support
* [ ] Add async queueing with Redis (optional)
* [ ] Dockerize the API for deployment

---

## 🧠 Notes for Contributors

* Follow PEP8
* Keep model wrappers clean and modular
* Avoid hardcoded paths; use `config.yaml` or CLI args

