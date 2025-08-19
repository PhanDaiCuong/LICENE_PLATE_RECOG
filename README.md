# License Plate Recognition System

A professional, modular system for automatic license plate recognition (ALPR) using deep learning.  
This repository provides a complete pipeline with a FastAPI backend for inference and a Streamlit frontend for user interaction.  
Supports image, video, and webcam input. Results include bounding boxes and recognized plate text.

---

## Features

- **YOLOv8-based detection** and ONNX OCR for robust plate recognition.
- **FastAPI backend**: REST API for image/video/webcam processing.
- **Streamlit frontend**: User-friendly web interface for uploading and viewing results.
- **Docker support**: Easy deployment.
- **GPU/CPU support**: Automatic device selection for ONNX models.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/licene-plate-recog.git
cd licene-plate-recog
```

### 2. Install dependencies

Recommended: Use Python 3.10+ and a virtual environment.

```bash
uv init
uv sync
source .venv/bin/activate
```


### 3. Prepare models

- Download YOLOv8 ONNX model and OCR ONNX model.
- Place them in `./models/` as specified in `configs/config.py`.

### 4. Run the backend

```bash
python app.py
```

### 5. Run the frontend

```bash
streamlit run streamlit.py
```

---

## Usage

- **Image**: Upload an image, view detected plates and bounding boxes.
- **Video**: Upload a video, view processed video with bounding boxes and recognized plates.
- **Webcam**: Capture a photo, view detection results.

---

## API Reference

### POST `/recognize/`

- **Parameters**:
  - `file`: Image or video file (multipart/form-data)
  - `input_type`: `"image"`, `"video"`, or `"webcam"`

- **Response**:
  - For images: JSON with bounding boxes and plate text.
  - For videos: JSON with results per frame (or processed video file if configured).
  - For webcam: JSON with detection results.

---

## Project Structure

```
.
├── app.py                # FastAPI backend
├── streamlit.py          # Streamlit frontend
├── domain/
│   └── licence_plate_recognition.py    #Domain 
├── configs/
│   └── config.py
├── models/               # ONNX models (not included)
├── requirements.txt      # Library requirement
├── pyproject.toml        # uv env
├── Dockerfile            # build docker
└── README.md             
```

---

## Docker

Build and run with Docker:

```bash
docker build -t licene-plate-recog .
docker run -p 8000:8000 licene-plate-recog
```

---

## Customization

- **Model paths**: Edit `configs/config.py` to change model locations.
- **Detection/recognition logic**: See `domain/licence_plate_recognition.py`.
- **Frontend UI**: Edit `streamlit.py`.

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or support, please open an issue or contact [phancuongtmtlhp@gmail.com]