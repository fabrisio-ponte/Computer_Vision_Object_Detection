# Face Detection with Age and Gender Analysis

A real-time face detection application that uses MediaPipe for face detection and ONNX models for age and gender prediction.

## Features

- Real-time face detection using MediaPipe
- Age estimation
- Gender classification
- Web interface for video streaming
- Efficient processing with background threads

## Requirements

- Python 3.9+
- OpenCV
- MediaPipe
- ONNX Runtime
- Flask
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fabrisio-ponte/face_stream_app.git
cd face_stream_app
```

2. Create and activate a virtual environment:
```bash
python -m venv my_env
source my_env/bin/activate  # On Windows: my_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the required models:
```bash
python download_models.py
```

## Usage

Run the application:
```bash
python app7.py
```

Access the web interface at `http://localhost:5013`

## Project Structure

- `app7.py`: Main application file
- `download_models.py`: Script to download and prepare ONNX models
- `templates/index7.html`: Web interface template
- `models/`: Directory for ONNX models

## License

MIT License 