This project is a comprehensive, end-to-end system for detecting deepfakes in multiple modalities: images, videos, and audio. It uses a powerful ensemble of deep learning models and exposes the detection capabilities through a simple and fast REST API.
Though not completely perfect, especially on modern, high quality deepfakes, it does showcases the fundamentals quality of models and importance of generalization in models to work better on "in-the-wild" deepfakes.

## Features

-   **Image Deepfake Detection:** Analyzes single images to detect facial manipulations.
-   **Video Deepfake Detection:** Processes videos on a frame-by-frame basis to identify deepfakes.
-   **Audio Deepfake Detection:** Detects synthesized or voice-converted speech by analyzing audio spectrograms.
-   **REST API:** A backend built with FastAPI to easily integrate the detection models into other applications.

## Tech Stack

-   **Deep Learning:** PyTorch, `timm` (PyTorch Image Models)
-   **Models:** EfficientNet, XceptionNet, ConvNeXT, Swin (Ensembled for Images) & Vision Transformer (ViT) (For Audio)
-   **Data Processing:** OpenCV, Librosa, Pillow, MTCNN (for face detection)
-   **Backend:** FastAPI, Uvicorn
-   **Frontend:** React/Streamlit 
-   **Data Science:** NumPy, scikit-learn

## Project Structure

```
.
├── DeepFake_App/
│   ├── backend/
│   │   ├── models/
│   │   │   ├── best_photo_detector.pth
│   │   │   └── vit_audio_best.pth
│   │   ├── utils/
│   │   │   └── face_crop.py
│   │   └── main.py
│   └── frontend/
│       └── frontend_react/
│       │   └── src/
│       │       └── App.jsx
│       └── app.py
├── notebooks/
│   ├── 1_Image_Video_Training.ipynb
│   └── 2_Audio_Training.ipynb
├── test_wild_audios/
├── test_wild_images/
├── README.md
└── requirements.txt
```

## Setup and Installation

**1. Clone the repository:**
```bash
git clone [https://github.com/AryakumarMishra/multimodal-deepfake-detector](https://github.com/AryakumarMishra/multimodal-deepfake-detector)
cd your-repo-name
```

**2. Create and activate a virtual environment:**
```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install the required packages:**
> **Note:** Make sure you have the correct PyTorch version with CUDA support installed for your system. See the [PyTorch website](https://pytorch.org/get-started/locally/) for instructions.

```bash
pip install -r requirements.txt
```

```bash
cd DeepFake_App/frontend/frontend_react
npm install
```

## How to Run

### Training
The training process for both the image/video and audio models is detailed in the Jupyter notebooks located in the `/notebooks` directory. You will need to download the respective datasets (Celeb-DF, FaceForensics++, ASVspoof 2019) to run the training from scratch.

### Running the Backend API
Once you have the trained models in the `backend/models/` folder, you can launch the FastAPI server.

Navigate to the `backend` directory and run:
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```
The API will now be live and accessible at `http://localhost:8000/docs`.

Navigate to the `frontend` directory (if not already) and run:
```bash
cd DeepFake_App/frontend/frontend_react
npm run dev
```

## API Endpoints

-   `POST /predict/photo`: Upload an image file (`.jpg`, `.png`) to detect deepfakes.
-   `POST /predict/video`: Upload a video file (`.mp4`) to perform frame-by-frame deepfake analysis.
-   `POST /predict/audio`: Upload an audio file (`.wav`, `.mp3`) to detect synthesized speech.

---

**_This project was developed as a final year B.Tech project._**