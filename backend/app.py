import io
import os
import tempfile
from typing import Optional
import uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from collections import OrderedDict
import subprocess 

import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import timm
import librosa
from huggingface_hub import hf_hub_download

from utils.face_crop import init_face_detector, crop_face_from_image_pil, crop_face_from_frame_bgr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_REPO_ID = "AryakumarMishra/deepfake-detector-models"
PHOTO_ENSEMBLE_FILENAME = "best_deepfake_detector_ensemble.pth"
AUDIO_MODEL_FILENAME = "vit_audio_best.pth"

print("Downloading models from Hugging Face Hub...")
try:
    PHOTO_ENSEMBLE_PATH = hf_hub_download(repo_id=MODEL_REPO_ID, filename=PHOTO_ENSEMBLE_FILENAME)
    AUDIO_MODEL_PATH = hf_hub_download(repo_id=MODEL_REPO_ID, filename=AUDIO_MODEL_FILENAME)
    print("Models downloaded successfully.")
except Exception as e:
    print(f"Error downloading models: {e}")

PHOTO_THRESHOLD = 0.25
AUDIO_THRESHOLD = 0.5


class WeightedEnsembleModel(nn.Module):
    """
    Weighted ensemble with learnable or fixed weights.
    Supports both averaging and learnable meta-classifier.
    """
    def __init__(self, models, weights=None, learnable=False):
        super(WeightedEnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.learnable = learnable
        
        if learnable:
            self.weights = nn.Parameter(torch.tensor(weights if weights else [1.0] * len(models)))
        else:
            self.register_buffer('weights', torch.tensor(weights if weights else [1.0] / len(models))) # type: ignore (pylance error; code works fine)
    
    def forward(self, x):
        """
        Forward pass through all models and combine predictions.
        """
        outputs = []
        
        with torch.set_grad_enabled(self.learnable):
            for model in self.models:
                out = model(x).squeeze(-1)
                outputs.append(out)
        
        stacked_outputs = torch.stack(outputs, dim=0)
        
        if self.learnable:
             normalized_weights = torch.softmax(self.weights, dim=0)
        else:
             normalized_weights = self.weights
        
        weighted_output = (stacked_outputs * normalized_weights.view(-1, 1)).sum(dim=0)
        
        return weighted_output.unsqueeze(-1)
    

def load_model_with_cleaning(model_path, model_creator, device):
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            name = k[10:]
        elif k.startswith('module.'):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    
    model = model_creator()
    model.load_state_dict(new_state_dict)
    return model



photo_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
audio_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

app = FastAPI(title="Multi-Modal Deepfake Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])



print("Loading photo/video ensemble model...")

efficientnet_model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1).to(DEVICE)
xception_model = timm.create_model('xception', pretrained=False, num_classes=1).to(DEVICE)
convnext_model = timm.create_model('convnext_base', pretrained=False, num_classes=1).to(DEVICE)
swin_model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=1).to(DEVICE)

WEIGHTS = [0.35, 0.30, 0.20, 0.15]
photo_model = WeightedEnsembleModel(
    models=[efficientnet_model, xception_model, convnext_model, swin_model],
    weights=WEIGHTS,
    learnable=False
).to(DEVICE)


photo_model.load_state_dict(torch.load(PHOTO_ENSEMBLE_PATH, map_location=DEVICE))
photo_model.eval()
print("Photo/Video model loaded successfully.")


print("Loading audio model...")
audio_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1)
audio_state_dict = torch.load(AUDIO_MODEL_PATH, map_location=DEVICE)


new_audio_state_dict = OrderedDict()
for k, v in audio_state_dict.items():
    name = k.replace('_orig_mod.', '').replace('module.', '')
    new_audio_state_dict[name] = v

audio_model.load_state_dict(new_audio_state_dict)
audio_model.to(DEVICE)
audio_model.eval()
print("Audio model loaded successfully.")


init_face_detector(device=DEVICE)


# HELPER FUNCTIONS
def decide_label(prob_fake: float, threshold: float) -> str:
    return "Fake" if prob_fake >= threshold else "Real"

def predict_face_pil(pil_img: Image.Image) -> float:
    img_tensor = photo_transform(pil_img).unsqueeze(0).to(DEVICE) # type: ignore (pylance error; code works fine)
    with torch.no_grad():
        logit = photo_model(img_tensor)
        prob_fake = torch.sigmoid(logit).cpu().item()
    return max(0.0, min(1.0, prob_fake))


def predict_spectrogram_pil(pil_img: Image.Image) -> float:
    img_tensor = audio_transform(pil_img).unsqueeze(0).to(DEVICE) # type: ignore (pylance error; code works fine)
    with torch.no_grad():
        logit = audio_model(img_tensor)
        prob_fake = torch.sigmoid(logit).cpu().item()
    return max(0.0, min(1.0, prob_fake))


# API ENDPOINTS
@app.post("/predict/photo")
async def predict_photo(file: UploadFile = File(...)):
    
    print("Running photo model...")
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return JSONResponse({"error": "Failed to read image.", "detail": str(e)}, status_code=400)

    face = crop_face_from_image_pil(pil_img)
    if face is None: return JSONResponse({"error": "No face detected."}, status_code=400)

    prob_fake = predict_face_pil(face)
    label = decide_label(prob_fake, PHOTO_THRESHOLD)
    print(f"Photo Prediction ==> Label: {label}, Prob(Fake): {prob_fake:.4f}")
    return {"label": label, "probability_fake": prob_fake}


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...), frame_rate: float = Query(1.0, ge=0.1, le=10.0)):

    print("Running video/audio models...")
    tmp_name = None
    audio_tmp_name = None
    visual_prob = 0.0
    audio_prob = None 
    frames_with_face = 0

    try:
        # Create temp file for video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            tmp_name = tmp.name
        
        # Create temp file for audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            audio_tmp_name = tmp_audio.name

        # Extract audio using ffmpeg
        print("Extracting audio stream with ffmpeg...")
        try:
            ffmpeg_command = [
                "ffmpeg",
                "-i", tmp_name,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                "-ac", "1",
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                audio_tmp_name
            ]
            subprocess.run(ffmpeg_command, check=True)
            print("Audio extraction successful.")
        except Exception as ffmpeg_e:
            print(f"ffmpeg audio extraction failed: {ffmpeg_e}. Proceeding with video only.")
            audio_prob = None
        
        # Visual Processing
        print("Processing video frames...")
        cap = cv2.VideoCapture(tmp_name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        sample_every = max(1, int(round(fps / frame_rate)))
        
        frame_idx, probs = 0, []
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % sample_every == 0:
                pil_face = crop_face_from_frame_bgr(frame)
                if pil_face:
                    probs.append(predict_face_pil(pil_face))
                    frames_with_face += 1
            frame_idx += 1
        cap.release()

        if frames_with_face == 0: 
            return JSONResponse({"error": "No faces detected in video."}, status_code=400)
        
        visual_prob = float(np.mean(probs))
        print(f"Visual processing complete. Prob(Fake): {visual_prob:.4f}")

        # Audio Processing
        if audio_prob is not False: 
            try:
                print("Processing extracted audio...")
                y, sr = librosa.load(audio_tmp_name, sr=None) 
                
                if y.size > 0 and np.any(y):
                    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
                    S_dB = librosa.power_to_db(S, ref=np.max)
                    S_dB_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min()) * 255.0
                    pil_img_audio = Image.fromarray(S_dB_norm.astype(np.uint8)).convert("RGB")
                    
                    audio_prob = predict_spectrogram_pil(pil_img_audio)
                    print(f"Audio processing complete. Prob(Fake): {audio_prob:.4f}")
                else:
                    print("Audio track is empty or silent, skipping audio analysis.")
                    audio_prob = None 

            except Exception as audio_e:
                print(f"Could not process audio: {audio_e}. Returning video-only result.")
                audio_prob = None
    
    except Exception as e:
        return JSONResponse({"error": "An error occurred during video processing.", "detail": str(e)}, status_code=500)

    finally:
        if tmp_name: os.unlink(tmp_name)
        if audio_tmp_name: os.unlink(audio_tmp_name) 

    # Combine visual and audio probabilities
    final_prob = 0.0
    if audio_prob is not None:
        final_prob = (visual_prob * 0.7) + (audio_prob * 0.3)
        print(f"Combined Prob: {final_prob:.4f} (70% Visual, 30% Audio)")
    else:
        final_prob = visual_prob
        print("Audio not detected or failed. Returning video-only probability.")

    final_label = decide_label(final_prob, PHOTO_THRESHOLD)
    
    print(f"Video Prediction ==> Label: {final_label}, Final Prob(Fake): {final_prob:.4f}")
    return {
        "label": final_label,
        "probability_fake_avg": final_prob,
        "visual_prob": visual_prob,
        "audio_prob": audio_prob if audio_prob is not None else "N/A",
        "frames_with_face": frames_with_face
    }


@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    tmp_name = None
    try:
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_name = tmp.name
        
        # Create spectrogram in memory
        y, sr = librosa.load(tmp_name, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        S_dB_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min()) * 255.0
        pil_img = Image.fromarray(S_dB_norm.astype(np.uint8)).convert("RGB")
        
        # Get prediction
        prob_fake = predict_spectrogram_pil(pil_img)
        label = decide_label(prob_fake, AUDIO_THRESHOLD)
        print(f"Audio Prediction ==> Label: {label}, Prob(Fake): {prob_fake:.4f}")

        return {"label": label, "probability_fake": prob_fake}

    except Exception as e:
        return JSONResponse({"error": "An error occurred during audio processing.", "detail": str(e)}, status_code=500)
    finally:
        if tmp_name: os.unlink(tmp_name)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860)