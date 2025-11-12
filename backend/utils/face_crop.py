# utils/face_crop.py
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import cv2

# instantiate once (device set when importing)
mtcnn = None

def init_face_detector(device='cpu'):
    global mtcnn
    if mtcnn is None:
        mtcnn = MTCNN(keep_all=True, device=device)

def crop_face_from_image_pil(pil_img):
    """
    Returns the largest detected face box region (PIL Image).
    If no face detected, returns centered crop around image center (fallback).
    """
    global mtcnn
    if mtcnn is None:
        init_face_detector()
    
    result = mtcnn.detect(pil_img) #type: ignore
    if result is None:
        return pil_img

    boxes, _ = result[0], result[1]

    if boxes is None:
        # fallback - center crop
        w,h = pil_img.size
        side = min(w,h)
        left = (w-side)//2
        top = (h-side)//2
        return pil_img.crop((left, top, left+side, top+side))
    # pick largest box by area
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    idx = int(np.argmax(areas))
    box = boxes[idx].astype(int)
    face = pil_img.crop((box[0], box[1], box[2], box[3]))
    return face

def crop_face_from_frame_bgr(frame_bgr):
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return crop_face_from_image_pil(pil)
