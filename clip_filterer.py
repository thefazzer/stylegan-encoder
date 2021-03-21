import torch
from PIL import Image
from pathlib import Path
import sys, os
import clip
import shutil
import random

src_dir = "/mnt/c/dev/media/photos"

clip_feature_dir = "/mnt/c/dev/media/photos/clip_feature.0"
clip_feature_dir_fls = "/mnt/c/dev/media/photos/clip_feature.1"

Path(clip_feature_dir).mkdir(parents=True, exist_ok=True)
Path(clip_feature_dir_fls).mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
filelist=list(Path(src_dir).rglob('*.jpg'))

while (True):
    file = random.choice(filelist)
    image = preprocess(Image.open(str(file))).unsqueeze(0).to(device)
    #text3 = clip.tokenize(["arms up", "arms down"]).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text3_features = model.encode_text(text3)
        logits_per_image, logits_per_text = model(image, text3)
        probs3 = logits_per_image.softmax(dim=-1).cpu().numpy()

        #print(file, "probs3:", probs3, ("*** ARMS UP ***" if probs3[0][0] > 0.7 else ""))

        if probs3[0][0] > 0.9:
            shutil.copy(file, clip_feature_dir)
        else:
            shutil.copy(file, clip_feature_dir_fls)
            #print(file, "ARMS UP!")