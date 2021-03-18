
import numpy as np
import os
import zipfile
import sys
import bz2
import argparse
from keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
import multiprocessing
import ray, rayrunner
from rayrunner import ProgressBar
import time
import multiprocessing
from tqdm import tqdm
from threading import Event
from typing import Tuple
from ray.exceptions import RayTaskError
from ray.actor import ActorHandle
import traceback
from pathlib import Path

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2', 
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))
    
landmarks_detector = LandmarksDetector(landmarks_model_path)
RAW_IMAGES_DIR = "/mnt/c/dev/media/photos"
image_files = os.listdir(RAW_IMAGES_DIR)
ray.init(num_cpus=multiprocessing.cpu_count())

@ray.remote
def process_image_shards(work):
    ALIGNED_IMAGES_DIR = "{0}/{1}".format('/mnt/c/dev/media/aligned', os.getpid())
    #ALIGNED_IMAGES_DIR = '/mnt/c/dev/media/aligned/'
   
    Path(ALIGNED_IMAGES_DIR).mkdir(parents=True, exist_ok=True)

    output_size = 1024
    x_scale = 1
    y_scale = 1
    em_scale = 0.1
    use_alpha = False   

    for it in work:
        for idx in range(len(it)):
            img_name = str(it[idx])
            # DEV img_name = str(it)
            print("Aligning {0}\n".format(img_name))
            try:
                raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
                fn = face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], 1)
                if os.path.isfile(fn):
                    continue
                print("Getting landmarks for {0}...".format(raw_img_path))
                for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                    try:
                        face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
                        print("Starting face alignment for {0}...".format(face_img_name))
                        aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
                        image_align(raw_img_path, aligned_face_path, face_landmarks, output_size=output_size, x_scale=x_scale, y_scale=y_scale, em_scale=em_scale, alpha=use_alpha)
                        print('Wrote result %s' % aligned_face_path)
                    except Exception:
                        try:
                            raise TypeError("Again !?!")
                        except:
                            pass
                        traceback.print_exc()
                        print("Exception in face alignment!")
            except:
                print("Exception in landmark detection!")

it = (
    ray.util.iter.from_items(image_files, num_shards=multiprocessing.cpu_count()).batch(50)
)

# work 
work = [process_image_shards.remote(img_shard) for img_shard in it.shards()]

try:
    # DEV SYNCHRONOUS
    # process_image_shards(image_files)

    ray.get(work)

except (StopIteration): 
    print("shutting down")
    ray.shutdown()