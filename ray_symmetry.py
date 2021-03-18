
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
import cv2
import shutil

RAW_IMAGES_DIR = "/mnt/c/dev/media/aligned"

ray.init(num_cpus=multiprocessing.cpu_count())

def find_files(root):
    for d, dirs, files in os.walk(root):
        for f in files:
            yield os.path.join(d, f)

def calc_symmetry_metric(img_path):
    img = cv2.imread(img_path)
    width = img.shape[1]
    width_cutoff = width // 2
    s1 = img[:, :width_cutoff]
    s2 = cv2.flip(img[:, width_cutoff:],1)
    G_Y1 = cv2.reduce(s1, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F)
    G_Y2 = cv2.reduce(s2, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F)
    metric = cv2.compareHist(G_Y1, G_Y2, cv2.HISTCMP_BHATTACHARYYA)
    print("Symmetry metric for {0} is {1}".format(img_path, metric))
    return metric

@ray.remote
def process_image_shards(work):
    #SYMMETRY_FILTERED_IMAGES_DIR = "{0}/{1}".format('/mnt/c/dev/media/symmetric', os.getpid())
    SYMMETRY_FILTERED_IMAGES_DIR = '/mnt/c/dev/media/symmetric'
    SYMMETRY_FILTERED_REJECTS_DIR = '/mnt/c/dev/media/symmetric/rejects'

    Path(SYMMETRY_FILTERED_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    Path(SYMMETRY_FILTERED_REJECTS_DIR).mkdir(parents=True, exist_ok=True)

    for it in work:
    #    img_name = it
        for idx in range(len(it)):
            img_name = str(it[idx])
            img_path = os.path.join(RAW_IMAGES_DIR, img_name)
            # DEV img_name = str(it)
            try:
                print("Applying symmetry filter {0}\n".format(img_name))
                try:
                    metric = calc_symmetry_metric(img_path)
                    if (metric < 0.05):
                        shutil.copy(img_path, SYMMETRY_FILTERED_IMAGES_DIR)
                    else:
                        shutil.copy(img_path, SYMMETRY_FILTERED_REJECTS_DIR)
                except Exception:
                    try:
                        raise TypeError("Again !?!")
                    except:
                        pass
                    traceback.print_exc()
                    print("Exception in image shard processing")
            except:
                print("Exception in image shard processing")

image_files = list(find_files(RAW_IMAGES_DIR))

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