import os
import ray
import zipfile
import numpy as np
import multiprocessing
from ray.exceptions import RayTaskError

RAW_IMAGES_DIR = "/mnt/c/dev/media"
image_file_zips = os.listdir(RAW_IMAGES_DIR)
ray.init(num_cpus=multiprocessing.cpu_count())

@ray.remote
def process_zip_shards(work):

    UNZIPPED_IMAGES_DIR = "{0}/{1}".format(RAW_IMAGES_DIR, os.getpid())
    
    for it in work:
        for idx in range(len(it)):
            zip_ref = str(it[idx])
            print("{0} upzipping {1}\n".format(os.getpid(), zip_ref))
            file_path =  "{0}/{1}".format(RAW_IMAGES_DIR, zip_ref)# get full path of files
            zip_ref = zipfile.ZipFile(file_path) # create zipfile object
            zip_ref.extractall(UNZIPPED_IMAGES_DIR) # extract file to dir
            zip_ref.close() # close file

it = (
    ray.util.iter.from_items(image_file_zips, num_shards=multiprocessing.cpu_count()).batch(50)
)

# work 
work = [process_zip_shards.remote(zip_shard) for zip_shard in it.shards()]

try:
    ray.get(work)

except (StopIteration): 
    print("shutting down")
    ray.shutdown()