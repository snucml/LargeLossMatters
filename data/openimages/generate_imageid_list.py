import pandas as pd
import argparse
import os
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
      '--download_folder',
      type=str,
      default=None,
      help='Folder where to download the images.')
args = parser.parse_args()

annotations = pd.read_csv('2017_11/train/annotations-human.csv')

image_ids = np.array(list(set(annotations.ImageID)))
exist_image_files = os.listdir(args.download_folder)
exist_image_ids = np.array([file.rstrip('.jpg') for file in exist_image_files])
rest_image_ids = np.setdiff1d(image_ids, exist_image_ids)

newline = '\n'
f = open('imageid_list_train_v3.txt', 'w')
for image_id in tqdm(rest_image_ids):
    f.write(f'train/{image_id}{newline}')
f.close()
