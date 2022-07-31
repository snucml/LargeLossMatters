import pandas as pd
import argparse
import os
from tqdm import tqdm
import numpy as np

annotations = pd.read_csv('2017_11/train/annotations-human.csv')

image_ids = np.array(list(set(annotations.ImageID)))

newline = '\n'
f = open('imageid_list_train_v3.txt', 'w')
for image_id in tqdm(image_ids):
    f.write(f'train/{image_id}{newline}')
f.close()
