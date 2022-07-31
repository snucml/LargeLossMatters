# Getting the Data

## PASCAL

1. Navigate to the PASCAL data directory:
```
cd /path/to/LargeLossMatters/data/pascal
```
2. Download the data:
```
curl http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar --output pascal_raw.tar
```
3. Extract the data:
```
tar -xf pascal_raw.tar
```
4. Clean up:
```
rm pascal_raw.tar
```

## COCO

1. Navigate to the COCO data directory:
```
cd /path/to/LargeLossMatters/data/coco
```
2. Download the data:
```
curl http://images.cocodataset.org/annotations/annotations_trainval2014.zip --output coco_annotations.zip
curl http://images.cocodataset.org/zips/train2014.zip --output coco_train_raw.zip
curl http://images.cocodataset.org/zips/val2014.zip --output coco_val_raw.zip
```
3. Extract the data:
```
unzip -q coco_annotations.zip
unzip -q coco_train_raw.zip
unzip -q coco_val_raw.zip
```
4. Clean up:
```
rm coco_train_raw.zip
rm coco_val_raw.zip
```

## NUSWIDE

*These instructions differ slightly from those for the other datasets because we re-crawled NUSWIDE.*

1. Follow the instructions [here](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) to request a download link for the NUSWIDE images. Once approved, you will receive a link to download `Flickr.zip` which contains the images for the NUSWIDE dataset. Download this file and move it to the NUSWIDE data directory, so that the full path is:
```
/path/to/LargeLossMatters/data/nuswide/Flickr.zip
```
2. Navigate to the NUSWIDE data directory:
```
cd /path/to/LargeLossMatters/data/nuswide
```
3. Extract the images:
```
unzip -q Flickr.zip
```
4. Clean up:
```
rm Flickr.zip
```
5. Download the files:
```
formatted_train_labels.npy
formatted_val_labels.npy
formatted_train_images.npy
formatted_val_images.npy
```
from [here](https://caltech.box.com/v/single-positive-multi-label) and copy them to `/path/to/LargeLossMatters/data/nuswide`. Note that these files correspond to the subset of the NUSWIDE dataset that we re-crawled ourselves. 


## CUB

1. Navigate to the CUB data directory:
```
cd /path/to/LargeLossMatters/data/cub
```
2. Download `CUB_200_2011.tgz` from [here](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz) and move the resulting file to the CUB data directory, resulting in the full path:
```
/path/to/LargeLossMatters/data/cub/CUB_200_2011.tgz
```
3. Extract the data:
```
tar -xf CUB_200_2011.tgz
```
4. Clean up:
```
rm CUB_200_2011.tgz
```

## OpenImages V3

1. Navigate to the OpenImages V3 data directory:
```
cd /path/to/LargeLossMatters/data/openimages
```
2. Download metadatas:
```
wget https://storage.googleapis.com/openimages/2017_11/images_2017_11.tar.gz
wget https://storage.googleapis.com/openimages/2017_11/annotations_human_2017_11.tar.gz
wget https://storage.googleapis.com/openimages/2017_11/classes_2017_11.tar.gz
```
3. Extract metadatas:
```
tar -zxvf images_2017_11.tar.gz
tar -zxvf annotations_human_2017_11.tar.gz
tar -zxvf classes_2017_11.tar.gz
```
4. Clean up:
```
rm images_2017_11.tar.gz
rm annotations_human_2017_11.tar.gz
rm classes_2017_11.tar.gz
```
5. Download training images (using tool slightly modified from [here](https://github.com/openimages/dataset/blob/main/downloader.py)):
```
mkdir images_train
python generate_imageid_list.py
python downloader.py imageid_list_train_v3.txt --download_folder images_train
```

6. Download val/test images (refer [here](https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations)):
```
mkdir images_val
mkdir images_test
aws s3 --no-sign-request sync s3://open-images-dataset/validation images_val
aws s3 --no-sign-request sync s3://open-images-dataset/test images_test
```

# Formatting the Data
The `preproc` folder contains a few scripts which can be used to produce uniformly formatted image lists and labels:
```
cd /path/to/LargeLossMatters/preproc
python format_coco.py
python format_cub.py
python format_pascal.py
```
Note that NUSWIDE does not have a formatting script. This is because we provide the formatted metadata directly.

# Generating Observed Labels
The script `preproc/generate_observed_labels.py` subsamples the entries of a complete label matrix to generate "observed labels" which simulate single positive labeling. To generate observed labels for a given dataset, run:
```
cd /path/to/LargeLossMatters/preproc
python generate_observed_labels.py --dataset X
```
where `X` is replaced by `pascal`, `coco`, `nuswide`, or `cub`. You will only need to do this once.
