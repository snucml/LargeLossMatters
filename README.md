# Large Loss Matters in Weakly Supervised Multi-Label Classification (CVPR 2022) | Paper

Youngwook Kim*, Jae Myung Kim*, Zeynep Akata, and Jungwoo Lee

Primary contact : [ywkim@cml.snu.ac.kr](ywkim@cml.snu.ac.kr)

## Abstract
Weakly supervised multi-label classification (WSML) task, which is to learn a multi-label classification using partially observed labels per image, is becoming increasingly important due to its huge annotation cost. In this work, we first regard unobserved labels as negative labels, casting the WSML task into noisy multi-label classification. From this point of view, we empirically observe that memorization effect, which was first discovered in a noisy multi-class setting, also occurs in a multi-label setting. That is, the model first learns the representation of clean labels, and then starts memorizing noisy labels. Based on this finding, we propose novel methods for WSML which reject or correct the large loss samples to prevent model from memorizing the noisy label. Without heavy and complex components, our proposed methods outperform previous state-of-the-art WSML methods on several partial label settings including Pascal VOC 2012, MS COCO, NUSWIDE, CUB, and OpenImages V3 datasets. Various analysis also show that our methodology actually works well, validating that treating large loss properly matters in a weakly supervised multi-label classification.

## Dataset Preparation
See the `README.md` file in the `data` directory for instructions on downloading and setting up the datasets.

## Model Training & Evaluation
You can train and evaluate the models by
```
python main.py --exp_name [expname] \
               --dataset [dataset] \
               --mod_scheme [scheme] \
               --delta_rel [delta_rel]
```
where ```[data_path]``` $\in \{$'pascal', 'coco', 'nuswide', 'cub'$\}$, ```[scheme]``` $\in \{$'LL-R', 'LL-Ct', 'LL-Cp'$\}$, and 
```[delta_rel]``` $\in \{0.1, 0.2, 0.3, 0.4, 0.5\}$.

For other configuration (for example, learning rate, batch size), see ```config.py```.

Currently we only support ''End-to-end'' training setting.

## Acknowledgements
Our code is heavily built upon [Multi-Label Learning from Single Positive Labels](https://github.com/elijahcole/single-positive-multi-label).