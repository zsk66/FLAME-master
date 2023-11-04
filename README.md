# FLAME-master
This repo holds the source code and scripts for reproducing the key experiments of our paper:

**"Personalized Federated Learning via ADMM with Moreau Envelope".**

Author: Shengkun Zhu, Jinshan Zeng, Sheng Wang, Yuan Sun, Zhiyong Peng.
## Datasets and Models
| Datesets | # of samples | ref. | Models |
| :----: | :----: | :----: | :----: |
Synthetic | 5k-10k/device | [Li et al.](https://proceedings.mlsys.org/paper_files/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf) | SVM
Mnist | 70,000 | [LeCun et al.](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=4cccb7c5b2d59bc0b86914340c81b26dd4835140) | MLR
Fmnist | 70,000 | [Xiao et al.](https://arxiv.org/pdf/1708.07747.pdf) | MLP |
Mmnist | 58,954 | [Kaggle](https://www.kaggle.com/datasets/andrewmvd/medical-mnist) | CNN

## Start

The default values for various parameters parsed to the experiment are given in options.py. Details are given on some of those parameters:
`---framework:` we provide two personalied federated learnign frameworks for the readers: FLAME and pFedMe.
