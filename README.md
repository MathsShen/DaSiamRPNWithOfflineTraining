# DaSiamRPNWithOfflineTraining

This repository adds offline training module and testing module (including distractor-awareness and local2global strategy) to the original PyTorch implementation of [DaSiamRPN](https://github.com/foolwood/DaSiamRPN).

## Introduction

**SiamRPN** formulates the task of visual tracking as a task of localization and identification simultaneously, initially described in an [CVPR2018 spotlight paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf). (Slides at [CVPR 2018 Spotlight](https://drive.google.com/open?id=1OGIOUqANvYfZjRoQfpiDqhPQtOvPCpdq))

**DaSiamRPN** improves the performances of SiamRPN by (1) introducing an effective sampling strategy to control the imbalanced sample distribution, (2) designing a novel distractor-aware module to perform incremental learning, (3) making a long-term tracking extension. [ECCV2018](https://arxiv.org/pdf/1808.06048.pdf). (Slides at [VOT-18 Real-time challenge winners talk](https://drive.google.com/open?id=1dsEI2uYHDfELK0CW2xgv7R4QdCs6lwfr))

Specifically, for (2), this repository implements ROI-align technique to achieve similarity matching between x and z. The insight of the ROI-align implementation can be seen from the figure below.
<img src="./ext/roi-align.png" width="500" hegiht="1000" align=center />

## Prerequisites

CPU: Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz
GPU: NVIDIA GTX1060

- python3.6
- pytorch == 0.4.0
- numpy
- opencv
- easydict

## Data Preparation
One can prepare his own dataset for training and testing DaSiamRPN. Putting aside the positive and negative pairing for distractor-aware training as specified in the paper, each training and testing sequence, say Basketball, is organized in a folder "Basketball" in which "Basketball_gt.txt" and a sub-folder "imgs" are stored. The gt files have lines of groundtruths in format of (x, y, w, h). In "imgs" folders are frames named in format "xxxx.jpg", (e.g. 0001.jpg-9999.jpg).

Besides the data, one should also prepare the corresponding list formatted as in ./data/whole_list.txt, where each row consists of the path of a sequence folder and the number of total frames of the sequence.

## Training Procedure
`python code/train.py`

The model will be saved in ./output/weights/

## Testing Procedure
`python code/test.py`

## Postscript
Currently, this repo remains under construction, meaning that its effectiveness is not guaranteed. But one can still get some insights from reading this repo, including myself. And that is exactly what really matters. However, more is coming in the immediate future, including: (1) the sampling strategy to control the imbalanced sample distribution and (2) other implementation details not specified clearly in the related paper.

To better this repo, I am looking forward to your suggestion. ^_^

