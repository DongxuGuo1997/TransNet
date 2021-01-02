# TransNet: Benchmark for Pedestrian Stop & Go Forecasting
This repository contains the code of *TransNet*, our benchmark for studying the pedestrian stop and go behaviors in the context of self-driving. 
Walking to standing and standing to walking transitions are specific and relatively rare cases, hence samples are collected from several external datasets and 
integrated with unified interface.

![exp](imgs/jaad_01.gif)

### Table of Contents
- [Installation](#installation)
- [Data preparation](#Data preparation)
- [Interface](#interface)
- [Statistics](#Statistics)
- [References](#references)
- [Citation](#citation)


## Installation

```
# To clone the repository using HTTPS
git clone https://github.com/DongxuGuo1997/PSGF.git
```

The project is written and tested using python 3.8. The interface also require external libraries like PyTorch,
opencv-python, etc.  All required packages can be found in `requirements.txt`. 
Please ensure all expected modules are installed before using the interface.


## Data preparation
#### Support datasets
At current stage we focus on the datasets related to the study of pedestrian actions/intentions from on-board vehicles.
The suitable datasets should provide RGB images captured from an egocentric view of a moving vehicle and accurately annotated.<br>

Currently TransNet support the following datasets:<br/>
* Joint Attention in Autonomous Driving([JAAD](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/)) Dataset
* Pedestrian Intention Estimation([PIE](http://data.nvision2.eecs.yorku.ca/PIE_dataset/)) Dataset
* Trajectory Inference using Targeted Action priors Network([TITAN](https://usa.honda-ri.com/titan)) Dataset

#### Download
##### JAAD

- Download the videos and annotations from [official page](https://github.com/ykotseruba/JAAD). 
- Use the [scripts](https://github.com/ykotseruba/JAAD/blob/JAAD_2.0/split_clips_to_frames.sh) provided to extract images from videos.
- Use JAAD's [interface](https://github.com/ykotseruba/JAAD/blob/JAAD_2.0/jaad_data.py#L421) to generate complete annotations in the form of python dictionary.
- Train / val / test split: [link](https://github.com/ykotseruba/JAAD/tree/JAAD_2.0/split_ids)

##### PIE

- Download the videos and annotations from [offical page](https://github.com/aras62/PIE#interface). 
- Use the [scripts](https://github.com/aras62/PIE/blob/master/split_clips_to_frames.sh) provided to extract images from videos.
- Use PIE's [interface](https://github.com/aras62/PIE/blob/master/pie_data.py#L441) to generate complete annotations in the form of python dictionary.
- Train / val / test split: [link](https://github.com/aras62/PIE/blob/2256f96b8ab24d8407af34fb1f0b9a4714cd532e/pie_data.py#L84)

##### TITAN
To obtain the dataset, please refer to this [page]( https://usa.honda-ri.com/titan) and contact them directly.

After download the data, please place the images, annotations and video split ids in [DATA](https://github.com/DongxuGuo1997/TransNet/tree/main/DATA).
The expected result:
```
├── DATA/
│   ├── annotations/ 
│   │   ├── JAAD/ 
│   │   │   ├── anns/  
│   │   │   │   ├── JAAD_DATA.pkl
│   │   │   └── splits/
│   │   │   │   ├── all_videos/
│   │   │   │   │   ├── train/
│   │   │   │   │   └── val/
│   │   │   │   │   └── test/
│   │   │   │   └── default/
│   │   │   │   │   ├── train/
│   │   │   │   │   └── val/
│   │   │   │   │   └── test/
│   │   │   │   └── high_visibility/
│   │   │   │   │   ├── train/
│   │   │   │   │   └── val/
│   │   │   │   │   └── test/  
│   │   └── PIE/
│   │   │   ├── anns/ 
│   │   │   │   ├── PIE_DATA.pkl
│   │   └── TITAN/
│   │   │   ├── anns/
│   │   │   │   ├── clip_x.csv
│   │   │   └── splits/
│   │   │   │   │   ├── train_set/
│   │   │   │   │   └── val_set/
│   │   │   │   │   └── test_set/
│   └── images/
│   │   ├── JAAD/
│   │   │   ├── video_xxxx/ # 346 videos
│   │   └── PIE/
│   │   │   ├── set_0x/ # 6 sets
│   │   │   │   ├── video_xxxx/ 
│   │   └── TITAN/
│   │   │   ├── clip_xxx/  # 786 clips
└── (+ files and folders containing the raw data)
```
<b> Note </b>: No need to gather all datasets. The benchmark works normally with arbitrary combination of supported datasets.

## Interface  
At the heart of TransNet data blocks is the [`TransDataset`](https://github.com/DongxuGuo1997/TransNet/blob/main/src/dataset/trans/data.py) class.
`TransDataset` integrates the functions to collect transition samples from original annotations of JAAD, PIE and TITAN.
Using attributes of `TransDataset`, the user can conveniently extract the frame and history instances related to the stop & go of pedestrians:<br>
* `extract_trans_frame()`: extract the frame where stop or go transitions occur and the annotations of involved pedestrian
* `extract_trans_history()`: extract the whole history of a pedestrian up to the frame when transition happens <br>
The extracted samples each has an unique id specifying the source dataset(`J`,`P`,`T`), transition type(`S`,`G`),data split(`trian`,`val`,`test`) 
and sample index,ie. `TG_003_train`. The data loading is done by customized PyTorch dataloader. For detailed usage please check the example in .

## Statistics
