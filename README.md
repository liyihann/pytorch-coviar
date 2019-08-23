# CoViAR

## Overview

This is an enhanced version of CoViAR in [PyTorch](http://pytorch.org/) based on `chaoyuaw`'s [work](https://github.com/chaoyuaw/pytorch-coviar).

Compared to the original version, our implementation supports taking stacked motion vectors as input for training, which improves prediction accuracy although leads to a bigger time cost(this is a work in progress and we are still working with this bottleneck.)

## Getting Started

### Installation

We strongly recommend you to use [Anaconda](https://www.anaconda.com/distribution) to configure the environment.

In this section, we take installation on Linux-x86_64 as an example. 

#### 1. Install `Anaconda3`

Choose a suitable anaconda version in this page : https://www.anaconda.com/distribution/#download-section

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
```

Install anaconda3.

```bash
sh Anaconda3-2019.03-Linux-x86_64.sh
```

Then, specify this `anaconda` in your bash shell.

```bash
eval "$(${ANACONDA_PATH}/bin/conda shell.bash hook)"
```

#### 2. Create Virtual Environment

Create a virtual environment for CoViAR using anaconda.

We specify version of dependencies for CoViAR here.

```bash
conda create -n coviar python=3 cudatoolkit=8.0 pytorch=0.3.0
```

Then activate this virtual environment so that we can install more packages for the project.

```bash
conda activate coviar
```
Install `numpy` and `opencv` in this virtual environment.
````bach
conda install numpy
conda install -c conda-forge opencv=3.1
````

#### 3. Clone Project

Navigate to a directory that you want to install CoViAR and clone this project.

```bash
cd ${COVIAR_PARENT_DIRECTORY_PATH}
git clone https://github.com/liyihann/pytorch-coviar.git
```

For convenience, we use `${COVIAR_PATH}` to represent `${COVIAR_PARENT_DIRECTORY_PATH/pytorch-coviar}`.

#### 4. Configure `ffmpeg` 

Download and unzip `ffmpeg-3.1.3` to `dataloader` folder.

```bash
cd ${COVIAR_PATH}/dataloader/
wget https://www.ffmpeg.org/releases/ffmpeg-3.1.3.tar.bz2
tar xjf ffmpeg-3.1.3.tar.bz2
mv ffmpeg-3.1.3 ffmpeg
```

Configure and compile `ffmpeg`.

```bash
cd ffmpeg
./configure --prefix=${COVIAR_PATH}/data_loader/ffmpeg --enable-pic --enable-shared
make
make install
```

Configure environment in `anaconda`.

```bash
ln -s ${COVIAR_PATH}/data_loader/ffmpeg/bin/ffmpeg ${ANACONDA_PATH}/envs/coviar/bin/
cp -R ${COVIAR_PATH}/data_loader/ffmpeg/lib/* ${ANACONDA_PATH}/envs/origin/lib/
```

Set environment variable `$LD_LIBRARY_PATH`. This environment variable will be effective once the environment is activated.

```bash
cd ${ANACONDA_PATH}/envs/coviar/
mkdir etc && cd etc 
mkdir conda && cd conda
mkdir activate.d && cd activate.d
vi env_vars.sh
```

Use vim to edit `env_vars.sh`.

```
#!/bin/sh
export LD_LIBRARY_PATH=${COVIAR_PATH}/data_loader/ffmpeg/lib/
```

Then unset environment variable `$LD_LIBRARY_PATH`. This environment variable will be ineffective once the environment is deactivated.

```bash
cd ${ANACONDA_PATH}/envs/coviar/etc/conda
mkdir deactivate.d && cd deactivate.d
vi env_vars.sh
```

Use vim to edit `env_vars.sh`.

```
#!/bin/sh
unset LD_LIBRARY_PATH
```

Re-activate virtual environment.

```bash
conda deactivate
conda activate coviar
```

#### 5. Build CoViAR DataLoader

```bash
cd ${COVIAR_PATH}/data_loader
rm -rf build
python setup.py build_ext
python setup.py install --user
```

#### 6. Download Datasets

Download datasets to  `data` folder.

```bash
cd ${COVIAR_PATH}/data
```

Download [`UCF-101`](https://www.crcv.ucf.edu/data/UCF101.php) dataset.

```bash
mkdir ucf101
nohup sh get_ucf101_data.sh &
```

Download [`HMDB51`](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) dataset.

```bash
mkdir hmdb51
nohup sh get_hmdb51_data.sh &
```

Here, `nohup` and `&` allow you to download datasets in background and save output to `nohup.out`.

#### 7. Re-encode Videos

Use `ffmpeg` to re-encode videos in the datasets to suitable format for training and testing later.

Re-encode `UCF-101` videos.

```bash
cd ${COVIAR_PATH}/data/ucf101
mkdir mpeg4_videos
cd ..
nohup sh reencode.sh ucf101/UCF-101/ ucf101/mpeg4_videos/ &
```

Re-encode `HMDB51` videos.

```bash
cd ${COVIAR_PATH}/data/hmdb51
mkdir mpeg4_videos
cd ..
nohup sh reencode.sh hmdb51/videos/ hmdb51/mpeg4_videos/ &
```

Now, we are ready for training and testing.

### Usage

This section describes how to use CoViAR data loader independently.

The data loader has two functions: 

- `get_num_frames` for counting the number of frames in a video.
- `load` for loading a representation.

To use either function, you will want to follow the following steps:

1. Specify `anaconda` in your bash shell.

```bash
eval "$(${ANACONDA_PATH}/bin/conda shell.bash hook)"
```

2. Activate CoViAR virtual environment.

```bash
conda activate coviar
```

3. Access python shell.

```bash
python
```

Function `get_num_frames` takes path to a specified video as parameters and returns the number of frames of the video.

```python
from coviar import get_num_frames
num_frames = get_num_frames([$VIDEO_PATH])
print(num_frames)
```

Function `load ` returns a specified frame image. It takes path to a specified video, GOP index and position index of a specified frame, representation type and a boolean variable `isAccumulate` as parameters.

```python
from coviar import load
image = load([$VIDEO_PATH], [gop_index], [frame_index], [representation_type], [isAccumulate])
```

Here, `gop_index` is specified by `frame_index=0,1,...` of one GOP which is specified by `frame_index=0,1,...`. `representation_type` can be `0`, `1`, or `2` where `0` is for I-frames, `1` is for motion vectors and `2` is for residuals. `isAccumulate` can be either `True` or `False` where `True` returns the accumulated representation and `False` returns the original compressed representations. 

For example,

```python
load(input.mp4, 3, 8, 1, True)
```

returns the accumulated motion vectors of the 9th frame of the 4th GOP.

### Training

To train the model, you will want to follow the following steps:

1. Specify `anaconda` in your bash shell.

```bash
eval "$(${ANACONDA_PATH}/bin/conda shell.bash hook)"
```

2. Activate CoViAR virtual environment.

```bash
conda activate coviar
```

3. Navigate to CoViAR directory.

```bash
cd ${COVIAR_PATH}
```

4.  Use the following commands to train` I-frame model` and `Motion vector model` on dataset `UCF-101`.

```bash
# I-frame model.
nohup python train.py --lr 0.0003 --batch-size 5 --arch resnet152 \
 	--data-name ucf101 --representation iframe \
 	--data-root data/ucf101/mpeg4_videos \
 	--train-list data/datalists/ucf101_split1_train.txt \
 	--test-list data/datalists/ucf101_split1_test.txt \
 	--model-prefix ucf101_iframe_model \
 	--lr-steps 150 270 390  --epochs 510 \
 	--gpus 0 &
```

```bash
# Motion vector model.
nohup python train.py --lr 0.01 --batch-size 40 --arch resnet18 \
 	--data-name ucf101 --representation mv \
 	--data-root data/ucf101/mpeg4_videos \
 	--train-list data/datalists/ucf101_split1_train.txt \
 	--test-list data/datalists/ucf101_split1_test.txt \
 	--model-prefix ucf101_mv_model \
 	--lr-steps 150 270 390  --epochs 510 \
 	--gpus 0 &
```

#### Training using stacked MVs

To train motion vector models  using multiple stacked motion vectors, add a parameter `--mv_stack_size` in the bash command. The default value is 1. Take 5 stacked mvs as an example, 

```bash
python train.py --lr 0.01 --batch-size 40 --arch resnet18 \
 	--data-name ucf101 --representation mv --mv_stack_size 5\
 	--data-root data/ucf101/mpeg4_videos \
 	--train-list data/datalists/ucf101_split1_train.txt \
 	--test-list data/datalists/ucf101_split1_test.txt \
 	--model-prefix ucf101_mv_model \
 	--lr-steps 150 270 390  --epochs 510 \
 	--gpus 0 
```

### Testing

Given a trained model, you will want to follow the following steps to test it.

1. Specify `anaconda` in your bash shell.

```bash
eval "$(${ANACONDA_PATH}/bin/conda shell.bash hook)"
```

2. Activate CoViAR virtual environment.

```bash
conda activate coviar
```

3. Navigate to CoViAR directory.

```bash
cd ${COVIAR_PATH}
```

4.  Use the following commands to test given` I-frame model` and `Motion vector model` on dataset `UCF-101`.

```bash
# I-frame model.
nohup python test.py --gpus 0 \
	--arch resnet152 --test-crops 1 \
	--data-name ucf101 --representation iframe \
	--data-root data/ucf101/mpeg4_videos \
	--test-list data/datalists/ucf101_split1_test.txt \
	--weights ${MODEL_PATH} \
	--save-scores ${SCORE_FILE_PATH} &
```

```bash
# Motion vector model.
nohup python test.py --gpus 0 \
	--arch resnet18 --test-crops 1 \
	--data-name ucf101 --representation mv \
	--data-root data/ucf101/mpeg4_videos \
	--test-list data/datalists/ucf101_split1_test.txt \
	--weights ${MODEL_PATH} \
	--save-scores ${SCORE_FILE_PATH} &
```

#### Testing using stacked MVs

To test motion vector models  using multiple stacked motion vectors, add a parameter `--mv_stack_size` in the bash command. The default value is 1. Take 5 stacked mvs as an example, 


```bash
python test.py --gpus 0 \
	--arch resnet18 --test-crops 1 \
	--data-name ucf101 --representation mv --mv_stack_size 5 \
	--data-root data/ucf101/mpeg4_videos \
	--test-list data/datalists/ucf101_split1_test.txt \
	--weights ${MODEL_PATH} \
	--save-scores ${SCORE_FILE_PATH}
```
### Combining Model Scores

After getting the evaluation results for each decoupled model using `test.py`, we use `combine.py` to combine the results and calculate the final accuracy.

```bash
python combine.py --iframe ${IFRAME_SCORE_FILE_PATH} \
	--mv ${MV_SCORE_FILE_PATH} \
	--res ${RESIDUAL_SCORE_FILE_PATH}
```

## Testing Results

